"""FastAPI application that serves the trained autoencoders.

Run the API locally with::

    uv run uvicorn generative_models.serving.api:app --reload

or::

    uv run python -m generative_models.serving.api

The interactive Swagger documentation is then available at
`http://127.0.0.1:8000/docs` and the ReDoc version at
`http://127.0.0.1:8000/redoc`.
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Annotated

import numpy as np
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, Query, Request, status
from fastapi.responses import JSONResponse
from wandb.errors import CommError

from generative_models.serving.api_schemas import (
    AvailableModelResponse,
    ErrorResponse,
    HealthResponse,
    LogReturnPredictionRequest,
    ModelMetadataResponse,
    PredictionResponse,
    PricePredictionRequest,
    ReloadRequest,
    WindowPredictionResponse,
)
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPrediction,
)
from generative_models.serving.model_service import AutoencoderModelService
from generative_models.serving.predict_autoencoder import predict_period

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
Outlier detection on daily stock log-returns with the `SimpleAutoencoder`
models trained by `AutoencoderTrainer` and stored in **Weights & Biases**.

## How it works

1. Models are downloaded from wandb **once** and kept in memory by a
   process-wide singleton (`AutoencoderModelService`). Every request
   reuses the in-memory model.
2. Input log-returns are normalized with the training statistics stored
   in the model artifact and split into sliding windows of
   `window_size` returns.
3. Each window is reconstructed by the autoencoder. Its mean squared
   reconstruction error is the **anomaly score**, and windows whose
   score exceeds the model `anomaly_threshold` are flagged as
   **outliers**.

## Model selection

Every prediction endpoint accepts an optional `ticker`. When it is
omitted, the **default model** is used: the most recently trained
autoencoder at the time it was first loaded. Use `POST /models/reload`
to pick up newly trained artifacts without restarting the service.

## Configuration

Environment variables (a local `.env` file is also read):

| Variable | Description |
|---|---|
| `WANDB_API_KEY` | wandb API key used to download the artifacts. |
| `AUTOENCODER_WANDB_PROJECT` | wandb project with the model artifacts. |
| `AUTOENCODER_WANDB_ENTITY` | wandb entity (defaults to the key owner). |
| `AUTOENCODER_DEVICE` | Torch device (`cpu`, `cuda`). |
| `AUTOENCODER_PRELOAD_MODEL` | `false` to skip loading the default model at startup. |
"""

OPENAPI_TAGS = [
    {
        "name": "Health",
        "description": "Service liveness and in-memory model status.",
    },
    {
        "name": "Models",
        "description": (
            "Inspect the models held in memory or available in wandb, "
            "and reload them."
        ),
    },
    {
        "name": "Predictions",
        "description": (
            "Score log-return windows and flag outliers, from "
            "user-provided data or from market data downloaded on "
            "demand."
        ),
    },
]

ERROR_RESPONSES = {
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "No servable model was found for the ticker.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "wandb could not be reached to load the model.",
    },
}

PREDICTION_ERROR_RESPONSES = {
    **ERROR_RESPONSES,
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": (
            "Invalid input, for example fewer values than the model "
            "window size."
        ),
    },
}


def get_model_service() -> AutoencoderModelService:
    """Provide the singleton model service to the endpoints.

    Returns
    -------
    AutoencoderModelService
        The process-wide service instance.
    """
    return AutoencoderModelService.get_instance()


ModelServiceDependency = Annotated[
    AutoencoderModelService,
    Depends(get_model_service),
]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Preload the default model when the application starts.

    A failure (for example, no network access) is logged and the
    model is loaded lazily on the first request instead.

    Parameters
    ----------
    application : FastAPI
        The application being started.

    Returns
    -------
    AsyncIterator[None]
        Context that keeps the application running.
    """
    if os.getenv("AUTOENCODER_PRELOAD_MODEL", "true").lower() != "false":
        model_service = application.dependency_overrides.get(
            get_model_service,
            get_model_service,
        )()
        try:
            model_service.get_predictor()
        # pylint: disable-next=broad-exception-caught
        except Exception as preload_error:
            logger.warning(
                "Default model not preloaded (%s); it will be loaded on "
                "the first request.",
                preload_error,
            )
    yield


app = FastAPI(
    title="Financial Market Simulator - Autoencoder Serving API",
    summary="Outlier detection on stock log-returns with autoencoders.",
    description=API_DESCRIPTION,
    version="0.1.0",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)


@app.exception_handler(LookupError)
async def handle_lookup_error(
    _request: Request,
    error: LookupError,
) -> JSONResponse:
    """Translate missing models into HTTP 404 responses.

    Parameters
    ----------
    _request : Request
        Request that raised the error.
    error : LookupError
        Raised error, including `KeyError`.

    Returns
    -------
    JSONResponse
        Response with status 404 and the error message.
    """
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(error).strip("'\"")},
    )


@app.exception_handler(ValueError)
async def handle_value_error(
    _request: Request,
    error: ValueError,
) -> JSONResponse:
    """Translate invalid prediction inputs into HTTP 422 responses.

    Parameters
    ----------
    _request : Request
        Request that raised the error.
    error : ValueError
        Raised error.

    Returns
    -------
    JSONResponse
        Response with status 422 and the error message.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": str(error)},
    )


@app.exception_handler(CommError)
async def handle_wandb_error(
    _request: Request,
    error: CommError,
) -> JSONResponse:
    """Translate wandb errors into HTTP 404 or 503 responses.

    Parameters
    ----------
    _request : Request
        Request that raised the error.
    error : CommError
        Error raised by the wandb client.

    Returns
    -------
    JSONResponse
        Status 404 when the artifact does not exist, 503 otherwise.
    """
    if "not found" in str(error).lower():
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": f"Model artifact not found: {error}"},
        )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"wandb is unavailable: {error}"},
    )


def build_model_metadata_response(
    metadata: AutoencoderModelMetadata,
) -> ModelMetadataResponse:
    """Convert predictor metadata into its API representation.

    Parameters
    ----------
    metadata : AutoencoderModelMetadata
        Metadata of a loaded predictor.

    Returns
    -------
    ModelMetadataResponse
        Serializable copy of `metadata`.
    """
    return ModelMetadataResponse(
        ticker=metadata.ticker,
        artifact_name=metadata.artifact_name,
        window_size=metadata.window_size,
        return_train_mean=metadata.return_train_mean,
        return_train_std=metadata.return_train_std,
        anomaly_threshold=metadata.anomaly_threshold,
    )


def build_prediction_response(
    metadata: AutoencoderModelMetadata,
    prediction: AutoencoderPrediction,
    prediction_table: pd.DataFrame | None = None,
) -> PredictionResponse:
    """Assemble the prediction response from the model outputs.

    Parameters
    ----------
    metadata : AutoencoderModelMetadata
        Metadata of the predictor that produced `prediction`.
    prediction : AutoencoderPrediction
        Per-window scores, outlier flags and end indices.
    prediction_table : pd.DataFrame | None
        Optional table from `build_prediction_table`, used to fill in
        the date and closing price of each window.

    Returns
    -------
    PredictionResponse
        Summary statistics and per-window predictions.
    """
    window_end_dates = [None] * len(prediction.reconstruction_scores)
    close_prices = [None] * len(prediction.reconstruction_scores)
    if prediction_table is not None:
        window_end_dates = [
            timestamp.date() for timestamp in prediction_table.index
        ]
        close_prices = prediction_table["close_price"].tolist()

    windows = [
        WindowPredictionResponse(
            window_end_index=int(window_end_index),
            window_end_date=window_end_date,
            close_price=close_price,
            reconstruction_score=float(reconstruction_score),
            is_outlier=bool(is_outlier),
        )
        for (
            window_end_index,
            window_end_date,
            close_price,
            reconstruction_score,
            is_outlier,
        ) in zip(
            prediction.window_end_indices,
            window_end_dates,
            close_prices,
            prediction.reconstruction_scores,
            prediction.is_outlier,
        )
    ]
    number_of_outliers = int(prediction.is_outlier.sum())
    return PredictionResponse(
        model=build_model_metadata_response(metadata),
        number_of_windows=len(windows),
        number_of_outliers=number_of_outliers,
        outlier_rate=number_of_outliers / len(windows),
        windows=windows,
    )


@app.get(
    "/health",
    tags=["Health"],
    summary="Check service health",
    response_model=HealthResponse,
)
def get_health(model_service: ModelServiceDependency) -> HealthResponse:
    """Report that the service is alive and which models are loaded.

    This endpoint never contacts wandb, so it is cheap enough for
    liveness probes.
    """
    return HealthResponse(
        status="ok",
        default_ticker=model_service.default_ticker,
        loaded_tickers=[
            metadata.ticker
            for metadata in model_service.list_loaded_models()
        ],
    )


@app.get(
    "/models/loaded",
    tags=["Models"],
    summary="List models held in memory",
    response_model=list[ModelMetadataResponse],
)
def list_loaded_models(
    model_service: ModelServiceDependency,
) -> list[ModelMetadataResponse]:
    """Return the metadata of every model already loaded in memory.

    Models are loaded at startup (default model) or on the first
    request that needs them.
    """
    return [
        build_model_metadata_response(metadata)
        for metadata in model_service.list_loaded_models()
    ]


@app.get(
    "/models/available",
    tags=["Models"],
    summary="List servable models stored in wandb",
    response_model=list[AvailableModelResponse],
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: ERROR_RESPONSES[
            status.HTTP_503_SERVICE_UNAVAILABLE
        ]
    },
)
def list_available_models(
    model_service: ModelServiceDependency,
) -> list[AvailableModelResponse]:
    """Return the latest servable artifact of each ticker in wandb.

    Only artifacts logged by `AutoencoderTrainer`, which include the
    normalization statistics and the anomaly threshold, are listed.
    The newest artifact comes first. Nothing is downloaded.
    """
    return [
        AvailableModelResponse(
            artifact_name=available_model.artifact_name,
            ticker=available_model.ticker,
            created_at=available_model.created_at,
        )
        for available_model in model_service.list_available_models()
    ]


@app.get(
    "/models/default",
    tags=["Models"],
    summary="Get the default model",
    response_model=ModelMetadataResponse,
    responses=ERROR_RESPONSES,
)
def get_default_model(
    model_service: ModelServiceDependency,
) -> ModelMetadataResponse:
    """Return the default model, loading it from wandb if needed.

    The default model is the most recently trained autoencoder at the
    moment it was first loaded.
    """
    return build_model_metadata_response(
        model_service.get_predictor().metadata
    )


@app.post(
    "/models/reload",
    tags=["Models"],
    summary="Reload a model from wandb",
    response_model=ModelMetadataResponse,
    responses=ERROR_RESPONSES,
)
def reload_model(
    model_service: ModelServiceDependency,
    reload_request: ReloadRequest,
) -> ModelMetadataResponse:
    """Download the latest artifact again and replace it in memory.

    - Without `ticker`: loads the most recently trained model of any
      ticker and makes it the new **default model**.
    - With `ticker`: reloads the latest model of that ticker.

    Use it after training a new model to serve it without restarting
    the API.
    """
    return build_model_metadata_response(
        model_service.reload(reload_request.ticker).metadata
    )


@app.post(
    "/predictions/log-returns",
    tags=["Predictions"],
    summary="Detect outliers in a log-return series",
    response_model=PredictionResponse,
    responses=PREDICTION_ERROR_RESPONSES,
)
def predict_from_log_returns(
    model_service: ModelServiceDependency,
    prediction_request: LogReturnPredictionRequest,
) -> PredictionResponse:
    """Score every sliding window of the given raw log-returns.

    The series must have at least `window_size` values (see
    `GET /models/default`). One prediction is returned per window,
    identified by the index of its last return.
    """
    predictor = model_service.get_predictor(prediction_request.ticker)
    prediction = predictor.predict(
        np.asarray(prediction_request.log_returns)
    )
    return build_prediction_response(predictor.metadata, prediction)


@app.post(
    "/predictions/prices",
    tags=["Predictions"],
    summary="Detect outliers from a closing price series",
    response_model=PredictionResponse,
    responses=PREDICTION_ERROR_RESPONSES,
)
def predict_from_prices(
    model_service: ModelServiceDependency,
    prediction_request: PricePredictionRequest,
) -> PredictionResponse:
    """Convert the prices into log-returns and score every window.

    The series must have at least `window_size + 1` prices. Window end
    index `i` refers to the return between prices `i` and `i + 1`.
    """
    predictor = model_service.get_predictor(prediction_request.ticker)
    prediction = predictor.predict_from_prices(
        np.asarray(prediction_request.prices)
    )
    return build_prediction_response(predictor.metadata, prediction)


@app.get(
    "/predictions/market",
    tags=["Predictions"],
    summary="Detect outliers in recent market data",
    response_model=PredictionResponse,
    responses=PREDICTION_ERROR_RESPONSES,
)
def predict_from_market_data(
    model_service: ModelServiceDependency,
    ticker: Annotated[
        str | None,
        Query(
            description=(
                "Ticker whose model and prices are used "
                "(case-insensitive). Defaults to the ticker of the "
                "default model."
            ),
            examples=["PETR4.SA"],
        ),
    ] = None,
    start_date: Annotated[
        date | None,
        Query(
            description=(
                "First day of the period (`YYYY-MM-DD`). Defaults to "
                "one year before `end_date`."
            ),
            examples=["2026-01-01"],
        ),
    ] = None,
    end_date: Annotated[
        date | None,
        Query(
            description="Last day of the period. Defaults to today.",
            examples=["2026-10-06"],
        ),
    ] = None,
) -> PredictionResponse:
    """Download the daily closing prices of the model ticker and score them.

    Prices are fetched from Yahoo Finance with the same
    `predict_period` function used by the `predict_autoencoder`
    script. Each window comes with the date and closing price of its
    last day.
    """
    period_end_date = end_date or date.today()
    period_start_date = start_date or period_end_date - timedelta(days=365)
    if period_start_date >= period_end_date:
        raise ValueError("start_date must be earlier than end_date.")

    predictor = model_service.get_predictor(ticker)
    prediction_table = predict_period(
        predictor,
        period_start_date.isoformat(),
        period_end_date.isoformat(),
    )
    number_of_windows = len(prediction_table)
    prediction = AutoencoderPrediction(
        reconstruction_scores=prediction_table[
            "reconstruction_score"
        ].to_numpy(),
        is_outlier=prediction_table["is_outlier"].to_numpy(),
        window_end_indices=np.arange(
            predictor.metadata.window_size - 1,
            predictor.metadata.window_size - 1 + number_of_windows,
        ),
    )
    return build_prediction_response(
        predictor.metadata,
        prediction,
        prediction_table,
    )


def main(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Start the API with uvicorn.

    Parameters
    ----------
    host : str
        Network interface to bind. Defaults to `"127.0.0.1"`.
    port : int
        TCP port to listen on. Defaults to `8000`.
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
