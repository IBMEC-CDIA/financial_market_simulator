"""REST API that serves the autoencoders registered in wandb.

Example
-------
uv run uvicorn generative_models.serving.api:app --reload
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
import wandb
from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

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
Detects outliers in daily log-returns with `SimpleAutoencoder` models
trained by `train_autoencoder` and registered as wandb artifacts.

Each request is scored by normalizing the log-returns with the training
statistics, building every sliding window of `window_size` values and
flagging the windows whose reconstruction error is above the anomaly
threshold saved during training.

**Model selection:** every prediction route accepts an optional
`ticker`. When it is given, the latest artifact of that ticker is used
(and downloaded on the first request). When it is omitted, the default
model is used: the most recently registered artifact of any ticker,
loaded at startup or by `POST /models/reload` without a ticker.

**Environment variables:**

| Variable | Purpose | Default |
|---|---|---|
| `AUTOENCODER_WANDB_PROJECT` | wandb project of the models | training one |
| `AUTOENCODER_WANDB_ENTITY` | wandb entity of the project | API key entity |
| `AUTOENCODER_DEVICE` | Torch device of the models | `cuda` if available |
| `AUTOENCODER_PRELOAD_MODEL` | `false` skips loading at startup | `true` |
| `WANDB_API_KEY` | wandb API key, read from `.env` | - |
"""

NOT_FOUND_RESPONSE = {
    404: {"model": ErrorResponse, "description": "Model not found."}
}
UNAVAILABLE_RESPONSE = {
    503: {"model": ErrorResponse, "description": "wandb is unavailable."}
}
INVALID_INPUT_RESPONSE = {
    422: {"model": ErrorResponse, "description": "Invalid input data."}
}


def get_model_service() -> AutoencoderModelService:
    """Return the process-wide model service.

    Returns
    -------
    AutoencoderModelService
        Singleton shared by every request.
    """
    return AutoencoderModelService.get_instance()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the default model when the API starts.

    Loading is skipped when `AUTOENCODER_PRELOAD_MODEL` is `"false"`,
    and a loading failure is only logged, so the API always starts.

    Parameters
    ----------
    application : FastAPI
        Application being started.

    Returns
    -------
    AsyncIterator[None]
        Generator that yields while the application runs.
    """
    preload_model = os.getenv("AUTOENCODER_PRELOAD_MODEL", "true")
    if preload_model.lower() != "false":
        model_service_provider = application.dependency_overrides.get(
            get_model_service,
            get_model_service,
        )
        try:
            model_service_provider().get_predictor()
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.warning("Could not preload the default model: %s", error)
    yield


app = FastAPI(
    title="Autoencoder Outlier Detection API",
    summary=(
        "Outlier detection in daily log-returns with autoencoders "
        "registered in wandb."
    ),
    description=API_DESCRIPTION,
    version="0.1.0",
    openapi_tags=[
        {"name": "Health", "description": "Service status."},
        {
            "name": "Models",
            "description": "Models in memory and in wandb.",
        },
        {
            "name": "Predictions",
            "description": "Outlier detection over log-return windows.",
        },
    ],
    lifespan=lifespan,
)

ModelServiceDependency = Annotated[
    AutoencoderModelService,
    Depends(get_model_service),
]


def build_model_metadata_response(
    metadata: AutoencoderModelMetadata,
) -> ModelMetadataResponse:
    """Convert model metadata into its API response.

    Parameters
    ----------
    metadata : AutoencoderModelMetadata
        Metadata of a predictor.

    Returns
    -------
    ModelMetadataResponse
        Response with the same fields.
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
    """Convert a prediction into its API response.

    Parameters
    ----------
    metadata : AutoencoderModelMetadata
        Metadata of the predictor that made the prediction.
    prediction : AutoencoderPrediction
        Scores, flags and window end indices.
    prediction_table : pd.DataFrame | None
        Table of `build_prediction_table`, aligned with `prediction`.
        When given, its dates and `close_price` fill each window.

    Returns
    -------
    PredictionResponse
        One window entry per scored window, in chronological order,
        plus the outlier counts.

    Example
    -------
    >>> prediction = predictor.predict(log_return_values)
    >>> response = build_prediction_response(
    ...     predictor.metadata,
    ...     prediction,
    ... )
    >>> response.number_of_windows == len(prediction.is_outlier)
    True
    """
    window_responses = []
    for window_position, window_end_index in enumerate(
        prediction.window_end_indices
    ):
        window_end_date = None
        close_price = None
        if prediction_table is not None:
            window_end_date = pd.Timestamp(
                prediction_table.index[window_position]
            ).date()
            close_price = float(
                prediction_table["close_price"].iloc[window_position]
            )
        window_responses.append(
            WindowPredictionResponse(
                window_end_index=int(window_end_index),
                window_end_date=window_end_date,
                close_price=close_price,
                reconstruction_score=float(
                    prediction.reconstruction_scores[window_position]
                ),
                is_outlier=bool(prediction.is_outlier[window_position]),
            )
        )

    number_of_windows = len(window_responses)
    number_of_outliers = int(np.sum(prediction.is_outlier))
    return PredictionResponse(
        model=build_model_metadata_response(metadata),
        number_of_windows=number_of_windows,
        number_of_outliers=number_of_outliers,
        outlier_rate=number_of_outliers / number_of_windows,
        windows=window_responses,
    )


@app.exception_handler(LookupError)
async def handle_lookup_error(
    _: Request,
    error: LookupError,
) -> JSONResponse:
    """Translate a missing model into an HTTP 404 response.

    Parameters
    ----------
    _ : Request
        Request that raised the error.
    error : LookupError
        Raised error, including `KeyError`.

    Returns
    -------
    JSONResponse
        404 response whose detail is the error message.
    """
    detail = str(error.args[0]) if error.args else str(error)
    return JSONResponse(status_code=404, content={"detail": detail})


@app.exception_handler(ValueError)
async def handle_value_error(
    _: Request,
    error: ValueError,
) -> JSONResponse:
    """Translate invalid input data into an HTTP 422 response.

    Parameters
    ----------
    _ : Request
        Request that raised the error.
    error : ValueError
        Raised error.

    Returns
    -------
    JSONResponse
        422 response whose detail is the error message.
    """
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.exception_handler(wandb.errors.CommError)
async def handle_wandb_communication_error(
    _: Request,
    error: wandb.errors.CommError,
) -> JSONResponse:
    """Translate a wandb communication error into an HTTP response.

    Parameters
    ----------
    _ : Request
        Request that raised the error.
    error : wandb.errors.CommError
        Raised error.

    Returns
    -------
    JSONResponse
        404 response when the artifact was not found, 503 otherwise.
    """
    if "not found" in str(error).lower():
        return JSONResponse(
            status_code=404,
            content={"detail": f"Model artifact not found: {error}"},
        )
    return JSONResponse(
        status_code=503,
        content={"detail": f"wandb is unavailable: {error}"},
    )


@app.get(
    "/health",
    tags=["Health"],
    summary="Check the service status",
    response_model=HealthResponse,
)
def get_health(model_service: ModelServiceDependency) -> HealthResponse:
    """Report the default ticker and the models in memory.

    This route never accesses wandb.
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
    summary="List the models in memory",
    response_model=list[ModelMetadataResponse],
)
def list_loaded_models(
    model_service: ModelServiceDependency,
) -> list[ModelMetadataResponse]:
    """Return the metadata of every model already loaded."""
    return [
        build_model_metadata_response(metadata)
        for metadata in model_service.list_loaded_models()
    ]


@app.get(
    "/models/available",
    tags=["Models"],
    summary="List the servable models in wandb",
    response_model=list[AvailableModelResponse],
    responses=UNAVAILABLE_RESPONSE,
)
def list_available_models(
    model_service: ModelServiceDependency,
) -> list[AvailableModelResponse]:
    """Return the servable artifacts, newest first, without loading."""
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
    responses={**NOT_FOUND_RESPONSE, **UNAVAILABLE_RESPONSE},
)
def get_default_model(
    model_service: ModelServiceDependency,
) -> ModelMetadataResponse:
    """Return the default model, loading it if needed."""
    predictor = model_service.get_predictor()
    return build_model_metadata_response(predictor.metadata)


@app.post(
    "/models/reload",
    tags=["Models"],
    summary="Reload a model from wandb",
    response_model=ModelMetadataResponse,
    responses={**NOT_FOUND_RESPONSE, **UNAVAILABLE_RESPONSE},
)
def reload_model(
    reload_request: ReloadRequest,
    model_service: ModelServiceDependency,
) -> ModelMetadataResponse:
    """Download the latest artifact again, without restarting the API.

    Without a ticker, the most recent model of any ticker becomes the
    new default model.
    """
    predictor = model_service.reload(reload_request.ticker)
    return build_model_metadata_response(predictor.metadata)


@app.post(
    "/predictions/log-returns",
    tags=["Predictions"],
    summary="Detect outliers in log-returns",
    response_model=PredictionResponse,
    responses={
        **NOT_FOUND_RESPONSE,
        **INVALID_INPUT_RESPONSE,
        **UNAVAILABLE_RESPONSE,
    },
)
def predict_log_returns(
    prediction_request: LogReturnPredictionRequest,
    model_service: ModelServiceDependency,
) -> PredictionResponse:
    """Score every window of the given log-returns."""
    predictor = model_service.get_predictor(prediction_request.ticker)
    prediction = predictor.predict(
        np.array(prediction_request.log_returns)
    )
    return build_prediction_response(predictor.metadata, prediction)


@app.post(
    "/predictions/prices",
    tags=["Predictions"],
    summary="Detect outliers in prices",
    response_model=PredictionResponse,
    responses={
        **NOT_FOUND_RESPONSE,
        **INVALID_INPUT_RESPONSE,
        **UNAVAILABLE_RESPONSE,
    },
)
def predict_prices(
    prediction_request: PricePredictionRequest,
    model_service: ModelServiceDependency,
) -> PredictionResponse:
    """Score every window of the log-returns of the given prices."""
    predictor = model_service.get_predictor(prediction_request.ticker)
    prediction = predictor.predict_from_prices(
        np.array(prediction_request.prices)
    )
    return build_prediction_response(predictor.metadata, prediction)


@app.get(
    "/predictions/market",
    tags=["Predictions"],
    summary="Detect outliers in downloaded market data",
    response_model=PredictionResponse,
    responses={
        **NOT_FOUND_RESPONSE,
        **INVALID_INPUT_RESPONSE,
        **UNAVAILABLE_RESPONSE,
    },
)
def predict_market(
    model_service: ModelServiceDependency,
    ticker: Annotated[
        str | None,
        Query(description="Ticker of the model; default model if omitted."),
    ] = None,
    start_date: Annotated[
        date | None,
        Query(description="First date; end_date minus 365 days if omitted."),
    ] = None,
    end_date: Annotated[
        date | None,
        Query(description="Last date; today if omitted."),
    ] = None,
) -> PredictionResponse:
    """Download the closing prices of a period and score every window.

    Each window carries the date and closing price of its last price.
    """
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    if start_date >= end_date:
        raise ValueError("start_date must be earlier than end_date.")

    predictor = model_service.get_predictor(ticker)
    prediction_table = predict_period(
        predictor,
        start_date.isoformat(),
        end_date.isoformat(),
    )
    window_size = predictor.metadata.window_size
    prediction = AutoencoderPrediction(
        reconstruction_scores=prediction_table[
            "reconstruction_score"
        ].to_numpy(),
        is_outlier=prediction_table["is_outlier"].to_numpy(),
        window_end_indices=np.arange(
            window_size - 1,
            window_size - 1 + len(prediction_table),
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
        Network interface to bind.
    port : int
        Port to listen on.
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
