"""FastAPI application that serves the trained autoencoders."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import AsyncIterator

import numpy as np
import pandas as pd
import uvicorn
import wandb
from fastapi import Depends, FastAPI, Request
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
from generative_models.serving.model_service import (
    AutoencoderModelService,
)
from generative_models.serving.predict_autoencoder import predict_period

PRELOAD_ENVIRONMENT_VARIABLE = "AUTOENCODER_PRELOAD_MODEL"
DEFAULT_HISTORY_DAYS = 365

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
Detects outlier windows in the log-return series of an asset, using
the dense autoencoders trained by the training pipeline and registered
as wandb artifacts.

### How a model is chosen

Every prediction route accepts an optional `ticker`. When it is
omitted, the default model is used: the most recently created artifact
of the project, loaded when the service starts. Each model is
downloaded only once per process and then kept in memory.

### How a window is judged

The reconstruction error of a window is compared with the
`anomaly_threshold` stored in the artifact metadata at training time.
The threshold is never recomputed from the data being scored.

### Environment variables

| Variable | What it sets | Default |
|---|---|---|
| `AUTOENCODER_WANDB_PROJECT` | wandb project read by the registry | the training project |
| `AUTOENCODER_WANDB_ENTITY` | wandb entity owning the project | default entity of the account |
| `AUTOENCODER_DEVICE` | device used for inference | `cuda` when available |
| `AUTOENCODER_PRELOAD_MODEL` | set to `false` to skip loading the default model at startup | `true` |
| `WANDB_API_KEY` | wandb credentials, read from a local `.env` file | - |
"""

TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Liveness of the service and models in memory.",
    },
    {
        "name": "Models",
        "description": (
            "Models available in wandb, models already loaded and "
            "reloading without restarting the service."
        ),
    },
    {
        "name": "Predictions",
        "description": (
            "Outlier detection over log-returns, prices or market "
            "data downloaded on the fly."
        ),
    },
]


def get_model_service() -> AutoencoderModelService:
    """Return the model service shared by every route.

    Returns
    -------
    AutoencoderModelService
        The singleton service of this process. Tests replace this
        dependency through `app.dependency_overrides`.
    """
    return AutoencoderModelService.get_instance()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the default model when the application starts.

    Parameters
    ----------
    application : FastAPI
        Application being started, used to honour any dependency
        override installed by the tests.

    Yields
    ------
    None
        Control is handed back to the server once the preload attempt
        has finished. A failure is only logged, so that the API still
        answers when wandb is unreachable.
    """
    preload_setting = os.getenv(PRELOAD_ENVIRONMENT_VARIABLE, "true")

    if preload_setting.lower() != "false":
        service_provider = application.dependency_overrides.get(
            get_model_service,
            get_model_service,
        )
        try:
            service_provider().get_predictor()
        # pylint: disable-next=broad-exception-caught
        except Exception as preload_error:
            logger.warning(
                "Could not preload the default model: %s",
                preload_error,
            )

    yield


app = FastAPI(
    title="Autoencoder Outlier Detection API",
    summary=(
        "Detects outlier windows in financial log-returns with "
        "trained dense autoencoders."
    ),
    description=API_DESCRIPTION,
    version="0.1.0",
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
)


def build_model_metadata_response(
    metadata: AutoencoderModelMetadata,
) -> ModelMetadataResponse:
    """Turn model metadata into its API representation.

    Parameters
    ----------
    metadata : AutoencoderModelMetadata
        Metadata of a loaded model.

    Returns
    -------
    ModelMetadataResponse
        The same values, ready to be serialized.
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
    """Turn a prediction into its API representation.

    Parameters
    ----------
    metadata : AutoencoderModelMetadata
        Metadata of the model that produced the scores.
    prediction : AutoencoderPrediction
        Scores, outlier flags and window end indices.
    prediction_table : pd.DataFrame | None
        Table returned by `predict_period`, used to fill the date and
        the closing price of each window. When `None`, both fields are
        left empty.

    Returns
    -------
    PredictionResponse
        One entry per window plus the outlier counts of the series.

    Example
    -------
    >>> response = build_prediction_response(metadata, prediction)
    >>> response.number_of_windows
    8
    """
    window_dates: list[date | None] = [None] * len(
        prediction.window_end_indices
    )
    close_prices: list[float | None] = list(window_dates)

    if prediction_table is not None:
        window_dates = [
            pd.Timestamp(table_index).date()
            for table_index in prediction_table.index
        ]
        close_prices = [
            float(close_price)
            for close_price in prediction_table["close_price"]
        ]

    windows = [
        WindowPredictionResponse(
            window_end_index=int(window_end_index),
            window_end_date=window_date,
            close_price=close_price,
            reconstruction_score=float(reconstruction_score),
            is_outlier=bool(is_outlier),
        )
        for (
            window_end_index,
            window_date,
            close_price,
            reconstruction_score,
            is_outlier,
        ) in zip(
            prediction.window_end_indices,
            window_dates,
            close_prices,
            prediction.reconstruction_scores,
            prediction.is_outlier,
        )
    ]

    number_of_windows = len(windows)
    number_of_outliers = int(prediction.is_outlier.sum())

    return PredictionResponse(
        model=build_model_metadata_response(metadata),
        number_of_windows=number_of_windows,
        number_of_outliers=number_of_outliers,
        outlier_rate=(
            number_of_outliers / number_of_windows
            if number_of_windows
            else 0.0
        ),
        windows=windows,
    )


@app.exception_handler(LookupError)
async def handle_lookup_error(
    _request: Request,
    exception: LookupError,
) -> JSONResponse:
    """Answer 404 when a model or artifact could not be found.

    Parameters
    ----------
    _request : Request
        Request being served, required by the FastAPI contract.
    exception : LookupError
        Error raised by the service or the registry. A `KeyError` is
        unwrapped, so that the quotes it adds do not reach the client.

    Returns
    -------
    JSONResponse
        Response with status 404 and the message of the error.
    """
    if isinstance(exception, KeyError) and exception.args:
        detail = str(exception.args[0])
    else:
        detail = str(exception)

    return JSONResponse(status_code=404, content={"detail": detail})


@app.exception_handler(ValueError)
async def handle_value_error(
    _request: Request,
    exception: ValueError,
) -> JSONResponse:
    """Answer 422 when the request cannot be served as given.

    Parameters
    ----------
    _request : Request
        Request being served, required by the FastAPI contract.
    exception : ValueError
        Error raised by the predictor or by the route validation.

    Returns
    -------
    JSONResponse
        Response with status 422 and the message of the error.
    """
    return JSONResponse(
        status_code=422,
        content={"detail": str(exception)},
    )


@app.exception_handler(wandb.errors.CommError)
async def handle_wandb_communication_error(
    _request: Request,
    exception: wandb.errors.CommError,
) -> JSONResponse:
    """Answer 404 or 503 when wandb cannot serve the artifact.

    Parameters
    ----------
    _request : Request
        Request being served, required by the FastAPI contract.
    exception : wandb.errors.CommError
        Error raised by the wandb client.

    Returns
    -------
    JSONResponse
        Status 404 when the message reports a missing artifact, and
        503 when wandb itself could not be reached.
    """
    error_message = str(exception)

    if "not found" in error_message.lower():
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"Model artifact not found: {error_message}"
            },
        )

    return JSONResponse(
        status_code=503,
        content={"detail": f"wandb is unavailable: {error_message}"},
    )


@app.get(
    "/health",
    tags=["Health"],
    summary="Check whether the service is up",
    response_model=HealthResponse,
)
async def read_health(
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> HealthResponse:
    """Report the service status and the models already in memory.

    This route never contacts wandb, so it stays fast and answers even
    when the tracking service is unreachable.
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
    summary="List the models held in memory",
    response_model=list[ModelMetadataResponse],
)
async def read_loaded_models(
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> list[ModelMetadataResponse]:
    """List the metadata of every model already downloaded."""
    return [
        build_model_metadata_response(metadata)
        for metadata in model_service.list_loaded_models()
    ]


@app.get(
    "/models/available",
    tags=["Models"],
    summary="List the models registered in wandb",
    response_model=list[AvailableModelResponse],
    responses={
        503: {
            "model": ErrorResponse,
            "description": "wandb could not be reached.",
        },
    },
)
async def read_available_models(
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> list[AvailableModelResponse]:
    """List the servable artifacts of the project, newest first.

    Nothing is downloaded: only the artifact metadata is read.
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
    summary="Describe the default model",
    response_model=ModelMetadataResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "No servable model was found.",
        },
        503: {
            "model": ErrorResponse,
            "description": "wandb could not be reached.",
        },
    },
)
async def read_default_model(
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> ModelMetadataResponse:
    """Describe the model served when a request omits the ticker."""
    predictor = model_service.get_predictor()

    return build_model_metadata_response(predictor.metadata)


@app.post(
    "/models/reload",
    tags=["Models"],
    summary="Download a model again",
    response_model=ModelMetadataResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "No artifact was found for the ticker.",
        },
        503: {
            "model": ErrorResponse,
            "description": "wandb could not be reached.",
        },
    },
)
async def reload_model(
    reload_request: ReloadRequest,
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> ModelMetadataResponse:
    """Fetch the latest version of a model without a restart.

    Without a ticker, the most recent model of the project is loaded
    and becomes the new default.
    """
    predictor = model_service.reload(reload_request.ticker)

    return build_model_metadata_response(predictor.metadata)


@app.post(
    "/predictions/log-returns",
    tags=["Predictions"],
    summary="Detect outliers in a log-return series",
    response_model=PredictionResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "No model was found for the ticker.",
        },
        422: {
            "model": ErrorResponse,
            "description": "The series is shorter than the window.",
        },
    },
)
async def predict_log_returns(
    prediction_request: LogReturnPredictionRequest,
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Score every window of a log-return series sent by the client."""
    predictor = model_service.get_predictor(prediction_request.ticker)
    prediction = predictor.predict(
        np.asarray(prediction_request.log_returns, dtype=np.float32)
    )

    return build_prediction_response(predictor.metadata, prediction)


@app.post(
    "/predictions/prices",
    tags=["Predictions"],
    summary="Detect outliers in a price series",
    response_model=PredictionResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "No model was found for the ticker.",
        },
        422: {
            "model": ErrorResponse,
            "description": "The series is shorter than the window.",
        },
    },
)
async def predict_prices(
    prediction_request: PricePredictionRequest,
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Score every window of a price series sent by the client.

    The prices are turned into log-returns first, so `m` prices
    produce `m - 1` log-returns.
    """
    predictor = model_service.get_predictor(prediction_request.ticker)
    prediction = predictor.predict_from_prices(
        np.asarray(prediction_request.prices, dtype=np.float32)
    )

    return build_prediction_response(predictor.metadata, prediction)


@app.get(
    "/predictions/market",
    tags=["Predictions"],
    summary="Detect outliers in downloaded market data",
    response_model=PredictionResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "No model was found for the ticker.",
        },
        422: {
            "model": ErrorResponse,
            "description": "The period is empty or shorter than the window.",
        },
        503: {
            "model": ErrorResponse,
            "description": "wandb could not be reached.",
        },
    },
)
async def predict_market_data(
    ticker: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Download a period of market data and score every window.

    Without dates, the last year of data is used. Each window carries
    the date and the closing price that close it.
    """
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=DEFAULT_HISTORY_DAYS)

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
    """Serve the application with uvicorn.

    Parameters
    ----------
    host : str
        Address the server binds to.
    port : int
        TCP port the server listens on.
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
