"""FastAPI application that serves the trained autoencoders over HTTP."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Annotated

import numpy as np
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from wandb.errors import CommError
from wandb.sdk.mailbox.mailbox import MailboxClosedError

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

DEFAULT_LOOKBACK_DAYS = 365
PRELOAD_ENVIRONMENT_VARIABLE = "AUTOENCODER_PRELOAD_MODEL"

logger = logging.getLogger(__name__)


def get_model_service() -> AutoencoderModelService:
    """Return the shared model service, used as a FastAPI dependency.

    Returns
    -------
    AutoencoderModelService
        The process-wide service. Tests replace it with
        `app.dependency_overrides`.
    """
    return AutoencoderModelService.get_instance()


ModelServiceDependency = Annotated[
    AutoencoderModelService,
    Depends(get_model_service),
]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the default model when the API starts.

    The preload is skipped when `AUTOENCODER_PRELOAD_MODEL` is `false`.
    A failure is logged and does not stop the API from starting.

    Parameters
    ----------
    application : FastAPI
        Application being started.

    Yields
    ------
    None
        Control returns to FastAPI while the application runs.
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
    summary="Detect outliers in market log-returns with autoencoders.",
    description=(
        "Each model is an autoencoder trained on the log-returns of one "
        "ticker and registered in Weights & Biases. A window is an outlier "
        "when its reconstruction score is above the anomaly threshold saved "
        "with the model.\n\n"
        "The default model is used when a request does not name a ticker. "
        "Models are downloaded once and kept in memory.\n\n"
        "Environment variables: `AUTOENCODER_WANDB_PROJECT` (wandb project), "
        "`AUTOENCODER_WANDB_ENTITY` (wandb entity), `AUTOENCODER_DEVICE` "
        "(`cpu` or `cuda`) and `AUTOENCODER_PRELOAD_MODEL` (`false` to skip "
        "loading the default model at startup)."
    ),
    version="0.1.0",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Status of the service and of the loaded models.",
        },
        {
            "name": "Models",
            "description": "Inspect, load and reload autoencoder models.",
        },
        {
            "name": "Predictions",
            "description": "Detect outlier windows in log-returns or prices.",
        },
    ],
    lifespan=lifespan,
)


def _error_message(error: Exception) -> str:
    """Return the message of an exception without Python's extra quotes.

    Parameters
    ----------
    error : Exception
        Exception raised by the service.

    Returns
    -------
    str
        First argument of the exception, or an empty string if it has
        none. `KeyError` wraps its message in quotes when converted with
        `str`, so the first argument is used instead.
    """
    if error.args:
        return str(error.args[0])

    return ""


@app.exception_handler(LookupError)
async def handle_lookup_error(
    _request: Request,
    error: LookupError,
) -> JSONResponse:
    """Translate lookup failures into HTTP 404 responses.

    Parameters
    ----------
    request : Request
        Request that raised the error.
    error : LookupError
        Error raised by the service or the registry.

    Returns
    -------
    JSONResponse
        Response with status 404 and an `ErrorResponse` body.
    """
    return JSONResponse(
        status_code=404,
        content={"detail": _error_message(error)},
    )


@app.exception_handler(ValueError)
async def handle_value_error(
    _request: Request,
    error: ValueError,
) -> JSONResponse:
    """Translate invalid input values into HTTP 422 responses.

    Parameters
    ----------
    request : Request
        Request that raised the error.
    error : ValueError
        Error raised by the service, the predictor or the routes.

    Returns
    -------
    JSONResponse
        Response with status 422 and an `ErrorResponse` body.
    """
    return JSONResponse(
        status_code=422,
        content={"detail": _error_message(error)},
    )


@app.exception_handler(CommError)
async def handle_wandb_error(
    _request: Request,
    error: CommError,
) -> JSONResponse:
    """Translate wandb communication errors into HTTP responses.

    A "not found" message becomes HTTP 404. Any other message becomes
    HTTP 503, because the wandb service is unavailable.

    Parameters
    ----------
    request : Request
        Request that raised the error.
    error : CommError
        Communication error raised by the wandb client.

    Returns
    -------
    JSONResponse
        Response with status 404 or 503 and an `ErrorResponse` body.
    """
    message = str(error)
    if "not found" in message.lower():
        return JSONResponse(
            status_code=404,
            content={"detail": f"Model artifact not found: {message}"},
        )

    return JSONResponse(
        status_code=503,
        content={"detail": f"wandb is unavailable: {message}"},
    )


@app.exception_handler(MailboxClosedError)
async def handle_wandb_connection_closed(
    _request: Request,
    _error: MailboxClosedError,
) -> JSONResponse:
    """Translate a closed wandb connection into HTTP 503.

    The wandb client is discarded, so the next request creates a new one
    and can succeed without restarting the API.

    Parameters
    ----------
    _request : Request
        Request that raised the error.
    _error : MailboxClosedError
        Error raised by wandb when its internal connection was closed.

    Returns
    -------
    JSONResponse
        Response with status 503 and an `ErrorResponse` body.
    """
    AutoencoderModelService.get_instance().registry.reset_api()

    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "The connection to wandb was closed. The client was reset; "
                "retry the request."
            ),
        },
    )


def build_model_metadata_response(
    metadata: AutoencoderModelMetadata,
) -> ModelMetadataResponse:
    """Convert the metadata of a predictor into its response schema.

    Parameters
    ----------
    metadata : AutoencoderModelMetadata
        Metadata of a predictor.

    Returns
    -------
    ModelMetadataResponse
        Response schema with the same values.
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
    """Build the response of a prediction, window by window.

    Parameters
    ----------
    metadata : AutoencoderModelMetadata
        Metadata of the model that made the prediction.
    prediction : AutoencoderPrediction
        Scores and outlier flags of the windows.
    prediction_table : pd.DataFrame | None
        Table with the `close_price` of each window, indexed by date, in
        the same order as the windows. When given, each window also gets
        its date and closing price.

    Returns
    -------
    PredictionResponse
        Response with one entry per window and the outlier summary.

    Example
    -------
    >>> response = build_prediction_response(metadata, prediction)
    >>> response.number_of_windows
    8
    """
    windows = []
    for position, window_end_index in enumerate(prediction.window_end_indices):
        window_end_date = None
        close_price = None
        if prediction_table is not None:
            window_end_date = prediction_table.index[position].date()
            close_price = float(
                prediction_table["close_price"].iloc[position]
            )
        windows.append(
            WindowPredictionResponse(
                window_end_index=int(window_end_index),
                window_end_date=window_end_date,
                close_price=close_price,
                reconstruction_score=float(
                    prediction.reconstruction_scores[position]
                ),
                is_outlier=bool(prediction.is_outlier[position]),
            )
        )

    number_of_windows = len(windows)
    number_of_outliers = int(np.sum(prediction.is_outlier))
    outlier_rate = (
        number_of_outliers / number_of_windows if number_of_windows else 0.0
    )

    return PredictionResponse(
        model=build_model_metadata_response(metadata),
        number_of_windows=number_of_windows,
        number_of_outliers=number_of_outliers,
        outlier_rate=outlier_rate,
        windows=windows,
    )


@app.get(
    "/health",
    tags=["Health"],
    summary="Check the service status",
    response_model=HealthResponse,
)
def get_health(service: ModelServiceDependency) -> HealthResponse:
    """Report the status, the default model and the models in memory.

    This route does not call wandb.
    """
    loaded_models = service.list_loaded_models()

    return HealthResponse(
        status="ok",
        default_ticker=service.default_ticker,
        loaded_tickers=[model.ticker for model in loaded_models],
    )


@app.get(
    "/models/loaded",
    tags=["Models"],
    summary="List the models in memory",
    response_model=list[ModelMetadataResponse],
)
def list_loaded_models(
    service: ModelServiceDependency,
) -> list[ModelMetadataResponse]:
    """List the metadata of the models currently loaded in memory."""
    return [
        build_model_metadata_response(metadata)
        for metadata in service.list_loaded_models()
    ]


@app.get(
    "/models/available",
    tags=["Models"],
    summary="List the models available in wandb",
    response_model=list[AvailableModelResponse],
    responses={
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def list_available_models(
    service: ModelServiceDependency,
) -> list[AvailableModelResponse]:
    """List the model artifacts that can be loaded, newest first.

    Nothing is downloaded.
    """
    return [
        AvailableModelResponse(
            artifact_name=model.artifact_name,
            ticker=model.ticker,
            created_at=model.created_at,
        )
        for model in service.list_available_models()
    ]


@app.get(
    "/models/default",
    tags=["Models"],
    summary="Get the default model",
    response_model=ModelMetadataResponse,
    responses={
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def get_default_model(
    service: ModelServiceDependency,
) -> ModelMetadataResponse:
    """Return the metadata of the default model, loading it if needed."""
    predictor = service.get_predictor()

    return build_model_metadata_response(predictor.metadata)


@app.post(
    "/models/reload",
    tags=["Models"],
    summary="Download a model again",
    response_model=ModelMetadataResponse,
    responses={
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def reload_model(
    request: ReloadRequest,
    service: ModelServiceDependency,
) -> ModelMetadataResponse:
    """Download a model again, replacing the one in memory.

    Without a ticker, the latest model of any ticker is loaded and
    becomes the default model.
    """
    predictor = service.reload(request.ticker)

    return build_model_metadata_response(predictor.metadata)


@app.post(
    "/predictions/log-returns",
    tags=["Predictions"],
    summary="Detect outliers in log-returns",
    response_model=PredictionResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict_from_log_returns(
    request: LogReturnPredictionRequest,
    service: ModelServiceDependency,
) -> PredictionResponse:
    """Score every window of a series of log-returns.

    The `window_end_date` and `close_price` fields are null in this
    response.
    """
    predictor = service.get_predictor(request.ticker)
    prediction = predictor.predict(request.log_returns)

    return build_prediction_response(predictor.metadata, prediction)


@app.post(
    "/predictions/prices",
    tags=["Predictions"],
    summary="Detect outliers in closing prices",
    response_model=PredictionResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict_from_prices(
    request: PricePredictionRequest,
    service: ModelServiceDependency,
) -> PredictionResponse:
    """Compute the log-returns of the prices and score their windows.

    The `window_end_date` and `close_price` fields are null in this
    response.
    """
    predictor = service.get_predictor(request.ticker)
    prediction = predictor.predict_from_prices(request.prices)

    return build_prediction_response(predictor.metadata, prediction)


@app.get(
    "/predictions/market",
    tags=["Predictions"],
    summary="Detect outliers in market data downloaded now",
    response_model=PredictionResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict_from_market_data(
    service: ModelServiceDependency,
    ticker: Annotated[
        str | None,
        Query(
            description=(
                "Ticker of the model to use. The default model is used "
                "when omitted."
            ),
            examples=["PETR4.SA"],
        ),
    ] = None,
    start_date: Annotated[
        date | None,
        Query(
            description=(
                "First date of the period. Defaults to 365 days before "
                "end_date."
            ),
            examples=["2026-01-01"],
        ),
    ] = None,
    end_date: Annotated[
        date | None,
        Query(
            description="Last date of the period. Defaults to today.",
            examples=["2026-09-30"],
        ),
    ] = None,
) -> PredictionResponse:
    """Download the prices of a period and detect outlier windows.

    Each window gets its date and closing price in the response.
    """
    effective_end_date = end_date or date.today()
    effective_start_date = start_date or (
        effective_end_date - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    )
    if effective_start_date >= effective_end_date:
        raise ValueError("start_date must be earlier than end_date.")

    predictor = service.get_predictor(ticker)
    prediction_table = predict_period(
        predictor,
        effective_start_date.isoformat(),
        effective_end_date.isoformat(),
    )
    window_size = predictor.metadata.window_size
    prediction = AutoencoderPrediction(
        reconstruction_scores=prediction_table[
            "reconstruction_score"
        ].to_numpy(),
        is_outlier=prediction_table["is_outlier"].to_numpy(dtype=bool),
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
        Network interface to listen on.
    port : int
        Port to listen on.
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
