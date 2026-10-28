"""FastAPI application that serves the trained autoencoders."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, timedelta

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
from generative_models.serving.model_service import AutoencoderModelService
from generative_models.serving.predict_autoencoder import predict_period

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
Detects outliers in daily log-returns with dense autoencoders trained
by `generative_models.models.autoencoder.train_autoencoder` and
registered as Weights & Biases model artifacts.

Each window of `window_size` log-returns is normalized with the
training statistics and reconstructed by the autoencoder. A window is
an outlier when its mean squared reconstruction error is strictly
greater than the anomaly threshold saved at training time.

**Model selection:** every prediction accepts an optional `ticker`.
Without it, the default model (the most recently registered artifact,
loaded at startup) is used. Models are downloaded once and kept in
memory; `POST /models/reload` downloads the latest version again.

**Environment variables:**

| Variable | Meaning | Default |
|---|---|---|
| `AUTOENCODER_WANDB_PROJECT` | wandb project | see below |
| `AUTOENCODER_WANDB_ENTITY` | wandb entity | API key default |
| `AUTOENCODER_DEVICE` | torch device | `cuda` if available |
| `AUTOENCODER_PRELOAD_MODEL` | preload default model | `true` |
| `WANDB_API_KEY` | wandb API key (`.env`) | - |

The default project is `autoencoder-simple-outlier-trading-agent`.
"""

NOT_FOUND_RESPONSE = {
    404: {"model": ErrorResponse, "description": "Model not found."}
}
UNAVAILABLE_RESPONSE = {
    503: {"model": ErrorResponse, "description": "wandb is unavailable."}
}
INVALID_INPUT_RESPONSE = {
    422: {"model": ErrorResponse, "description": "Invalid input."}
}


def get_model_service() -> AutoencoderModelService:
    """Return the shared model service.

    Returns
    -------
    AutoencoderModelService
        Process-wide singleton instance.
    """
    return AutoencoderModelService.get_instance()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Preload the default model when the application starts.

    Parameters
    ----------
    application : FastAPI
        Application being started.

    Returns
    -------
    AsyncIterator[None]
        Context that yields while the application runs.
    """
    preload_setting = os.getenv("AUTOENCODER_PRELOAD_MODEL", "true")
    if preload_setting.lower() != "false":
        service_factory = application.dependency_overrides.get(
            get_model_service, get_model_service
        )
        try:
            service_factory().get_predictor()
        # pylint: disable-next=broad-exception-caught
        except Exception as preload_error:
            logger.warning(
                "Could not preload the default model: %s", preload_error
            )
    yield


app = FastAPI(
    title="Autoencoder Outlier Detection API",
    summary="Log-return outlier detection with wandb-registered "
    "autoencoders.",
    description=API_DESCRIPTION,
    version="0.1.0",
    openapi_tags=[
        {"name": "Health", "description": "Service liveness."},
        {"name": "Models", "description": "Loaded and available models."},
        {"name": "Predictions", "description": "Outlier detection."},
    ],
    lifespan=lifespan,
)


@app.exception_handler(LookupError)
async def handle_lookup_error(
    _request: Request,
    error: LookupError,
) -> JSONResponse:
    """Translate a missing model into HTTP 404.

    Parameters
    ----------
    _request : Request
        Request that failed.
    error : LookupError
        Raised error, including `KeyError`.

    Returns
    -------
    JSONResponse
        404 response with the error message.
    """
    message = error.args[0] if error.args else str(error)
    return JSONResponse(status_code=404, content={"detail": str(message)})


@app.exception_handler(ValueError)
async def handle_value_error(
    _request: Request,
    error: ValueError,
) -> JSONResponse:
    """Translate an invalid input into HTTP 422.

    Parameters
    ----------
    _request : Request
        Request that failed.
    error : ValueError
        Raised error.

    Returns
    -------
    JSONResponse
        422 response with the error message.
    """
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.exception_handler(wandb.errors.CommError)
async def handle_wandb_error(
    _request: Request,
    error: wandb.errors.CommError,
) -> JSONResponse:
    """Translate a wandb communication error into HTTP 404 or 503.

    Parameters
    ----------
    _request : Request
        Request that failed.
    error : wandb.errors.CommError
        Raised error.

    Returns
    -------
    JSONResponse
        404 when the artifact was not found, 503 otherwise.
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


def build_model_metadata_response(
    metadata: AutoencoderModelMetadata,
) -> ModelMetadataResponse:
    """Convert predictor metadata into its response schema.

    Parameters
    ----------
    metadata : AutoencoderModelMetadata
        Metadata of a loaded predictor.

    Returns
    -------
    ModelMetadataResponse
        Serializable metadata.
    """
    return ModelMetadataResponse(
        ticker=metadata.ticker,
        artifact_name=metadata.artifact_name,
        window_size=int(metadata.window_size),
        return_train_mean=float(metadata.return_train_mean),
        return_train_std=float(metadata.return_train_std),
        anomaly_threshold=float(metadata.anomaly_threshold),
    )


def build_prediction_response(
    metadata: AutoencoderModelMetadata,
    prediction: AutoencoderPrediction,
    prediction_table: pd.DataFrame | None = None,
) -> PredictionResponse:
    """Convert a prediction into its response schema.

    Parameters
    ----------
    metadata : AutoencoderModelMetadata
        Metadata of the predictor used.
    prediction : AutoencoderPrediction
        Per-window prediction.
    prediction_table : pd.DataFrame | None
        Table from `build_prediction_table`, used to fill the window
        dates and closing prices. Defaults to `None`.

    Returns
    -------
    PredictionResponse
        Serializable prediction with summary counts.
    """
    windows = []
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
        windows.append(
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

    number_of_windows = len(windows)
    number_of_outliers = int(sum(window.is_outlier for window in windows))
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


@app.get(
    "/health",
    tags=["Health"],
    summary="Check that the API is running",
    response_model=HealthResponse,
)
def get_health(
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> HealthResponse:
    """Return the service status and the models in memory."""
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
    summary="List the models kept in memory",
    response_model=list[ModelMetadataResponse],
)
def list_loaded_models(
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> list[ModelMetadataResponse]:
    """Return the metadata of every model in memory."""
    return [
        build_model_metadata_response(metadata)
        for metadata in model_service.list_loaded_models()
    ]


@app.get(
    "/models/available",
    tags=["Models"],
    summary="List the servable models stored in wandb",
    response_model=list[AvailableModelResponse],
    responses=UNAVAILABLE_RESPONSE,
)
def list_available_models(
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> list[AvailableModelResponse]:
    """Return the servable artifacts, newest first, without downloading."""
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
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> ModelMetadataResponse:
    """Return the metadata of the default model, loading it if needed."""
    return build_model_metadata_response(
        model_service.get_predictor().metadata
    )


@app.post(
    "/models/reload",
    tags=["Models"],
    summary="Download a model again from wandb",
    response_model=ModelMetadataResponse,
    responses={**NOT_FOUND_RESPONSE, **UNAVAILABLE_RESPONSE},
)
def reload_model(
    reload_request: ReloadRequest,
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> ModelMetadataResponse:
    """Reload a ticker's model, or the latest model as the default."""
    return build_model_metadata_response(
        model_service.reload(reload_request.ticker).metadata
    )


@app.post(
    "/predictions/log-returns",
    tags=["Predictions"],
    summary="Detect outliers in a log-return series",
    response_model=PredictionResponse,
    responses={
        **NOT_FOUND_RESPONSE,
        **INVALID_INPUT_RESPONSE,
        **UNAVAILABLE_RESPONSE,
    },
)
def predict_log_returns(
    prediction_request: LogReturnPredictionRequest,
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Score every window of the given raw log-returns."""
    predictor = model_service.get_predictor(prediction_request.ticker)
    prediction = predictor.predict(prediction_request.log_returns)
    return build_prediction_response(predictor.metadata, prediction)


@app.post(
    "/predictions/prices",
    tags=["Predictions"],
    summary="Detect outliers in a price series",
    response_model=PredictionResponse,
    responses={
        **NOT_FOUND_RESPONSE,
        **INVALID_INPUT_RESPONSE,
        **UNAVAILABLE_RESPONSE,
    },
)
def predict_prices(
    prediction_request: PricePredictionRequest,
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Score every window of the log-returns of the given prices."""
    predictor = model_service.get_predictor(prediction_request.ticker)
    prediction = predictor.predict_from_prices(prediction_request.prices)
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
    ticker: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    model_service: AutoencoderModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Download a period of prices and score every window.

    `end_date` defaults to today and `start_date` to 365 days before
    `end_date`.
    """
    end_date = end_date or date.today()
    start_date = start_date or end_date - timedelta(days=365)
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
        window_end_indices=pd.RangeIndex(
            window_size - 1, window_size - 1 + len(prediction_table)
        ).to_numpy(),
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
    """Run the API with uvicorn.

    Parameters
    ----------
    host : str
        Interface to bind. Defaults to `"127.0.0.1"`.
    port : int
        Port to listen on. Defaults to 8000.
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
