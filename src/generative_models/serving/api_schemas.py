"""Request and response schemas of the autoencoder serving API."""

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

TickerField = Annotated[
    str | None,
    Field(
        default=None,
        description=(
            "Ticker whose model should be used (case-insensitive). "
            "When omitted, the default model is used: the most "
            "recently trained autoencoder of any ticker."
        ),
        examples=["PETR4.SA"],
    ),
]


class HealthResponse(BaseModel):
    """Liveness information of the service."""

    status: str = Field(
        description="Always `ok` while the service is running.",
        examples=["ok"],
    )
    default_ticker: str | None = Field(
        description=(
            "Ticker of the default model, or `null` if it has not "
            "been loaded yet."
        ),
        examples=["PETR4.SA"],
    )
    loaded_tickers: list[str] = Field(
        description="Tickers whose models are held in memory.",
        examples=[["PETR4.SA"]],
    )


class ModelMetadataResponse(BaseModel):
    """Metadata of a model loaded in memory."""

    ticker: str = Field(
        description="Ticker symbol the model was trained on.",
        examples=["PETR4.SA"],
    )
    artifact_name: str = Field(
        description="wandb artifact name and version.",
        examples=["simple-autoencoder-petr4.sa:v0"],
    )
    window_size: int = Field(
        description=(
            "Number of consecutive log-returns in each input window. "
            "At least this many log-returns (or this many plus one "
            "prices) are required to predict."
        ),
        examples=[10],
    )
    return_train_mean: float = Field(
        description="Mean of the training log-returns (normalization).",
        examples=[0.00041],
    )
    return_train_std: float = Field(
        description=(
            "Standard deviation of the training log-returns "
            "(normalization)."
        ),
        examples=[0.0285],
    )
    anomaly_threshold: float = Field(
        description=(
            "Reconstruction error above which a window is classified "
            "as an outlier."
        ),
        examples=[1.045328],
    )


class AvailableModelResponse(BaseModel):
    """Servable model artifact stored in wandb."""

    artifact_name: str = Field(
        description="wandb artifact name and version.",
        examples=["simple-autoencoder-petr4.sa:v0"],
    )
    ticker: str = Field(
        description="Ticker symbol the model was trained on.",
        examples=["PETR4.SA"],
    )
    created_at: str = Field(
        description="Artifact creation timestamp (ISO 8601, UTC).",
        examples=["2026-10-06T02:49:45Z"],
    )


class ReloadRequest(BaseModel):
    """Body of the model reload endpoint."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"ticker": None}, {"ticker": "AAPL"}]}
    )

    ticker: TickerField


class LogReturnPredictionRequest(BaseModel):
    """Body of the log-return prediction endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ticker": "PETR4.SA",
                    "log_returns": [
                        0.0042,
                        -0.0113,
                        0.0021,
                        0.0087,
                        -0.0035,
                        0.0004,
                        -0.0721,
                        0.0156,
                        0.0019,
                        -0.0048,
                        0.0062,
                    ],
                }
            ]
        }
    )

    ticker: TickerField
    log_returns: list[float] = Field(
        min_length=1,
        description=(
            "Raw (not normalized) daily log-returns, in chronological "
            "order. Must contain at least `window_size` values; the "
            "API normalizes them with the training statistics."
        ),
    )


class PricePredictionRequest(BaseModel):
    """Body of the price prediction endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ticker": "PETR4.SA",
                    "prices": [
                        38.10,
                        38.42,
                        37.95,
                        38.20,
                        38.61,
                        38.47,
                        38.49,
                        35.81,
                        36.37,
                        36.44,
                        36.27,
                        36.50,
                    ],
                }
            ]
        }
    )

    ticker: TickerField
    prices: list[Annotated[float, Field(gt=0)]] = Field(
        min_length=2,
        description=(
            "Positive daily closing prices, in chronological order. "
            "Must contain at least `window_size + 1` values."
        ),
    )


class WindowPredictionResponse(BaseModel):
    """Prediction for a single sliding window."""

    window_end_index: int = Field(
        description=(
            "Index, in the log-return series, of the last return of "
            "the window. For price input, it refers to the return "
            "between prices `i` and `i + 1`."
        ),
        examples=[9],
    )
    window_end_date: date | None = Field(
        default=None,
        description=(
            "Date of the last return of the window. Only filled in "
            "by the market data endpoint."
        ),
        examples=["2026-10-05"],
    )
    close_price: float | None = Field(
        default=None,
        description=(
            "Closing price at `window_end_date`. Only filled in by the "
            "market "
            "data endpoint."
        ),
        examples=[55.36],
    )
    reconstruction_score: float = Field(
        description="Mean squared reconstruction error of the window.",
        examples=[0.867375],
    )
    is_outlier: bool = Field(
        description=(
            "`true` when `reconstruction_score` is above the model "
            "anomaly threshold."
        ),
        examples=[False],
    )


class PredictionResponse(BaseModel):
    """Outlier detection result for a series."""

    model: ModelMetadataResponse = Field(
        description="Model used to compute the prediction.",
    )
    number_of_windows: int = Field(
        description="Number of windows scored.",
        examples=[239],
    )
    number_of_outliers: int = Field(
        description="Number of windows classified as outliers.",
        examples=[3],
    )
    outlier_rate: float = Field(
        description="Fraction of windows classified as outliers.",
        examples=[0.0126],
    )
    windows: list[WindowPredictionResponse] = Field(
        description="Per-window predictions, in chronological order.",
    )


class ErrorResponse(BaseModel):
    """Error payload returned by the API."""

    detail: str = Field(
        description="Human-readable description of the error.",
        examples=["No model found for ticker 'XYZ'."],
    )
