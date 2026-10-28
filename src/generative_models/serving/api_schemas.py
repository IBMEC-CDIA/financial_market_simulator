"""Request and response schemas of the autoencoder REST API."""

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

TickerField = Annotated[
    str | None,
    Field(
        default=None,
        description=(
            "Ticker symbol of the model to use. When omitted, the "
            "default model of the service is used."
        ),
        examples=["PETR4.SA"],
    ),
]


class HealthResponse(BaseModel):
    """Service status and models currently held in memory."""

    status: str = Field(
        description="Fixed value reporting that the service is up.",
        examples=["ok"],
    )
    default_ticker: str | None = Field(
        description=(
            "Ticker of the model served when a request omits one, or "
            "null when no default model has been loaded yet."
        ),
        examples=["PETR4.SA"],
    )
    loaded_tickers: list[str] = Field(
        description="Tickers whose models are already in memory.",
        examples=[["PETR4.SA", "AAPL"]],
    )


class ModelMetadataResponse(BaseModel):
    """Description of one trained model available for inference."""

    ticker: str = Field(
        description="Ticker symbol the model was trained on.",
        examples=["PETR4.SA"],
    )
    artifact_name: str = Field(
        description="Versioned name of the wandb artifact.",
        examples=["simple-autoencoder-petr4.sa:v3"],
    )
    window_size: int = Field(
        description=(
            "Number of consecutive log-returns the model reconstructs."
        ),
        examples=[10],
    )
    return_train_mean: float = Field(
        description=(
            "Mean of the training log-returns, used to normalize new "
            "data with the statistics seen during training."
        ),
        examples=[0.0004],
    )
    return_train_std: float = Field(
        description="Standard deviation of the training log-returns.",
        examples=[0.0213],
    )
    anomaly_threshold: float = Field(
        description=(
            "Reconstruction error above which a window is an outlier."
        ),
        examples=[1.8742],
    )


class AvailableModelResponse(BaseModel):
    """Summary of a model artifact that could be served."""

    artifact_name: str = Field(
        description="Versioned name of the wandb artifact.",
        examples=["simple-autoencoder-aapl:v2"],
    )
    ticker: str = Field(
        description="Ticker symbol stored in the artifact metadata.",
        examples=["AAPL"],
    )
    created_at: str = Field(
        description="Creation timestamp of the artifact version.",
        examples=["2026-09-15T18:24:07"],
    )


class ReloadRequest(BaseModel):
    """Body asking the service to download a model again."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"ticker": "PETR4.SA"},
                {"ticker": None},
            ]
        }
    )

    ticker: TickerField


class LogReturnPredictionRequest(BaseModel):
    """Body with a log-return series to be scored."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ticker": "PETR4.SA",
                    "log_returns": [
                        0.0121,
                        -0.0043,
                        0.0067,
                        -0.0192,
                        0.0038,
                        0.0104,
                        -0.0076,
                        0.0015,
                        -0.0231,
                        0.0089,
                    ],
                }
            ]
        }
    )

    ticker: TickerField
    log_returns: list[float] = Field(
        min_length=1,
        description=(
            "Log-return series, in chronological order. It needs at "
            "least as many values as the window size of the model, "
            "which the model itself validates."
        ),
        examples=[[0.0121, -0.0043, 0.0067, -0.0192, 0.0038]],
    )


class PricePredictionRequest(BaseModel):
    """Body with a price series to be scored."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ticker": "PETR4.SA",
                    "prices": [
                        36.12,
                        36.41,
                        36.09,
                        35.78,
                        36.02,
                        36.55,
                        36.21,
                        35.94,
                        35.12,
                        35.47,
                        35.83,
                    ],
                }
            ]
        }
    )

    ticker: TickerField
    prices: list[Annotated[float, Field(gt=0)]] = Field(
        min_length=2,
        description=(
            "Price series, in chronological order, with strictly "
            "positive values. It needs at least one price more than "
            "the window size of the model."
        ),
        examples=[[36.12, 36.41, 36.09, 35.78, 36.02, 36.55]],
    )


class WindowPredictionResponse(BaseModel):
    """Reconstruction result of a single window."""

    window_end_index: int = Field(
        description=(
            "Index, in the log-return series, of the last value of "
            "the window."
        ),
        examples=[9],
    )
    window_end_date: date | None = Field(
        default=None,
        description=(
            "Date that closes the window. Only filled by the market "
            "data route."
        ),
        examples=["2026-09-15"],
    )
    close_price: float | None = Field(
        default=None,
        description=(
            "Closing price of the date that ends the window. Only "
            "filled by the market data route."
        ),
        examples=[35.47],
    )
    reconstruction_score: float = Field(
        description="Mean squared reconstruction error of the window.",
        examples=[0.8431],
    )
    is_outlier: bool = Field(
        description=(
            "Whether the score is strictly above the anomaly "
            "threshold of the model."
        ),
        examples=[False],
    )


class PredictionResponse(BaseModel):
    """Outlier detection result over a whole series."""

    model_config = ConfigDict(protected_namespaces=())

    model: ModelMetadataResponse = Field(
        description="Model that produced the scores.",
    )
    number_of_windows: int = Field(
        description="Number of scored windows.",
        examples=[96],
    )
    number_of_outliers: int = Field(
        description="Number of windows above the anomaly threshold.",
        examples=[5],
    )
    outlier_rate: float = Field(
        description="Outlier windows divided by scored windows.",
        examples=[0.052],
    )
    windows: list[WindowPredictionResponse] = Field(
        description="One entry per window, in chronological order.",
    )


class ErrorResponse(BaseModel):
    """Body returned by every documented error response."""

    detail: str = Field(
        description="Human-readable description of the failure.",
        examples=["No model loaded for ticker 'AAPL'."],
    )
