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
            "default model is used."
        ),
        examples=["PETR4.SA"],
    ),
]


class HealthResponse(BaseModel):
    """Status of the API and of the models in memory."""

    status: str = Field(
        description="Service status.",
        examples=["ok"],
    )
    default_ticker: str | None = Field(
        description="Ticker of the default model, if already loaded.",
        examples=["PETR4.SA"],
    )
    loaded_tickers: list[str] = Field(
        description="Tickers whose models are in memory.",
        examples=[["PETR4.SA", "AAPL"]],
    )


class ModelMetadataResponse(BaseModel):
    """Training information of a served model."""

    ticker: str = Field(
        description="Ticker symbol the model was trained on.",
        examples=["PETR4.SA"],
    )
    artifact_name: str = Field(
        description="Versioned wandb artifact name of the model.",
        examples=["simple-autoencoder-petr4.sa:v3"],
    )
    window_size: int = Field(
        description="Number of consecutive log-returns in each window.",
        examples=[10],
    )
    return_train_mean: float = Field(
        description="Mean of the training log-returns.",
        examples=[0.0004],
    )
    return_train_std: float = Field(
        description="Standard deviation of the training log-returns.",
        examples=[0.021],
    )
    anomaly_threshold: float = Field(
        description="Reconstruction score above which a window is an "
        "outlier.",
        examples=[1.35],
    )


class AvailableModelResponse(BaseModel):
    """Servable model artifact registered in wandb."""

    artifact_name: str = Field(
        description="Versioned wandb artifact name of the model.",
        examples=["simple-autoencoder-petr4.sa:v3"],
    )
    ticker: str = Field(
        description="Ticker symbol the model was trained on.",
        examples=["PETR4.SA"],
    )
    created_at: str = Field(
        description="Creation timestamp of the artifact.",
        examples=["2026-09-15T12:00:00Z"],
    )


class ReloadRequest(BaseModel):
    """Body of a model reload request."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"ticker": "PETR4.SA"}, {}]}
    )

    ticker: TickerField


class LogReturnPredictionRequest(BaseModel):
    """Body of a prediction request over log-returns."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ticker": "PETR4.SA",
                    "log_returns": [
                        0.012,
                        -0.004,
                        0.007,
                        -0.021,
                        0.003,
                        0.015,
                        -0.009,
                        0.001,
                        0.004,
                        -0.002,
                        0.085,
                        -0.006,
                    ],
                }
            ]
        }
    )

    ticker: TickerField
    log_returns: list[float] = Field(
        min_length=1,
        description=(
            "Consecutive daily log-returns. The model requires at least "
            "window_size values."
        ),
        examples=[[0.012, -0.004, 0.007, -0.021, 0.003]],
    )


class PricePredictionRequest(BaseModel):
    """Body of a prediction request over prices."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ticker": "PETR4.SA",
                    "prices": [
                        36.1,
                        36.5,
                        36.3,
                        36.6,
                        35.8,
                        35.9,
                        36.4,
                        36.1,
                        36.2,
                        36.3,
                        36.2,
                        39.4,
                    ],
                }
            ]
        }
    )

    ticker: TickerField
    prices: list[Annotated[float, Field(gt=0)]] = Field(
        min_length=2,
        description=(
            "Consecutive positive closing prices. The model requires at "
            "least window_size + 1 values."
        ),
        examples=[[36.1, 36.5, 36.3, 36.6, 35.8, 35.9]],
    )


class WindowPredictionResponse(BaseModel):
    """Score and outlier flag of one window."""

    window_end_index: int = Field(
        description="Index of the last log-return of the window.",
        examples=[9],
    )
    window_end_date: date | None = Field(
        default=None,
        description=(
            "Date of the last price of the window. Only filled by the "
            "market data route."
        ),
        examples=["2026-09-14"],
    )
    close_price: float | None = Field(
        default=None,
        description=(
            "Closing price at the end of the window. Only filled by the "
            "market data route."
        ),
        examples=[36.2],
    )
    reconstruction_score: float = Field(
        description="Mean squared reconstruction error of the window.",
        examples=[0.42],
    )
    is_outlier: bool = Field(
        description="Whether the score is above the anomaly threshold.",
        examples=[False],
    )


class PredictionResponse(BaseModel):
    """Result of scoring every window of a series."""

    model: ModelMetadataResponse = Field(
        description="Model used for the prediction.",
        examples=[
            {
                "ticker": "PETR4.SA",
                "artifact_name": "simple-autoencoder-petr4.sa:v3",
                "window_size": 10,
                "return_train_mean": 0.0004,
                "return_train_std": 0.021,
                "anomaly_threshold": 1.35,
            }
        ],
    )
    number_of_windows: int = Field(
        description="Number of scored windows.",
        examples=[240],
    )
    number_of_outliers: int = Field(
        description="Number of windows flagged as outliers.",
        examples=[12],
    )
    outlier_rate: float = Field(
        description="Fraction of windows flagged as outliers.",
        examples=[0.05],
    )
    windows: list[WindowPredictionResponse] = Field(
        description="Prediction of each window, in chronological order.",
        examples=[
            [
                {
                    "window_end_index": 9,
                    "window_end_date": "2026-09-14",
                    "close_price": 36.2,
                    "reconstruction_score": 0.42,
                    "is_outlier": False,
                }
            ]
        ],
    )


class ErrorResponse(BaseModel):
    """Body of every documented error response."""

    detail: str = Field(
        description="Description of the error.",
        examples=["No model loaded for ticker 'AAPL'."],
    )
