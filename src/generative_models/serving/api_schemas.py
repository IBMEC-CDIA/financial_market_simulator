"""Request and response schemas of the autoencoder REST API."""

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

TickerField = Annotated[
    str | None,
    Field(
        default=None,
        description=(
            "Ticker of the model to use. When omitted, the default "
            "model is used."
        ),
        examples=["PETR4.SA"],
    ),
]


class HealthResponse(BaseModel):
    """Liveness information of the API."""

    status: str = Field(description="Service status.", examples=["ok"])
    default_ticker: str | None = Field(
        description="Ticker of the default model, if loaded.",
        examples=["PETR4.SA"],
    )
    loaded_tickers: list[str] = Field(
        description="Tickers of the models kept in memory.",
        examples=[["PETR4.SA", "AAPL"]],
    )


class ModelMetadataResponse(BaseModel):
    """Training information of a loaded model."""

    ticker: str = Field(
        description="Ticker the model was trained on.",
        examples=["PETR4.SA"],
    )
    artifact_name: str = Field(
        description="Versioned wandb artifact name.",
        examples=["simple-autoencoder-petr4.sa:v2"],
    )
    window_size: int = Field(
        description="Number of log-returns in each window.",
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
        examples=[1.85],
    )


class AvailableModelResponse(BaseModel):
    """Servable artifact stored in wandb."""

    artifact_name: str = Field(
        description="Versioned wandb artifact name.",
        examples=["simple-autoencoder-aapl:v3"],
    )
    ticker: str = Field(
        description="Ticker of the model.",
        examples=["AAPL"],
    )
    created_at: str = Field(
        description="Creation timestamp of the artifact.",
        examples=["2026-10-20T18:57:02Z"],
    )


class ReloadRequest(BaseModel):
    """Body of a model reload request."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"ticker": "PETR4.SA"}, {}]}
    )

    ticker: TickerField


class LogReturnPredictionRequest(BaseModel):
    """Body of a prediction over raw log-returns."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ticker": "PETR4.SA",
                    "log_returns": [
                        0.012,
                        -0.004,
                        0.003,
                        -0.021,
                        0.008,
                        0.001,
                        -0.006,
                        0.015,
                        -0.002,
                        0.004,
                        -0.085,
                        0.007,
                    ],
                }
            ]
        }
    )

    ticker: TickerField
    log_returns: list[float] = Field(
        min_length=1,
        description="Daily log-returns in chronological order.",
        examples=[[0.012, -0.004, 0.003]],
    )


class PricePredictionRequest(BaseModel):
    """Body of a prediction over a price series."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ticker": "PETR4.SA",
                    "prices": [
                        38.1,
                        38.4,
                        38.2,
                        38.6,
                        37.9,
                        38.3,
                        38.5,
                        38.1,
                        38.7,
                        38.6,
                        38.9,
                        35.2,
                    ],
                }
            ]
        }
    )

    ticker: TickerField
    prices: list[Annotated[float, Field(gt=0)]] = Field(
        min_length=2,
        description="Strictly positive prices in chronological order.",
        examples=[[38.1, 38.4, 38.2]],
    )


class WindowPredictionResponse(BaseModel):
    """Outcome of a single window."""

    window_end_index: int = Field(
        description="Index of the last log-return of the window.",
        examples=[9],
    )
    window_end_date: date | None = Field(
        default=None,
        description="Date where the window ends (market route only).",
        examples=["2026-09-14"],
    )
    close_price: float | None = Field(
        default=None,
        description="Closing price where the window ends (market route "
        "only).",
        examples=[38.6],
    )
    reconstruction_score: float = Field(
        description="Mean squared reconstruction error of the window.",
        examples=[0.42],
    )
    is_outlier: bool = Field(
        description="Whether the score exceeds the anomaly threshold.",
        examples=[False],
    )


class PredictionResponse(BaseModel):
    """Outcome of a prediction request."""

    model: ModelMetadataResponse = Field(
        description="Model used for the prediction.",
    )
    number_of_windows: int = Field(
        description="Number of scored windows.",
        examples=[8],
    )
    number_of_outliers: int = Field(
        description="Number of windows flagged as outliers.",
        examples=[1],
    )
    outlier_rate: float = Field(
        description="Fraction of windows flagged as outliers.",
        examples=[0.125],
    )
    windows: list[WindowPredictionResponse] = Field(
        description="Per-window outcomes in chronological order.",
    )


class ErrorResponse(BaseModel):
    """Error payload of every documented failure."""

    detail: str = Field(
        description="Human-readable error message.",
        examples=["No model loaded for ticker 'AAPL'."],
    )
