"""Request and response schemas of the autoencoder prediction API."""

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SAMPLE_LOG_RETURNS = [
    0.01,
    -0.02,
    0.005,
    0.0,
    0.013,
    -0.04,
    0.002,
    0.01,
    -0.007,
    0.003,
    0.02,
    -0.01,
]

TickerField = Annotated[
    str | None,
    Field(
        default=None,
        description=(
            "Ticker of the model to use. When omitted, the default model "
            "is used."
        ),
        examples=["PETR4.SA"],
    ),
]


class HealthResponse(BaseModel):
    """Status of the service and of the models in memory.

    Parameters
    ----------
    status : str
        Always `"ok"` while the service is up.
    default_ticker : str | None
        Ticker of the default model, or `None` if none is loaded yet.
    loaded_tickers : list[str]
        Tickers of the models currently in memory.
    """

    status: str = Field(
        description="Service status. It is `ok` while the API is up.",
        examples=["ok"],
    )
    default_ticker: str | None = Field(
        description="Ticker of the default model, or null if none is loaded.",
        examples=["PETR4.SA"],
    )
    loaded_tickers: list[str] = Field(
        description="Tickers of the models currently in memory.",
        examples=[["PETR4.SA", "AAPL"]],
    )


class ModelMetadataResponse(BaseModel):
    """Metadata of a trained autoencoder used for inference.

    Parameters
    ----------
    ticker : str
        Ticker symbol the model was trained on.
    artifact_name : str
        Versioned name of the wandb artifact.
    window_size : int
        Number of log-returns in each window.
    return_train_mean : float
        Mean of the training log-returns.
    return_train_std : float
        Standard deviation of the training log-returns.
    anomaly_threshold : float
        Reconstruction score above which a window is an outlier.
    """

    ticker: str = Field(
        description="Ticker symbol the model was trained on.",
        examples=["PETR4.SA"],
    )
    artifact_name: str = Field(
        description="Versioned name of the wandb artifact.",
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
        examples=[0.0213],
    )
    anomaly_threshold: float = Field(
        description=(
            "Reconstruction score above which a window is an outlier."
        ),
        examples=[0.0051],
    )


class AvailableModelResponse(BaseModel):
    """A model artifact that can be loaded from wandb.

    Parameters
    ----------
    artifact_name : str
        Versioned name of the artifact.
    ticker : str
        Ticker symbol the model was trained on.
    created_at : str
        Creation time of the artifact, as reported by wandb.
    """

    artifact_name: str = Field(
        description="Versioned name of the wandb artifact.",
        examples=["simple-autoencoder-petr4.sa:v2"],
    )
    ticker: str = Field(
        description="Ticker symbol the model was trained on.",
        examples=["PETR4.SA"],
    )
    created_at: str = Field(
        description="Creation time of the artifact, as reported by wandb.",
        examples=["2026-09-15T18:30:00"],
    )


class ReloadRequest(BaseModel):
    """Body of a request to download a model again.

    Parameters
    ----------
    ticker : str | None
        Ticker to reload. When omitted, the latest model of any ticker is
        loaded and becomes the default model.
    """

    ticker: TickerField

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"ticker": "PETR4.SA"}, {}],
        },
    )


class LogReturnPredictionRequest(BaseModel):
    """Body of a request to score a series of log-returns.

    Parameters
    ----------
    ticker : str | None
        Ticker of the model to use. The default model is used when
        omitted.
    log_returns : list[float]
        Consecutive log-returns, oldest first. It must contain at least
        one value, and at least `window_size` values in practice.
    """

    ticker: TickerField
    log_returns: list[float] = Field(
        min_length=1,
        description=(
            "Consecutive log-returns, oldest first. The model's window "
            "size sets the minimum number of values."
        ),
        examples=[SAMPLE_LOG_RETURNS],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"ticker": "PETR4.SA", "log_returns": SAMPLE_LOG_RETURNS},
            ],
        },
    )


class PricePredictionRequest(BaseModel):
    """Body of a request to score a series of closing prices.

    Parameters
    ----------
    ticker : str | None
        Ticker of the model to use. The default model is used when
        omitted.
    prices : list[float]
        Positive closing prices, oldest first. It must contain at least
        two values.
    """

    ticker: TickerField
    prices: list[Annotated[float, Field(gt=0)]] = Field(
        min_length=2,
        description=(
            "Positive closing prices, oldest first. At least two values "
            "are required."
        ),
        examples=[[10.0, 10.2, 10.1, 10.4, 10.3, 10.5]],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ticker": "PETR4.SA",
                    "prices": [10.0, 10.2, 10.1, 10.4, 10.3, 10.5],
                },
            ],
        },
    )


class WindowPredictionResponse(BaseModel):
    """Score of one window of the series.

    Parameters
    ----------
    window_end_index : int
        Index of the last log-return of the window.
    window_end_date : date | None
        Date of the window's price. Only set by the market data route.
    close_price : float | None
        Closing price of the window's date. Only set by the market data
        route.
    reconstruction_score : float
        Mean squared reconstruction error of the window.
    is_outlier : bool
        `True` when the score is above the anomaly threshold.
    """

    window_end_index: int = Field(
        description="Index of the last log-return of the window.",
        examples=[4],
    )
    window_end_date: date | None = Field(
        default=None,
        description=(
            "Date of the window's price. Only filled by the market data "
            "route; null otherwise."
        ),
        examples=["2026-09-14"],
    )
    close_price: float | None = Field(
        default=None,
        description=(
            "Closing price of the window's date. Only filled by the market "
            "data route; null otherwise."
        ),
        examples=[41.25],
    )
    reconstruction_score: float = Field(
        description="Mean squared reconstruction error of the window.",
        examples=[0.0031],
    )
    is_outlier: bool = Field(
        description="True when the score is above the anomaly threshold.",
        examples=[False],
    )


class PredictionResponse(BaseModel):
    """Outlier detection result for a whole series.

    Parameters
    ----------
    model : ModelMetadataResponse
        Metadata of the model that made the prediction.
    number_of_windows : int
        Total number of windows scored.
    number_of_outliers : int
        Number of windows flagged as outliers.
    outlier_rate : float
        Fraction of windows flagged as outliers.
    windows : list[WindowPredictionResponse]
        Score of each window, in chronological order.
    """

    model: ModelMetadataResponse = Field(
        description="Metadata of the model that made the prediction.",
    )
    number_of_windows: int = Field(
        description="Total number of windows scored.",
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
        description="Score of each window, in chronological order.",
    )


class ErrorResponse(BaseModel):
    """Body of every documented error response.

    Parameters
    ----------
    detail : str
        Human-readable description of the error.
    """

    detail: str = Field(
        description="Human-readable description of the error.",
        examples=["No model loaded for ticker 'AAPL'. Call load_model first."],
    )
