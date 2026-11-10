"""One-day-ahead close forecasts from the regression model of a ticker."""

import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from generative_models.models.regression.feed_forward_regression_model import (
    FeedForwardRegressionModel,
)

REGRESSION_DIRECTORY_NAME = "regression"
REQUIRED_ARTIFACT_KEYS = (
    "state_dict",
    "input_window_size",
    "hidden_layer_size",
    "return_train_mean",
    "return_train_std",
    "artifact_name",
)
LOOKBACK_CALENDAR_DAYS = 120
WEEKEND_FIRST_WEEKDAY = 5
FORECAST_DECIMALS = 4


@dataclass(frozen=True)
class RegressionForecaster:
    """Regression model of one ticker, ready for inference.

    Parameters
    ----------
    ticker : str
        Ticker the model was trained on, in upper case.
    model : nn.Module
        Feed-forward regression model in evaluation mode.
    input_window_size : int
        Number of past log-returns given as input.
    return_train_mean : float
        Mean of the log-returns used in training.
    return_train_std : float
        Standard deviation of the log-returns used in training.
    artifact_name : str
        Versioned name of the trained artifact.
    """

    ticker: str
    model: nn.Module
    input_window_size: int
    return_train_mean: float
    return_train_std: float
    artifact_name: str


def regression_artifact_path(
    ticker: str,
    model_directory: Path,
) -> Path:
    """Path of the regression artifact of a ticker.

    Parameters
    ----------
    ticker : str
        Ticker of the model, in any letter case.
    model_directory : Path
        Directory that holds the models of the service.

    Returns
    -------
    Path
        `<model_directory>/regression/<TICKER>.pt`.

    Example
    -------
    >>> regression_artifact_path("aapl", Path("models")).name
    'AAPL.pt'
    """
    directory = model_directory / REGRESSION_DIRECTORY_NAME
    return directory / f"{ticker.upper()}.pt"


def load_regression_forecaster(
    ticker: str,
    model_directory: Path,
) -> RegressionForecaster | None:
    """Load the regression model of a ticker from its artifact.

    Parameters
    ----------
    ticker : str
        Ticker whose model is loaded.
    model_directory : Path
        Directory that holds the models of the service.

    Returns
    -------
    RegressionForecaster | None
        The forecaster, or None when the artifact is missing or does not
        have every key of the contract.

    Example
    -------
    >>> load_regression_forecaster("AAPL", Path("models")) is None
    True
    """
    path = regression_artifact_path(ticker, model_directory)
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or any(
        key not in payload for key in REQUIRED_ARTIFACT_KEYS
    ):
        return None
    model = FeedForwardRegressionModel(
        input_window_size=int(payload["input_window_size"]),
        hidden_layer_size=int(payload["hidden_layer_size"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return RegressionForecaster(
        ticker=ticker.upper(),
        model=model,
        input_window_size=int(payload["input_window_size"]),
        return_train_mean=float(payload["return_train_mean"]),
        return_train_std=float(payload["return_train_std"]),
        artifact_name=str(payload["artifact_name"]),
    )


def next_business_day(
    day: date,
) -> date:
    """First weekday after a day. Exchange holidays are not considered.

    Parameters
    ----------
    day : date
        Reference day.

    Returns
    -------
    date
        The next day from Monday to Friday.

    Example
    -------
    >>> next_business_day(date(2026, 5, 15))
    datetime.date(2026, 5, 18)
    """
    candidate = day + timedelta(days=1)
    while candidate.weekday() >= WEEKEND_FIRST_WEEKDAY:
        candidate += timedelta(days=1)
    return candidate


def log_returns_of(
    closes: np.ndarray,
) -> np.ndarray:
    """Daily log-returns of a series of closing prices.

    Parameters
    ----------
    closes : np.ndarray
        Positive closing prices in chronological order.

    Returns
    -------
    np.ndarray
        `ln(close_t / close_{t-1})`, one value less than `closes`.

    Example
    -------
    >>> log_returns_of(np.array([1.0, np.e]))
    array([1.])
    """
    return np.diff(np.log(closes))


def predict_next_return(
    forecaster: RegressionForecaster,
    log_returns: np.ndarray,
) -> float:
    """Predict the next log-return from the most recent window.

    Parameters
    ----------
    forecaster : RegressionForecaster
        Model of the ticker, with its training normalization.
    log_returns : np.ndarray
        Log-returns in chronological order; the last values are the most
        recent ones.

    Returns
    -------
    float
        Predicted log-return, already de-normalized.

    Raises
    ------
    ValueError
        When there are fewer log-returns than the input window size.

    Example
    -------
    >>> predict_next_return(forecaster, returns)  # doctest: +SKIP
    0.001
    """
    window = forecaster.input_window_size
    if len(log_returns) < window:
        raise ValueError(
            f"At least {window} log-returns are needed to forecast."
        )
    mean = forecaster.return_train_mean
    std = forecaster.return_train_std
    recent = (log_returns[-window:] - mean) / std
    tensor = torch.tensor(recent, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        normalized = float(forecaster.model(tensor).item())
    return normalized * std + mean


def forecast_next_close(
    forecaster: RegressionForecaster,
    closes: pd.Series,
    as_of: date,
) -> dict[str, Any]:
    """One-day-ahead forecast of the close, using only the data up to a day.

    Parameters
    ----------
    forecaster : RegressionForecaster
        Model of the ticker.
    closes : pd.Series
        Closing prices indexed by date. Days after `as_of` are ignored.
    as_of : date
        Last day whose data may be used.

    Returns
    -------
    dict[str, Any]
        Forecast with `ticker`, `as_of`, `target_date`, `predicted_close`
        and `window_days`.

    Raises
    ------
    ValueError
        When there are not enough closes up to `as_of`, or the predicted
        close is not a positive finite number.

    Example
    -------
    >>> forecast_next_close(forecaster, closes, as_of)  # doctest: +SKIP
    {'ticker': 'AAPL', 'as_of': '2026-05-12', ...}
    """
    index = pd.DatetimeIndex(pd.to_datetime(closes.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    cutoff = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    values = closes.to_numpy(dtype=float)
    visible = values[index < cutoff]
    visible = visible[np.isfinite(visible)]
    required = forecaster.input_window_size + 1
    if len(visible) < required:
        raise ValueError(
            f"At least {required} closing prices up to {as_of.isoformat()} "
            "are needed to forecast."
        )
    predicted_return = predict_next_return(
        forecaster,
        log_returns_of(visible),
    )
    predicted_close = float(visible[-1]) * math.exp(predicted_return)
    if not math.isfinite(predicted_close) or predicted_close <= 0:
        raise ValueError("The forecast of the model is not a valid price.")
    return {
        "ticker": forecaster.ticker,
        "as_of": as_of.isoformat(),
        "target_date": next_business_day(as_of).isoformat(),
        "predicted_close": round(predicted_close, FORECAST_DECIMALS),
        "window_days": forecaster.input_window_size,
    }
