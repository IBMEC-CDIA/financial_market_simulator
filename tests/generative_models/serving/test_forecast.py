"""Tests of the one-day-ahead forecast of the regression model."""

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from generative_models.models.regression.feed_forward_regression_model import (
    FeedForwardRegressionModel,
)
from generative_models.serving.forecast import (
    RegressionForecaster,
    forecast_next_close,
    load_regression_forecaster,
    log_returns_of,
    next_business_day,
    predict_next_return,
    regression_artifact_path,
)

WINDOW = 4
HIDDEN = 3
MEAN = 0.001
STD = 0.02


def constant_model(output: float) -> FeedForwardRegressionModel:
    """Forecast model whose output is the given constant."""
    model = FeedForwardRegressionModel(
        input_window_size=WINDOW,
        hidden_layer_size=HIDDEN,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.fill_(output)
    model.eval()
    return model


def forecaster(output: float) -> RegressionForecaster:
    """Forecaster built on the constant model with fixed statistics."""
    return RegressionForecaster(
        ticker="AAPL",
        model=constant_model(output),
        input_window_size=WINDOW,
        return_train_mean=MEAN,
        return_train_std=STD,
        artifact_name="regression-AAPL",
    )


def closes_series(values: list[float], first_day: date) -> pd.Series:
    """Series of closes on consecutive business days."""
    index = pd.bdate_range(first_day, periods=len(values))
    return pd.Series(values, index=index)


def test_next_business_day_skips_the_weekend():
    """Check that the next business day skips the weekend."""
    assert next_business_day(date(2026, 5, 15)) == date(2026, 5, 18)
    assert next_business_day(date(2026, 5, 16)) == date(2026, 5, 18)
    assert next_business_day(date(2026, 5, 14)) == date(2026, 5, 15)


def test_log_returns_are_daily_log_changes():
    """Check that log-returns are daily log changes."""
    returns = log_returns_of(np.array([1.0, math.e, math.e**2]))
    assert returns == pytest.approx([1.0, 1.0])


def test_the_predicted_return_is_denormalized_from_the_model_output():
    """Check that the predicted return is denormalized."""
    value = predict_next_return(forecaster(0.5), np.full(10, 0.01))
    assert value == pytest.approx(0.5 * STD + MEAN)


def test_a_window_shorter_than_the_input_is_refused():
    """Check that a window shorter than the input is refused."""
    with pytest.raises(ValueError, match="log-returns"):
        predict_next_return(forecaster(0.0), np.full(WINDOW - 1, 0.01))


def test_the_forecast_applies_the_predicted_return_to_the_last_close():
    """Check that the forecast applies the return to the last close."""
    closes = closes_series([100.0 + i for i in range(10)], date(2026, 4, 1))
    as_of = closes.index[-1].date()
    result = forecast_next_close(forecaster(0.5), closes, as_of)
    expected = round(float(closes.iloc[-1]) * math.exp(0.5 * STD + MEAN), 4)
    assert result["predicted_close"] == expected
    assert result["ticker"] == "AAPL"
    assert result["as_of"] == as_of.isoformat()
    assert result["target_date"] == next_business_day(as_of).isoformat()
    assert result["window_days"] == WINDOW


def test_prices_after_as_of_never_change_the_forecast():
    """Check that prices after as of never change the forecast."""
    closes = closes_series([100.0 + i for i in range(10)], date(2026, 4, 1))
    as_of = closes.index[5].date()
    future = closes.copy()
    future.iloc[6:] = 1_000_000.0
    clean = forecast_next_close(forecaster(0.2), closes, as_of)
    changed = forecast_next_close(forecaster(0.2), future, as_of)
    assert clean == changed


def test_too_few_closes_up_to_as_of_are_refused():
    """Check that too few closes up to as of are refused."""
    closes = closes_series([100.0, 101.0, 102.0], date(2026, 4, 1))
    with pytest.raises(ValueError, match="closing prices"):
        forecast_next_close(forecaster(0.0), closes, closes.index[-1].date())


def test_a_missing_artifact_gives_no_forecaster(tmp_path: Path):
    """Check that a missing artifact gives no forecaster."""
    assert load_regression_forecaster("AAPL", tmp_path) is None


def test_an_artifact_without_the_contract_keys_is_ignored(tmp_path: Path):
    """Check that an artifact without the contract keys is ignored."""
    path = regression_artifact_path("AAPL", tmp_path)
    path.parent.mkdir(parents=True)
    torch.save({"state_dict": {}}, path)
    assert load_regression_forecaster("AAPL", tmp_path) is None


def test_an_artifact_round_trips_through_the_loader(tmp_path: Path):
    """Check that an artifact round trips through the loader."""
    model = constant_model(0.3)
    path = regression_artifact_path("aapl", tmp_path)
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_window_size": WINDOW,
            "hidden_layer_size": HIDDEN,
            "return_train_mean": MEAN,
            "return_train_std": STD,
            "artifact_name": "regression-AAPL",
        },
        path,
    )
    loaded = load_regression_forecaster("AAPL", tmp_path)
    assert loaded is not None
    assert loaded.ticker == "AAPL"
    assert loaded.input_window_size == WINDOW
    assert predict_next_return(loaded, np.full(WINDOW, 0.0)) == pytest.approx(
        0.3 * STD + MEAN
    )
