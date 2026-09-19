"""Unit tests for AutoencoderTickerModelManager."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from generative_models.models.autoencoder.autoencoder_ticker_model_manager import (  # pylint: disable=line-too-long
    AutoencoderTickerModelManager,
    AutoencoderTrainingConfig,
)
from generative_models.models.autoencoder.simple_autoencoder import (
    SimpleAutoencoder,
)
from generative_models.models.ticker_model_manager import TickerModelManager

MODULE_PATH = (
    "generative_models.models.autoencoder.autoencoder_ticker_model_manager"
)


def _build_manager() -> AutoencoderTickerModelManager:
    return AutoencoderTickerModelManager(
        tickers={
            "AAPL": {"start_date": "2020-01-01", "end_date": "2026-01-01"},
            "MSFT": {"start_date": "2021-01-01", "end_date": "2026-01-01"},
        },
        training_config=AutoencoderTrainingConfig(
            window_size=5,
            batch_size=8,
            number_of_epochs=2,
            device="cpu",
        ),
    )


def _build_price_data() -> pd.DataFrame:
    random_generator = np.random.default_rng(0)
    close_prices = 100 * np.exp(
        np.cumsum(random_generator.normal(0, 0.01, size=120))
    )
    return pd.DataFrame(
        {"close_price": close_prices},
        index=pd.bdate_range("2024-01-01", periods=120),
    )


def test_manager_inherits_from_ticker_model_manager() -> None:
    """Check that the class reuses the TickerModelManager interface."""
    assert issubclass(AutoencoderTickerModelManager, TickerModelManager)


def test_list_tickers_keeps_constructor_order() -> None:
    """Check that tickers are listed in the order they were given."""
    assert _build_manager().list_tickers() == ["AAPL", "MSFT"]


def test_get_date_range_returns_configured_dates() -> None:
    """Check that the date range of a ticker is returned as a tuple."""
    manager = _build_manager()

    assert manager.get_date_range("MSFT") == ("2021-01-01", "2026-01-01")


def test_get_date_range_raises_for_unknown_ticker() -> None:
    """Check that an unmanaged ticker raises KeyError."""
    with pytest.raises(KeyError):
        _build_manager().get_date_range("XXXX")


@patch(f"{MODULE_PATH}.fetch_price_series")
def test_fetch_price_data_uses_configured_date_range(
    mock_fetch_price_series,
) -> None:
    """Check that prices are fetched with the ticker's own dates."""
    manager = _build_manager()

    price_data = manager.fetch_price_data("AAPL")

    mock_fetch_price_series.assert_called_once_with(
        "AAPL", "2020-01-01", "2026-01-01"
    )
    assert price_data is mock_fetch_price_series.return_value


def test_get_model_raises_before_training() -> None:
    """Check that requesting an untrained model raises KeyError."""
    with pytest.raises(KeyError, match="AAPL"):
        _build_manager().get_model("AAPL")


@patch(f"{MODULE_PATH}.WandbExperimentTracker")
@patch(f"{MODULE_PATH}.fetch_price_series")
def test_train_model_registers_model_and_metrics(
    mock_fetch_price_series,
    mock_tracker_class,
) -> None:
    """Check that training stores the model, losses and statistics."""
    mock_fetch_price_series.return_value = _build_price_data()
    manager = _build_manager()

    manager.train_model("AAPL")

    assert isinstance(manager.get_model("AAPL"), SimpleAutoencoder)
    training_result = manager.training_results["AAPL"]
    assert len(training_result.training_loss_history) == 2
    assert training_result.validation_loss > 0
    assert len(training_result.normalization_statistics) == 2
    assert "MSFT" not in manager.models
    mock_tracker_class.assert_called_once()


@patch(f"{MODULE_PATH}.WandbExperimentTracker")
@patch(f"{MODULE_PATH}.fetch_price_series")
def test_train_model_tracks_experiment_with_wandb(
    mock_fetch_price_series,
    mock_tracker_class,
) -> None:
    """Check the tracker lifecycle: start, log per epoch, model, finish."""
    mock_fetch_price_series.return_value = _build_price_data()
    tracker = mock_tracker_class.return_value
    manager = _build_manager()

    manager.train_model("AAPL")

    tracker.start_run.assert_called_once()
    logged_metrics = [
        call.args[0] for call in tracker.log_metrics.call_args_list
    ]
    train_logs = [m for m in logged_metrics if "train/loss_mse" in m]
    validation_logs = [m for m in logged_metrics if "validation/loss_mse" in m]
    assert len(train_logs) == 2
    assert len(validation_logs) == 1
    tracker.log_model.assert_called_once()
    assert tracker.log_model.call_args.kwargs["model_name"] == (
        "simple-autoencoder-aapl"
    )
    tracker.finish_run.assert_called_once()


@patch(f"{MODULE_PATH}.WandbExperimentTracker", new=MagicMock())
@patch(f"{MODULE_PATH}.fetch_price_series")
def test_train_model_is_reproducible_with_same_seed(
    mock_fetch_price_series,
) -> None:
    """Check that two trainings with the same seed give equal losses."""
    mock_fetch_price_series.return_value = _build_price_data()
    first_manager = _build_manager()
    second_manager = _build_manager()

    first_manager.train_model("AAPL")
    second_manager.train_model("AAPL")

    assert (
        first_manager.training_results["AAPL"].training_loss_history
        == second_manager.training_results["AAPL"].training_loss_history
    )


@patch(f"{MODULE_PATH}.WandbExperimentTracker", new=MagicMock())
@patch(f"{MODULE_PATH}.fetch_price_series")
def test_train_all_models_registers_a_model_per_ticker(
    mock_fetch_price_series,
) -> None:
    """Check that the inherited method trains every managed ticker."""
    mock_fetch_price_series.return_value = _build_price_data()
    manager = _build_manager()

    manager.train_all_models()

    assert set(manager.models) == {"AAPL", "MSFT"}
    assert set(manager.training_results) == {"AAPL", "MSFT"}
