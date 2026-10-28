"""Unit tests for the autoencoder training pipeline."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder
from generative_models.models.autoencoder.config import (
    AutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
    TrackingConfig,
)
from generative_models.models.autoencoder.trainer import (
    AutoencoderTrainer,
    build_ticker_slug,
    compute_reconstruction_scores,
)

FETCH_PRICE_SERIES_PATH = (
    "generative_models.models.autoencoder.trainer.fetch_price_series"
)


def _make_price_frame(number_of_prices: int) -> pd.DataFrame:
    """Build a synthetic positive close price series."""
    random_generator = np.random.default_rng(0)
    log_return_values = random_generator.normal(0, 0.01, number_of_prices)
    close_prices = 100 * np.exp(np.cumsum(log_return_values))

    return pd.DataFrame(
        {"close_price": close_prices},
        index=pd.date_range("2024-01-01", periods=number_of_prices),
    )


def _make_config(tmp_path: Path) -> AutoencoderTrainingConfig:
    """Build a small CPU training configuration for fast tests."""
    return AutoencoderTrainingConfig(
        market_data=MarketDataConfig(window_size=5, batch_size=8),
        optimization=OptimizationConfig(number_of_epochs=3),
        tracking=TrackingConfig(output_directory=str(tmp_path / "models")),
        device="cpu",
    )


def _make_trainer(
    tmp_path: Path,
    ticker: str = "AAPL",
) -> AutoencoderTrainer:
    """Build a trainer backed by a mocked experiment tracker."""
    return AutoencoderTrainer(
        ticker=ticker,
        config=_make_config(tmp_path),
        experiment_tracker=MagicMock(),
    )


def test_build_ticker_slug_normalizes_ticker() -> None:
    """CA-01: check lowercasing and replacement of invalid characters."""
    assert build_ticker_slug("PETR4.SA") == "petr4.sa"
    assert build_ticker_slug("^BVSP") == "-bvsp"


def test_compute_reconstruction_scores_returns_one_score_per_window() -> None:
    """CA-02: check the shape and sign of the reconstruction scores."""
    model = SimpleAutoencoder(input_dim=5)
    input_tensor = torch.randn(7, 5)

    reconstruction_scores = compute_reconstruction_scores(model, input_tensor)

    assert isinstance(reconstruction_scores, np.ndarray)
    assert reconstruction_scores.shape == (7,)
    assert bool((reconstruction_scores >= 0).all())


def test_model_name_and_file_path_depend_on_ticker(tmp_path) -> None:
    """CA-03: check the artifact name and weights path for a ticker."""
    trainer = _make_trainer(tmp_path, ticker="PETR4.SA")

    assert trainer.model_name == "simple-autoencoder-petr4.sa"
    assert trainer.model_file_path == (
        tmp_path / "models" / "simple-autoencoder-petr4.sa.pt"
    )


@patch(FETCH_PRICE_SERIES_PATH)
def test_prepare_data_builds_expected_number_of_windows(
    mock_fetch_price_series, tmp_path
) -> None:
    """CA-04: check the window counts of the three datasets."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    trainer = _make_trainer(tmp_path)

    prepared_data = trainer.prepare_data()

    assert len(prepared_data.train_dataset) == 76
    assert len(prepared_data.validation_dataset) == 16
    assert len(prepared_data.full_dataset) == 96


@patch(FETCH_PRICE_SERIES_PATH)
def test_prepare_data_normalizes_with_train_statistics(
    mock_fetch_price_series, tmp_path
) -> None:
    """CA-05: check the train series is centered and the std is positive."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    trainer = _make_trainer(tmp_path)

    prepared_data = trainer.prepare_data()

    train_series_mean = prepared_data.train_dataset.series_values.mean()
    assert abs(train_series_mean.item()) < 1e-5
    assert prepared_data.return_train_std > 0


@patch(FETCH_PRICE_SERIES_PATH)
def test_prepare_data_raises_when_not_enough_data(
    mock_fetch_price_series, tmp_path
) -> None:
    """CA-06: check that a too short series raises ValueError."""
    mock_fetch_price_series.return_value = _make_price_frame(12)
    trainer = _make_trainer(tmp_path)

    with pytest.raises(ValueError, match="Not enough data"):
        trainer.prepare_data()


@patch(FETCH_PRICE_SERIES_PATH)
def test_train_calls_tracker_lifecycle_once(
    mock_fetch_price_series, tmp_path
) -> None:
    """CA-07: check start_run, log_model and finish_run are called once."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    trainer = _make_trainer(tmp_path)

    trainer.train()

    trainer.experiment_tracker.start_run.assert_called_once()
    trainer.experiment_tracker.log_model.assert_called_once()
    trainer.experiment_tracker.finish_run.assert_called_once()


@patch(FETCH_PRICE_SERIES_PATH)
def test_train_logs_epoch_and_validation_metrics(
    mock_fetch_price_series, tmp_path
) -> None:
    """CA-08: check per-epoch and validation metric logging."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    trainer = _make_trainer(tmp_path)

    trainer.train()

    logged_metrics = [
        call.args[0]
        for call in trainer.experiment_tracker.log_metrics.call_args_list
    ]
    epoch_metrics = [
        metrics for metrics in logged_metrics if "train/loss_mse" in metrics
    ]
    validation_metrics = [
        metrics
        for metrics in logged_metrics
        if "validation/loss_mse" in metrics
    ]
    assert [metrics["epoch"] for metrics in epoch_metrics] == [1, 2, 3]
    assert len(validation_metrics) == 1


@patch(FETCH_PRICE_SERIES_PATH)
def test_train_returns_complete_result(
    mock_fetch_price_series, tmp_path
) -> None:
    """CA-09: check the loss history, validation loss and threshold."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    trainer = _make_trainer(tmp_path)

    training_result = trainer.train()

    assert len(training_result.training_loss_history) == 3
    assert math.isfinite(training_result.validation_loss)
    assert isinstance(training_result.anomaly_threshold, float)
    assert training_result.anomaly_threshold > 0
    assert training_result.model_file_path == trainer.model_file_path


@patch(FETCH_PRICE_SERIES_PATH)
def test_train_logs_model_with_inference_metadata(
    mock_fetch_price_series, tmp_path
) -> None:
    """CA-10: check the metadata keys passed to log_model."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    trainer = _make_trainer(tmp_path)

    trainer.train()

    log_model_call = trainer.experiment_tracker.log_model.call_args
    assert log_model_call.kwargs["model_name"] == trainer.model_name
    assert log_model_call.kwargs["model_file_path"] == str(
        trainer.model_file_path
    )
    assert set(log_model_call.kwargs["metadata"]) == {
        "final_train_loss",
        "validation_loss",
        "anomaly_threshold",
        "return_train_mean",
        "return_train_std",
    }


@patch(FETCH_PRICE_SERIES_PATH)
def test_train_finishes_run_when_log_model_fails(
    mock_fetch_price_series, tmp_path
) -> None:
    """CA-11: check finish_run is called even if log_model raises."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    trainer = _make_trainer(tmp_path)
    trainer.experiment_tracker.log_model.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        trainer.train()

    trainer.experiment_tracker.finish_run.assert_called_once()


@patch("generative_models.models.autoencoder.trainer.WandbExperimentTracker")
def test_default_tracker_uses_tracking_config(
    mock_tracker_class, tmp_path
) -> None:
    """CA-12: check the default tracker project name and run config."""
    training_config = _make_config(tmp_path)

    trainer = AutoencoderTrainer(ticker="AAPL", config=training_config)

    tracker_kwargs = mock_tracker_class.call_args.kwargs
    assert trainer.experiment_tracker is mock_tracker_class.return_value
    assert tracker_kwargs["project_name"] == (
        training_config.tracking.project_name
    )
    assert tracker_kwargs["env_file_path"] == (
        training_config.tracking.env_file_path
    )
    assert tracker_kwargs["config"]["ticker"] == "AAPL"
    assert set(training_config.to_flat_dict()) <= set(
        tracker_kwargs["config"]
    )
    assert tracker_kwargs["config"]["optimizer"] == "Adam"
    assert tracker_kwargs["config"]["loss_function"] == "MSE"
    assert tracker_kwargs["config"]["model"] == "SimpleAutoencoder"


@patch(FETCH_PRICE_SERIES_PATH)
def test_train_is_reproducible(mock_fetch_price_series, tmp_path) -> None:
    """CA-13: check that two runs produce the same loss history."""
    mock_fetch_price_series.return_value = _make_price_frame(101)

    first_result = _make_trainer(tmp_path).train()
    second_result = _make_trainer(tmp_path).train()

    assert (
        first_result.training_loss_history
        == second_result.training_loss_history
    )
