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

FETCH_PATH = "generative_models.models.autoencoder.trainer.fetch_price_series"
TRACKER_PATH = (
    "generative_models.models.autoencoder.trainer.WandbExperimentTracker"
)


def _make_price_data(number_of_prices: int) -> pd.DataFrame:
    """Build a synthetic positive price series."""
    random_generator = np.random.default_rng(0)
    price_values = 100 * np.exp(
        np.cumsum(random_generator.normal(0, 0.01, number_of_prices))
    )
    return pd.DataFrame({"close_price": price_values})


def _make_config(tmp_path: Path) -> AutoencoderTrainingConfig:
    """Build a small configuration for fast CPU tests."""
    return AutoencoderTrainingConfig(
        market_data=MarketDataConfig(window_size=5, batch_size=8),
        optimization=OptimizationConfig(number_of_epochs=3),
        tracking=TrackingConfig(output_directory=str(tmp_path / "models")),
        device="cpu",
    )


def _make_trainer(
    tmp_path: Path,
    ticker: str = "PETR4.SA",
) -> AutoencoderTrainer:
    """Build a trainer with a mocked experiment tracker."""
    return AutoencoderTrainer(
        ticker=ticker,
        config=_make_config(tmp_path),
        experiment_tracker=MagicMock(),
    )


def test_build_ticker_slug() -> None:
    """CA-01: tickers are lowercased and unsafe characters replaced."""
    assert build_ticker_slug("PETR4.SA") == "petr4.sa"
    assert build_ticker_slug("^BVSP") == "-bvsp"


def test_compute_reconstruction_scores_shape_and_sign() -> None:
    """CA-02: one non-negative score per window."""
    model = SimpleAutoencoder(input_dim=5)

    scores = compute_reconstruction_scores(model, torch.randn(7, 5))

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (7,)
    assert np.all(scores >= 0)


def test_model_name_and_file_path(tmp_path: Path) -> None:
    """CA-03: model name and file path depend on the ticker slug."""
    trainer = _make_trainer(tmp_path)

    assert trainer.model_name == "simple-autoencoder-petr4.sa"
    assert trainer.model_file_path == (
        tmp_path / "models" / "simple-autoencoder-petr4.sa.pt"
    )


@patch(FETCH_PATH, return_value=_make_price_data(101))
def test_prepare_data_dataset_sizes(_mock_fetch, tmp_path: Path) -> None:
    """CA-04: 100 log-returns yield 76, 16 and 96 windows."""
    prepared_data = _make_trainer(tmp_path).prepare_data()

    assert len(prepared_data.train_dataset) == 76
    assert len(prepared_data.validation_dataset) == 16
    assert len(prepared_data.full_dataset) == 96


@patch(FETCH_PATH, return_value=_make_price_data(101))
def test_prepare_data_normalizes_train_split(
    _mock_fetch,
    tmp_path: Path,
) -> None:
    """CA-05: the training series is centered and the std is positive."""
    prepared_data = _make_trainer(tmp_path).prepare_data()

    train_series = prepared_data.train_dataset.series_values
    assert abs(train_series.mean().item()) < 1e-5
    assert prepared_data.return_train_std > 0


@patch(FETCH_PATH, return_value=_make_price_data(12))
def test_prepare_data_raises_with_short_series(
    _mock_fetch,
    tmp_path: Path,
) -> None:
    """CA-06: too few log-returns raise ValueError."""
    with pytest.raises(ValueError, match="Not enough data"):
        _make_trainer(tmp_path).prepare_data()


@patch(FETCH_PATH, return_value=_make_price_data(101))
def test_train_calls_tracker_lifecycle_once(
    _mock_fetch,
    tmp_path: Path,
) -> None:
    """CA-07: start_run, log_model and finish_run are called once."""
    trainer = _make_trainer(tmp_path)

    trainer.train()

    trainer.experiment_tracker.start_run.assert_called_once()
    trainer.experiment_tracker.log_model.assert_called_once()
    trainer.experiment_tracker.finish_run.assert_called_once()


@patch(FETCH_PATH, return_value=_make_price_data(101))
def test_train_logs_epoch_and_validation_metrics(
    _mock_fetch,
    tmp_path: Path,
) -> None:
    """CA-08: one train metric per epoch and one validation metric."""
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


@patch(FETCH_PATH, return_value=_make_price_data(101))
def test_train_returns_result_summary(_mock_fetch, tmp_path: Path) -> None:
    """CA-09: history length, finite validation loss, positive threshold."""
    result = _make_trainer(tmp_path).train()

    assert len(result.training_loss_history) == 3
    assert math.isfinite(result.validation_loss)
    assert isinstance(result.anomaly_threshold, float)
    assert result.anomaly_threshold > 0


@patch(FETCH_PATH, return_value=_make_price_data(101))
def test_train_logs_model_metadata(_mock_fetch, tmp_path: Path) -> None:
    """CA-10: log_model metadata has the five serving keys."""
    trainer = _make_trainer(tmp_path)

    trainer.train()

    metadata = trainer.experiment_tracker.log_model.call_args.kwargs[
        "metadata"
    ]
    assert {
        "final_train_loss",
        "validation_loss",
        "anomaly_threshold",
        "return_train_mean",
        "return_train_std",
    } <= set(metadata.keys())


@patch(FETCH_PATH, return_value=_make_price_data(101))
def test_train_finishes_run_on_failure(_mock_fetch, tmp_path: Path) -> None:
    """CA-11: finish_run is called even when log_model fails."""
    trainer = _make_trainer(tmp_path)
    trainer.experiment_tracker.log_model.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        trainer.train()

    trainer.experiment_tracker.finish_run.assert_called_once()


@patch(TRACKER_PATH)
def test_default_tracker_is_built_from_config(
    mock_tracker_class,
    tmp_path: Path,
) -> None:
    """CA-12: the default tracker receives the project and run config."""
    config = _make_config(tmp_path)

    trainer = AutoencoderTrainer(ticker="AAPL", config=config)

    assert trainer.experiment_tracker is mock_tracker_class.return_value
    tracker_kwargs = mock_tracker_class.call_args.kwargs
    assert tracker_kwargs["project_name"] == config.tracking.project_name
    run_config = tracker_kwargs["config"]
    expected_keys = {
        "ticker",
        "optimizer",
        "loss_function",
        "model",
        *config.to_flat_dict().keys(),
    }
    assert expected_keys <= set(run_config.keys())
    assert run_config["ticker"] == "AAPL"


@patch(FETCH_PATH, return_value=_make_price_data(101))
def test_train_is_reproducible(_mock_fetch, tmp_path: Path) -> None:
    """CA-13: two runs with the same setup yield the same history."""
    first_result = _make_trainer(tmp_path).train()
    second_result = _make_trainer(tmp_path).train()

    assert (
        first_result.training_loss_history
        == second_result.training_loss_history
    )
