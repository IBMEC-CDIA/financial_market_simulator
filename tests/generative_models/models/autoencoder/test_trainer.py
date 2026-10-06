"""Unit tests for AutoencoderTrainer."""

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


def _make_price_data(number_of_prices: int) -> pd.DataFrame:
    """Build a synthetic positive price series indexed by business day."""
    random_generator = np.random.default_rng(0)
    log_returns = random_generator.normal(0.0, 0.01, size=number_of_prices)
    return pd.DataFrame(
        {"close_price": 100.0 * np.exp(np.cumsum(log_returns))},
        index=pd.bdate_range("2024-01-01", periods=number_of_prices),
    )


def _make_config(tmp_path) -> AutoencoderTrainingConfig:
    """Build a small configuration that trains quickly on the CPU."""
    return AutoencoderTrainingConfig(
        market_data=MarketDataConfig(window_size=5, batch_size=8),
        optimization=OptimizationConfig(number_of_epochs=3),
        tracking=TrackingConfig(output_directory=str(tmp_path / "models")),
        device="cpu",
    )


def test_build_ticker_slug_replaces_unsafe_characters() -> None:
    """Check that tickers become lowercase and artifact-safe."""
    assert build_ticker_slug("PETR4.SA") == "petr4.sa"
    assert build_ticker_slug("^BVSP") == "-bvsp"


def test_compute_reconstruction_scores_returns_one_score_per_window() -> None:
    """Check that one non-negative score is produced per window."""
    model = SimpleAutoencoder(input_dim=5)

    scores = compute_reconstruction_scores(model, torch.randn(7, 5))

    assert scores.shape == (7,)
    assert (scores >= 0).all()


def test_model_file_path_uses_ticker_and_output_directory(tmp_path) -> None:
    """Check that the model file is named after the ticker."""
    trainer = AutoencoderTrainer(
        ticker="PETR4.SA",
        config=_make_config(tmp_path),
        experiment_tracker=MagicMock(),
    )

    assert trainer.model_name == "simple-autoencoder-petr4.sa"
    assert trainer.model_file_path == (
        tmp_path / "models" / "simple-autoencoder-petr4.sa.pt"
    )


@patch(FETCH_PATH)
def test_prepare_data_splits_and_normalizes(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check the dataset sizes and training normalization statistics."""
    mock_fetch_price_series.return_value = _make_price_data(101)
    trainer = AutoencoderTrainer(
        ticker="AAPL",
        config=_make_config(tmp_path),
        experiment_tracker=MagicMock(),
    )

    prepared_data = trainer.prepare_data()

    assert len(prepared_data.train_dataset) == 80 - 5 + 1
    assert len(prepared_data.validation_dataset) == 20 - 5 + 1
    assert len(prepared_data.full_dataset) == 100 - 5 + 1
    assert prepared_data.return_train_std > 0
    assert abs(
        prepared_data.train_dataset.series_values.mean().item()
    ) < 1e-5


@patch(FETCH_PATH)
def test_prepare_data_raises_when_series_is_too_short(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that a series too short for the window size is rejected."""
    mock_fetch_price_series.return_value = _make_price_data(12)
    trainer = AutoencoderTrainer(
        ticker="AAPL",
        config=_make_config(tmp_path),
        experiment_tracker=MagicMock(),
    )

    with pytest.raises(ValueError, match="Not enough data"):
        trainer.prepare_data()


@patch(FETCH_PATH)
def test_train_tracks_experiment_and_returns_result(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check the tracker calls and the returned training result."""
    mock_fetch_price_series.return_value = _make_price_data(101)
    experiment_tracker = MagicMock()
    trainer = AutoencoderTrainer(
        ticker="AAPL",
        config=_make_config(tmp_path),
        experiment_tracker=experiment_tracker,
    )

    result = trainer.train()

    mock_fetch_price_series.assert_called_once_with(
        "AAPL",
        "2020-01-01",
        "2026-09-15",
    )
    experiment_tracker.start_run.assert_called_once()
    experiment_tracker.finish_run.assert_called_once()
    assert experiment_tracker.log_metrics.call_count == 3 + 1

    log_model_arguments = experiment_tracker.log_model.call_args.kwargs
    assert log_model_arguments["model"] is result.model
    assert log_model_arguments["model_name"] == "simple-autoencoder-aapl"
    assert log_model_arguments["model_file_path"] == str(
        result.model_file_path
    )
    assert log_model_arguments["metadata"]["anomaly_threshold"] == (
        result.anomaly_threshold
    )

    assert len(result.training_loss_history) == 3
    assert result.validation_loss > 0
    assert result.anomaly_threshold > 0
    assert result.model_file_path.parent.is_dir()


@patch(FETCH_PATH)
def test_train_finishes_run_when_training_fails(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that the wandb run is finished even if training raises."""
    mock_fetch_price_series.return_value = _make_price_data(101)
    experiment_tracker = MagicMock()
    experiment_tracker.log_metrics.side_effect = RuntimeError("boom")
    trainer = AutoencoderTrainer(
        ticker="AAPL",
        config=_make_config(tmp_path),
        experiment_tracker=experiment_tracker,
    )

    with pytest.raises(RuntimeError, match="boom"):
        trainer.train()

    experiment_tracker.finish_run.assert_called_once()


@patch("generative_models.models.autoencoder.trainer.WandbExperimentTracker")
def test_default_tracker_receives_ticker_and_config(
    mock_tracker_class, tmp_path
) -> None:
    """Check that the default tracker is built from the configuration."""
    config = _make_config(tmp_path)

    trainer = AutoencoderTrainer(ticker="MSFT", config=config)

    tracker_arguments = mock_tracker_class.call_args.kwargs
    assert trainer.experiment_tracker is mock_tracker_class.return_value
    assert tracker_arguments["project_name"] == config.tracking.project_name
    assert tracker_arguments["config"]["ticker"] == "MSFT"
    assert tracker_arguments["config"]["window_size"] == 5
    assert tracker_arguments["config"]["optimizer"] == "Adam"
