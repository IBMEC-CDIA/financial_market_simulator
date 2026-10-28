"""Unit tests for the dense autoencoder training pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from generative_models.models.autoencoder.autoencoder import (
    SimpleAutoencoder,
)
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

TRAINER_MODULE = "generative_models.models.autoencoder.trainer"


def build_price_dataframe(number_of_prices: int) -> pd.DataFrame:
    """Build a synthetic positive price series for the tests.

    Parameters
    ----------
    number_of_prices : int
        Number of daily prices to generate.

    Returns
    -------
    pd.DataFrame
        Single-column data frame indexed by business day, holding the
        synthetic prices in the close_price column.
    """
    random_generator = np.random.default_rng(7)
    log_increments = random_generator.normal(
        loc=0.0,
        scale=0.01,
        size=number_of_prices,
    )
    price_values = 100.0 * np.exp(np.cumsum(log_increments))
    price_dates = pd.bdate_range("2020-01-01", periods=number_of_prices)

    return pd.DataFrame({"close_price": price_values}, index=price_dates)


def build_small_config(output_directory: Path) -> AutoencoderTrainingConfig:
    """Build a fast CPU-only configuration for the tests.

    Parameters
    ----------
    output_directory : Path
        Directory where the model weights are written.

    Returns
    -------
    AutoencoderTrainingConfig
        Configuration with small windows, batches and epoch count.
    """
    return AutoencoderTrainingConfig(
        market_data=MarketDataConfig(window_size=5, batch_size=8),
        optimization=OptimizationConfig(number_of_epochs=3),
        tracking=TrackingConfig(output_directory=str(output_directory)),
        device="cpu",
    )


def build_trainer(
    output_directory: Path,
    experiment_tracker: MagicMock,
) -> AutoencoderTrainer:
    """Build a trainer wired to a mocked experiment tracker.

    Parameters
    ----------
    output_directory : Path
        Directory where the model weights are written.
    experiment_tracker : MagicMock
        Stand-in for the wandb experiment tracker.

    Returns
    -------
    AutoencoderTrainer
        Trainer configured for the PETR4.SA ticker.
    """
    return AutoencoderTrainer(
        ticker="PETR4.SA",
        config=build_small_config(output_directory),
        experiment_tracker=experiment_tracker,
    )


def test_build_ticker_slug_normalizes_symbols() -> None:
    """CA-01: the slug is lowercase and keeps only safe characters."""
    assert build_ticker_slug("PETR4.SA") == "petr4.sa"
    assert build_ticker_slug("^BVSP") == "-bvsp"


def test_compute_reconstruction_scores_returns_one_score_per_window() -> None:
    """CA-02: seven windows of size five yield seven non-negative scores."""
    model = SimpleAutoencoder(input_dim=5)
    input_tensor = torch.randn(7, 5)

    scores = compute_reconstruction_scores(model, input_tensor)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (7,)
    assert bool((scores >= 0).all())


def test_model_name_and_file_path_depend_on_ticker(tmp_path: Path) -> None:
    """CA-03: the artifact name and weights path carry the ticker slug."""
    trainer = build_trainer(tmp_path / "artifacts", MagicMock())

    assert trainer.model_name == "simple-autoencoder-petr4.sa"
    assert trainer.model_file_path == (
        tmp_path / "artifacts" / "simple-autoencoder-petr4.sa.pt"
    )


def test_prepare_data_splits_windows_chronologically(tmp_path: Path) -> None:
    """CA-04: 101 prices produce 76, 16 and 96 windows."""
    trainer = build_trainer(tmp_path / "artifacts", MagicMock())

    with patch(
        f"{TRAINER_MODULE}.fetch_price_series",
        return_value=build_price_dataframe(101),
    ):
        prepared_data = trainer.prepare_data()

    assert len(prepared_data.train_dataset) == 76
    assert len(prepared_data.validation_dataset) == 16
    assert len(prepared_data.full_dataset) == 96


def test_prepare_data_normalizes_with_train_statistics(
    tmp_path: Path,
) -> None:
    """CA-05: the training series is centered and its deviation positive."""
    trainer = build_trainer(tmp_path / "artifacts", MagicMock())

    with patch(
        f"{TRAINER_MODULE}.fetch_price_series",
        return_value=build_price_dataframe(101),
    ):
        prepared_data = trainer.prepare_data()

    train_series = prepared_data.train_dataset.series_values

    assert float(train_series.mean()) == pytest.approx(0.0, abs=1e-5)
    assert prepared_data.return_train_std > 0


def test_prepare_data_with_too_few_prices_raises_value_error(
    tmp_path: Path,
) -> None:
    """CA-06: a series shorter than the window raises ValueError."""
    trainer = build_trainer(tmp_path / "artifacts", MagicMock())

    with patch(
        f"{TRAINER_MODULE}.fetch_price_series",
        return_value=build_price_dataframe(12),
    ):
        with pytest.raises(ValueError, match="Not enough data"):
            trainer.prepare_data()


def test_train_drives_the_tracker_lifecycle(tmp_path: Path) -> None:
    """CA-07: train calls start_run, log_model and finish_run once each."""
    experiment_tracker = MagicMock()
    trainer = build_trainer(tmp_path / "artifacts", experiment_tracker)

    with patch(
        f"{TRAINER_MODULE}.fetch_price_series",
        return_value=build_price_dataframe(101),
    ):
        trainer.train()

    experiment_tracker.start_run.assert_called_once()
    experiment_tracker.log_model.assert_called_once()
    experiment_tracker.finish_run.assert_called_once()


def test_train_logs_metrics_for_every_epoch(tmp_path: Path) -> None:
    """CA-08: three epochs are logged plus a single validation metric."""
    experiment_tracker = MagicMock()
    trainer = build_trainer(tmp_path / "artifacts", experiment_tracker)

    with patch(
        f"{TRAINER_MODULE}.fetch_price_series",
        return_value=build_price_dataframe(101),
    ):
        trainer.train()

    logged_metrics = [
        call.args[0]
        for call in experiment_tracker.log_metrics.call_args_list
    ]
    epoch_metrics = [
        metrics for metrics in logged_metrics if "epoch" in metrics
    ]
    validation_metrics = [
        metrics
        for metrics in logged_metrics
        if "validation/loss_mse" in metrics
    ]

    assert [metrics["epoch"] for metrics in epoch_metrics] == [1, 2, 3]
    assert all("train/loss_mse" in metrics for metrics in epoch_metrics)
    assert len(validation_metrics) == 1


def test_train_returns_complete_result(tmp_path: Path) -> None:
    """CA-09: the result carries history, validation loss and threshold."""
    trainer = build_trainer(tmp_path / "artifacts", MagicMock())

    with patch(
        f"{TRAINER_MODULE}.fetch_price_series",
        return_value=build_price_dataframe(101),
    ):
        result = trainer.train()

    assert len(result.training_loss_history) == 3
    assert np.isfinite(result.validation_loss)
    assert isinstance(result.anomaly_threshold, float)
    assert result.anomaly_threshold > 0


def test_train_logs_inference_metadata(tmp_path: Path) -> None:
    """CA-10: log_model receives the five metadata keys serving needs."""
    experiment_tracker = MagicMock()
    trainer = build_trainer(tmp_path / "artifacts", experiment_tracker)

    with patch(
        f"{TRAINER_MODULE}.fetch_price_series",
        return_value=build_price_dataframe(101),
    ):
        trainer.train()

    metadata = experiment_tracker.log_model.call_args.kwargs["metadata"]

    assert set(metadata) == {
        "final_train_loss",
        "validation_loss",
        "anomaly_threshold",
        "return_train_mean",
        "return_train_std",
    }


def test_train_finishes_the_run_when_log_model_fails(tmp_path: Path) -> None:
    """CA-11: a failing log_model still finishes the run and propagates."""
    experiment_tracker = MagicMock()
    experiment_tracker.log_model.side_effect = RuntimeError("artifact failed")
    trainer = build_trainer(tmp_path / "artifacts", experiment_tracker)

    with patch(
        f"{TRAINER_MODULE}.fetch_price_series",
        return_value=build_price_dataframe(101),
    ):
        with pytest.raises(RuntimeError, match="artifact failed"):
            trainer.train()

    experiment_tracker.finish_run.assert_called_once()


def test_default_tracker_is_built_from_the_configuration(
    tmp_path: Path,
) -> None:
    """CA-12: without a tracker one is built from the tracking config."""
    config = build_small_config(tmp_path / "artifacts")

    with patch(f"{TRAINER_MODULE}.WandbExperimentTracker") as tracker_class:
        AutoencoderTrainer(ticker="PETR4.SA", config=config)

    keyword_arguments = tracker_class.call_args.kwargs
    run_config = keyword_arguments["config"]

    assert keyword_arguments["project_name"] == config.tracking.project_name
    assert run_config["ticker"] == "PETR4.SA"
    assert set(config.to_flat_dict()).issubset(run_config)
    assert run_config["optimizer"] == "Adam"
    assert run_config["loss_function"] == "MSE"
    assert run_config["model"] == "SimpleAutoencoder"


def test_train_is_reproducible(tmp_path: Path) -> None:
    """CA-13: two runs with the same seed share the same loss history."""
    price_data = build_price_dataframe(101)

    with patch(
        f"{TRAINER_MODULE}.fetch_price_series",
        return_value=price_data,
    ):
        first_trainer = build_trainer(tmp_path / "first", MagicMock())
        second_trainer = build_trainer(tmp_path / "second", MagicMock())
        first_result = first_trainer.train()
        second_result = second_trainer.train()

    assert (
        first_result.training_loss_history
        == second_result.training_loss_history
    )
