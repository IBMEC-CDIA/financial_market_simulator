"""Unit tests for the autoencoder training pipeline."""

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
from generative_models.data.market_data import (
    add_log_return_column,
    add_split_column,
)
from generative_models.models.autoencoder.trainer import (
    AutoencoderTrainer,
    build_ticker_slug,
    compute_reconstruction_scores,
)

FETCH_PRICE_SERIES_PATH = (
    "generative_models.models.autoencoder.trainer.fetch_price_series"
)
TRACKER_PATH = (
    "generative_models.models.autoencoder.trainer.WandbExperimentTracker"
)


def _make_price_frame(number_of_prices: int) -> pd.DataFrame:
    """Build a synthetic positive price series in the yfinance format."""
    rng = np.random.default_rng(0)
    price_values = 100 * np.exp(
        np.cumsum(rng.normal(0.0, 0.01, number_of_prices))
    )

    return pd.DataFrame({"close_price": price_values})


def _make_config(output_directory) -> AutoencoderTrainingConfig:
    """Build a small training configuration for fast unit tests."""
    return AutoencoderTrainingConfig(
        market_data=MarketDataConfig(
            window_size=5,
            batch_size=8,
            train_split_ratio=0.8,
        ),
        optimization=OptimizationConfig(number_of_epochs=3),
        tracking=TrackingConfig(output_directory=str(output_directory)),
        device="cpu",
    )


def _make_trainer(tmp_path, experiment_tracker=None) -> AutoencoderTrainer:
    """Create a trainer for PETR4.SA with a mocked tracker by default."""
    if experiment_tracker is None:
        experiment_tracker = MagicMock()

    return AutoencoderTrainer(
        ticker="PETR4.SA",
        config=_make_config(tmp_path),
        experiment_tracker=experiment_tracker,
    )


def test_build_ticker_slug_keeps_dots_and_replaces_symbols() -> None:
    """Check the slug for a Brazilian ticker and for an index symbol."""
    assert build_ticker_slug("PETR4.SA") == "petr4.sa"
    assert build_ticker_slug("^BVSP") == "-bvsp"


def test_compute_reconstruction_scores_returns_one_score_per_window() -> None:
    """Check the shape and sign of the per-window reconstruction scores."""
    model = SimpleAutoencoder(input_dim=5)
    window_tensor = torch.randn(7, 5)

    reconstruction_scores = compute_reconstruction_scores(
        model,
        window_tensor,
    )

    assert isinstance(reconstruction_scores, np.ndarray)
    assert reconstruction_scores.shape == (7,)
    assert bool((reconstruction_scores >= 0).all())


def test_model_name_and_file_path_depend_on_ticker(tmp_path) -> None:
    """Check the artifact name and the weights path for a ticker."""
    trainer = _make_trainer(tmp_path)

    assert trainer.model_name == "simple-autoencoder-petr4.sa"
    assert trainer.model_file_path == (
        tmp_path / "simple-autoencoder-petr4.sa.pt"
    )


@patch(FETCH_PRICE_SERIES_PATH)
def test_prepare_data_uses_given_prices_without_downloading(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that prices passed to the trainer are used instead of fetched."""
    trainer = AutoencoderTrainer(
        ticker="PETR4.SA",
        config=_make_config(tmp_path),
        experiment_tracker=MagicMock(),
        price_data=_make_price_frame(101),
    )

    prepared_data = trainer.prepare_data()

    mock_fetch_price_series.assert_not_called()
    assert len(prepared_data.train_dataset) == 76
    assert len(prepared_data.validation_dataset) == 16


@patch(FETCH_PRICE_SERIES_PATH)
def test_prepare_data_reuses_stored_log_returns(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that a stored log_return column gives the same windows."""
    price_data = add_log_return_column(_make_price_frame(101))
    trainer = AutoencoderTrainer(
        ticker="PETR4.SA",
        config=_make_config(tmp_path),
        experiment_tracker=MagicMock(),
        price_data=price_data,
    )

    prepared_data = trainer.prepare_data()

    mock_fetch_price_series.assert_not_called()
    assert len(prepared_data.train_dataset) == 76
    assert len(prepared_data.validation_dataset) == 16
    assert len(prepared_data.full_dataset) == 96


@patch(FETCH_PRICE_SERIES_PATH)
def test_prepare_data_uses_the_stored_split_column(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that the split column sets the train, validation and test sets."""
    price_data = add_split_column(
        add_log_return_column(_make_price_frame(101)),
        train_split_ratio=0.5,
        validation_split_ratio=0.25,
    )
    trainer = AutoencoderTrainer(
        ticker="PETR4.SA",
        config=_make_config(tmp_path),
        experiment_tracker=MagicMock(),
        price_data=price_data,
    )

    prepared_data = trainer.prepare_data()

    mock_fetch_price_series.assert_not_called()
    assert len(prepared_data.train_dataset) == 46
    assert len(prepared_data.validation_dataset) == 21
    assert len(prepared_data.test_dataset) == 21
    assert len(prepared_data.full_dataset) == 96


@patch(FETCH_PRICE_SERIES_PATH)
def test_prepare_data_builds_expected_window_counts(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check the window counts of the train, validation and full sets."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    trainer = _make_trainer(tmp_path)

    prepared_data = trainer.prepare_data()

    assert len(prepared_data.train_dataset) == 76
    assert len(prepared_data.validation_dataset) == 16
    assert len(prepared_data.full_dataset) == 96


@patch(FETCH_PRICE_SERIES_PATH)
def test_prepare_data_normalizes_training_series_to_zero_mean(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that the training series is centered and has a valid std."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    trainer = _make_trainer(tmp_path)

    prepared_data = trainer.prepare_data()
    training_series = prepared_data.train_dataset.series_values

    assert abs(training_series.mean().item()) < 1e-5
    assert prepared_data.return_train_std > 0


@patch(FETCH_PRICE_SERIES_PATH)
def test_prepare_data_raises_when_not_enough_data(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that a too short series is rejected with a clear message."""
    mock_fetch_price_series.return_value = _make_price_frame(12)
    trainer = _make_trainer(tmp_path)

    with pytest.raises(ValueError, match="Not enough data"):
        trainer.prepare_data()


@patch(FETCH_PRICE_SERIES_PATH)
def test_train_starts_and_finishes_the_run_once(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that the wandb run is started, logged and finished once."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    experiment_tracker = MagicMock()
    trainer = _make_trainer(tmp_path, experiment_tracker)

    trainer.train()

    experiment_tracker.start_run.assert_called_once()
    experiment_tracker.log_model.assert_called_once()
    experiment_tracker.finish_run.assert_called_once()


@patch(FETCH_PRICE_SERIES_PATH)
def test_train_logs_metrics_for_every_epoch_and_validation(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check the epoch metrics and the single validation metric."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    experiment_tracker = MagicMock()
    trainer = _make_trainer(tmp_path, experiment_tracker)

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


@patch(FETCH_PRICE_SERIES_PATH)
def test_train_returns_finite_results(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check the loss history length and the finite result values."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    trainer = _make_trainer(tmp_path)

    result = trainer.train()

    assert len(result.training_loss_history) == 3
    assert np.isfinite(result.validation_loss)
    assert isinstance(result.anomaly_threshold, float)
    assert result.anomaly_threshold > 0
    assert result.model_file_path == trainer.model_file_path


@patch(FETCH_PRICE_SERIES_PATH)
def test_log_model_metadata_contains_training_summary(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that the artifact metadata includes the six summary keys."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    experiment_tracker = MagicMock()
    trainer = _make_trainer(tmp_path, experiment_tracker)

    trainer.train()

    metadata = experiment_tracker.log_model.call_args.kwargs["metadata"]
    assert set(metadata.keys()) == {
        "final_train_loss",
        "train_evaluation_loss",
        "validation_loss",
        "anomaly_threshold",
        "return_train_mean",
        "return_train_std",
    }


@patch(FETCH_PRICE_SERIES_PATH)
def test_train_finishes_run_even_when_log_model_fails(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that finish_run is called and the error is propagated."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    experiment_tracker = MagicMock()
    experiment_tracker.log_model.side_effect = RuntimeError("upload failed")
    trainer = _make_trainer(tmp_path, experiment_tracker)

    with pytest.raises(RuntimeError, match="upload failed"):
        trainer.train()

    experiment_tracker.finish_run.assert_called_once()


@patch(TRACKER_PATH)
def test_default_tracker_receives_project_and_full_config(
    mock_tracker_class, tmp_path
) -> None:
    """Check the tracker created when no tracker is given."""
    config = _make_config(tmp_path)

    AutoencoderTrainer(ticker="PETR4.SA", config=config)

    call_kwargs = mock_tracker_class.call_args.kwargs
    assert call_kwargs["project_name"] == config.tracking.project_name
    run_config = call_kwargs["config"]
    assert run_config["ticker"] == "PETR4.SA"
    assert run_config["optimizer"] == "Adam"
    assert run_config["loss_function"] == "MSE"
    assert run_config["model"] == "SimpleAutoencoder"
    for key, value in config.to_flat_dict().items():
        assert run_config[key] == value


@patch(FETCH_PRICE_SERIES_PATH)
def test_train_is_reproducible_with_the_same_seed(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that two runs with the same seed give the same losses."""
    mock_fetch_price_series.return_value = _make_price_frame(101)

    first_result = _make_trainer(tmp_path, MagicMock()).train()
    second_result = _make_trainer(tmp_path, MagicMock()).train()

    np.testing.assert_allclose(
        first_result.training_loss_history,
        second_result.training_loss_history,
    )
