"""Unit tests for the CNN autoencoder training pipeline."""

# pylint: disable=duplicate-code

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from generative_models.data.market_data import (
    add_log_return_column,
    add_split_column,
)
from generative_models.models.convolutional_autoencoder.autoencoder import (
    ConvolutionalAutoencoder,
)
from generative_models.models.convolutional_autoencoder.config import (
    ConvolutionalAutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
    TrackingConfig,
)
from generative_models.models.convolutional_autoencoder.trainer import (
    ConvolutionalAutoencoderTrainer,
    build_ticker_slug,
    compute_mean_baseline_loss,
    compute_reconstruction_scores,
)

TRAINER_MODULE = (
    "generative_models.models.convolutional_autoencoder.trainer"
)
FETCH_PRICE_SERIES_PATH = f"{TRAINER_MODULE}.fetch_price_series"
TRACKER_PATH = f"{TRAINER_MODULE}.WandbExperimentTracker"


def _make_price_frame(number_of_prices: int) -> pd.DataFrame:
    """Build a synthetic positive price series in the yfinance format."""
    rng = np.random.default_rng(0)
    price_values = 100 * np.exp(
        np.cumsum(rng.normal(0.0, 0.01, number_of_prices))
    )

    return pd.DataFrame({"close_price": price_values})


def _make_config(
    output_directory,
) -> ConvolutionalAutoencoderTrainingConfig:
    """Build a small training configuration for fast unit tests."""
    return ConvolutionalAutoencoderTrainingConfig(
        market_data=MarketDataConfig(
            window_size=5,
            batch_size=8,
            train_split_ratio=0.8,
        ),
        optimization=OptimizationConfig(number_of_epochs=3),
        tracking=TrackingConfig(output_directory=str(output_directory)),
        device="cpu",
    )


def _make_trainer(
    tmp_path,
    experiment_tracker=None,
) -> ConvolutionalAutoencoderTrainer:
    """Create a trainer for PETR4.SA with a mocked tracker by default."""
    if experiment_tracker is None:
        experiment_tracker = MagicMock()

    return ConvolutionalAutoencoderTrainer(
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
    model = ConvolutionalAutoencoder(input_dim=5)
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

    assert trainer.model_name == "convolutional-autoencoder-petr4.sa"
    assert trainer.model_file_path == (
        tmp_path / "convolutional-autoencoder-petr4.sa.pt"
    )


def test_build_model_uses_the_configured_architecture(tmp_path) -> None:
    """Check that the channel count and the latent size reach the model."""
    trainer = _make_trainer(tmp_path)

    model = trainer.build_model()

    assert isinstance(model, ConvolutionalAutoencoder)
    assert model.hidden_channels == 16
    assert model.encoder_projection[1].out_features == 4


@patch(FETCH_PRICE_SERIES_PATH)
def test_prepare_data_uses_given_prices_without_downloading(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that prices passed to the trainer are used instead of fetched."""
    trainer = ConvolutionalAutoencoderTrainer(
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
    trainer = ConvolutionalAutoencoderTrainer(
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
    trainer = ConvolutionalAutoencoderTrainer(
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


def test_compute_mean_baseline_loss_is_the_mean_square(tmp_path) -> None:
    """Check the naive baseline on a dataset with known values."""
    trainer = ConvolutionalAutoencoderTrainer(
        ticker="PETR4.SA",
        config=_make_config(tmp_path),
        experiment_tracker=MagicMock(),
        price_data=add_split_column(
            add_log_return_column(_make_price_frame(101)),
            train_split_ratio=0.5,
            validation_split_ratio=0.25,
        ),
    )

    prepared_data = trainer.prepare_data()

    assert compute_mean_baseline_loss(prepared_data.test_dataset) > 0


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
    summary_metrics = [
        metrics for metrics in logged_metrics if "epoch" not in metrics
    ]
    assert [metrics["epoch"] for metrics in epoch_metrics] == [1, 2, 3]
    assert all("train/loss_mse" in metrics for metrics in epoch_metrics)
    assert all(
        "validation/loss_mse" in metrics for metrics in epoch_metrics
    )
    assert len(summary_metrics) == 1


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
def test_train_reports_the_architecture_for_the_serving_fields(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that the result carries the sizes the report needs."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    trainer = _make_trainer(tmp_path)

    result = trainer.train()

    assert result.window_size == 5
    assert result.hidden_layer_size == 16
    assert result.latent_layer_size == 4


@patch(FETCH_PRICE_SERIES_PATH)
def test_log_model_registers_the_experiment_alias(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that the model is registered under the experiment alias."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    experiment_tracker = MagicMock()
    trainer = _make_trainer(tmp_path, experiment_tracker)

    trainer.train()

    aliases = experiment_tracker.log_model.call_args.kwargs["aliases"]
    assert aliases == [trainer.config.tracking.experiment_name]


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
        "best_epoch",
        "stopped_at_epoch",
        "ticker",
        "window_size",
        "model_family",
        "architecture",
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

    ConvolutionalAutoencoderTrainer(ticker="PETR4.SA", config=config)

    call_kwargs = mock_tracker_class.call_args.kwargs
    assert call_kwargs["project_name"] == config.tracking.project_name
    run_config = call_kwargs["config"]
    assert run_config["ticker"] == "PETR4.SA"
    assert run_config["optimizer"] == "Adam"
    assert run_config["loss_function"] == "MSE"
    assert run_config["model"] == "ConvolutionalAutoencoder"
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


def _make_patience_config(
    output_directory,
    number_of_epochs: int,
    patience: int | None,
) -> ConvolutionalAutoencoderTrainingConfig:
    """Build a configuration with the given epoch limit and patience."""
    return ConvolutionalAutoencoderTrainingConfig(
        market_data=MarketDataConfig(
            window_size=5,
            batch_size=8,
            train_split_ratio=0.8,
        ),
        optimization=OptimizationConfig(
            number_of_epochs=number_of_epochs,
            early_stopping_patience=patience,
        ),
        tracking=TrackingConfig(output_directory=str(output_directory)),
        device="cpu",
    )


def _logged_validation_losses(experiment_tracker) -> list[float]:
    """Collect the per-epoch validation losses sent to the tracker."""
    return [
        call.args[0]["validation/loss_mse"]
        for call in experiment_tracker.log_metrics.call_args_list
        if "epoch" in call.args[0]
    ]


@patch(FETCH_PRICE_SERIES_PATH)
def test_early_stopping_ends_training_before_the_epoch_limit(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that a small patience stops well before the last epoch."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    experiment_tracker = MagicMock()
    trainer = ConvolutionalAutoencoderTrainer(
        ticker="PETR4.SA",
        config=_make_patience_config(tmp_path, 200, 2),
        experiment_tracker=experiment_tracker,
    )

    result = trainer.train()

    assert len(result.training_loss_history) < 200
    metadata = experiment_tracker.log_model.call_args.kwargs["metadata"]
    assert metadata["stopped_at_epoch"] == len(result.training_loss_history)
    assert metadata["best_epoch"] <= metadata["stopped_at_epoch"]


@patch(FETCH_PRICE_SERIES_PATH)
def test_early_stopping_restores_the_best_validation_weights(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that the kept weights are the ones of the best epoch."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    experiment_tracker = MagicMock()
    trainer = ConvolutionalAutoencoderTrainer(
        ticker="PETR4.SA",
        config=_make_patience_config(tmp_path, 60, 5),
        experiment_tracker=experiment_tracker,
    )

    result = trainer.train()

    validation_losses = _logged_validation_losses(experiment_tracker)
    assert result.validation_loss == pytest.approx(min(validation_losses))


@patch(FETCH_PRICE_SERIES_PATH)
def test_patience_of_none_runs_every_epoch(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that early stopping can be switched off."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    trainer = ConvolutionalAutoencoderTrainer(
        ticker="PETR4.SA",
        config=_make_patience_config(tmp_path, 7, None),
        experiment_tracker=MagicMock(),
    )

    result = trainer.train()

    assert len(result.training_loss_history) == 7


@patch(f"{TRAINER_MODULE}.torch.optim.Adam")
@patch(FETCH_PRICE_SERIES_PATH)
def test_weight_decay_reaches_the_optimizer(
    mock_fetch_price_series, mock_adam, tmp_path
) -> None:
    """Check that the configured L2 penalty is given to Adam."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    config = _make_patience_config(tmp_path, 1, None)
    config.optimization.weight_decay = 0.25
    trainer = ConvolutionalAutoencoderTrainer(
        ticker="PETR4.SA",
        config=config,
        experiment_tracker=MagicMock(),
    )

    trainer.train()

    assert mock_adam.call_args.kwargs["weight_decay"] == 0.25


@patch(FETCH_PRICE_SERIES_PATH)
def test_train_writes_the_weights_directory(
    mock_fetch_price_series, tmp_path
) -> None:
    """Check that the directory of the weights file is created."""
    mock_fetch_price_series.return_value = _make_price_frame(101)
    output_directory = tmp_path / "nested" / "artifacts"
    config = ConvolutionalAutoencoderTrainingConfig(
        market_data=MarketDataConfig(
            window_size=5,
            batch_size=8,
            train_split_ratio=0.8,
        ),
        optimization=OptimizationConfig(number_of_epochs=1),
        tracking=TrackingConfig(output_directory=str(output_directory)),
        device="cpu",
    )
    trainer = ConvolutionalAutoencoderTrainer(
        ticker="PETR4.SA",
        config=config,
        experiment_tracker=MagicMock(),
    )

    trainer.train()

    assert output_directory.is_dir()
