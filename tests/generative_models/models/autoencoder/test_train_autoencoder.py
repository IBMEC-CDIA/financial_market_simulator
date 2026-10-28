"""Unit tests for the autoencoder training command-line script."""

from unittest.mock import patch

import pytest

from generative_models.models.autoencoder.config import (
    AutoencoderTrainingConfig,
)
from generative_models.models.autoencoder.train_autoencoder import (
    build_argument_parser,
    build_training_config,
    main,
)

TRAINER_PATH = (
    "generative_models.models.autoencoder.train_autoencoder."
    "AutoencoderTrainer"
)


def _parse_config(
    command_line_arguments: list[str],
) -> AutoencoderTrainingConfig:
    """Parse arguments and build the training configuration."""
    arguments = build_argument_parser().parse_args(command_line_arguments)
    return build_training_config(arguments)


def test_missing_ticker_exits() -> None:
    """CA-01: parsing without --ticker raises SystemExit."""
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args([])


def test_ticker_only_uses_default_config() -> None:
    """CA-02: only --ticker yields the default configuration."""
    assert _parse_config(["--ticker", "AAPL"]) == AutoencoderTrainingConfig()


def test_arguments_override_config_fields() -> None:
    """CA-03: optional arguments override the matching fields."""
    config = _parse_config(
        [
            "--ticker",
            "AAPL",
            "--epochs",
            "50",
            "--window-size",
            "20",
            "--learning-rate",
            "0.01",
            "--device",
            "cpu",
        ]
    )

    assert config.optimization.number_of_epochs == 50
    assert config.market_data.window_size == 20
    assert config.optimization.learning_rate == 0.01
    assert config.device == "cpu"


def test_arguments_are_converted_to_numeric_types() -> None:
    """CA-04: ratio and contamination are floats, batch size is int."""
    config = _parse_config(
        [
            "--ticker",
            "AAPL",
            "--train-split-ratio",
            "0.7",
            "--contamination",
            "0.1",
            "--batch-size",
            "16",
        ]
    )

    assert isinstance(config.market_data.train_split_ratio, float)
    assert config.market_data.train_split_ratio == 0.7
    assert isinstance(config.contamination, float)
    assert config.contamination == 0.1
    assert isinstance(config.market_data.batch_size, int)
    assert config.market_data.batch_size == 16


@patch(TRAINER_PATH)
def test_main_trains_with_parsed_config(mock_trainer_class) -> None:
    """CA-05: main builds the trainer, calls train once, returns result."""
    mock_trainer = mock_trainer_class.return_value
    mock_trainer.train.return_value.training_loss_history = [0.5]
    mock_trainer.train.return_value.validation_loss = 0.4
    mock_trainer.train.return_value.anomaly_threshold = 1.2

    result = main(["--ticker", "AAPL", "--epochs", "2"])

    trainer_kwargs = mock_trainer_class.call_args.kwargs
    assert trainer_kwargs["ticker"] == "AAPL"
    assert trainer_kwargs["config"].optimization.number_of_epochs == 2
    mock_trainer.train.assert_called_once()
    assert result is mock_trainer.train.return_value


@patch(TRAINER_PATH)
def test_main_rejects_invalid_config(mock_trainer_class) -> None:
    """CA-06: --epochs 0 raises ValueError before building the trainer."""
    with pytest.raises(ValueError, match="number_of_epochs"):
        main(["--ticker", "AAPL", "--epochs", "0"])

    mock_trainer_class.assert_not_called()
