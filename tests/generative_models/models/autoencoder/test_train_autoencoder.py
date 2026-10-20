"""Unit tests for the autoencoder training script."""

from unittest.mock import patch

import pytest

from generative_models.models.autoencoder.train_autoencoder import (
    build_argument_parser,
    build_training_config,
    main,
)


def test_parser_requires_ticker() -> None:
    """Check that the script fails without a ticker."""
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args([])


def test_build_training_config_uses_defaults() -> None:
    """Check that omitted arguments keep the configuration defaults."""
    arguments = build_argument_parser().parse_args(["--ticker", "AAPL"])

    config = build_training_config(arguments)

    assert config.market_data.window_size == 10
    assert config.optimization.number_of_epochs == 300
    assert config.device is None


def test_build_training_config_applies_overrides() -> None:
    """Check that command-line overrides reach the configuration."""
    arguments = build_argument_parser().parse_args(
        [
            "--ticker",
            "PETR4.SA",
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

    config = build_training_config(arguments)

    assert config.optimization.number_of_epochs == 50
    assert config.market_data.window_size == 20
    assert config.optimization.learning_rate == 0.01
    assert config.device == "cpu"


@patch(
    "generative_models.models.autoencoder.train_autoencoder."
    "AutoencoderTrainer"
)
def test_main_trains_with_given_ticker(mock_trainer_class) -> None:
    """Check that main builds the trainer for the ticker and trains."""
    mock_result = mock_trainer_class.return_value.train.return_value
    mock_result.training_loss_history = [0.5]
    mock_result.validation_loss = 0.6
    mock_result.anomaly_threshold = 1.2

    result = main(["--ticker", "AAPL", "--epochs", "2"])

    trainer_arguments = mock_trainer_class.call_args.kwargs
    assert trainer_arguments["ticker"] == "AAPL"
    assert trainer_arguments["config"].optimization.number_of_epochs == 2
    assert result is mock_result
