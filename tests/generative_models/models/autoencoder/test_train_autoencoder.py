"""Unit tests for the autoencoder training command-line script."""

from unittest.mock import MagicMock, patch

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


def _parse_configuration(command_line_arguments: list[str]):
    """Parse arguments and build the matching training configuration."""
    parser = build_argument_parser()
    arguments = parser.parse_args(command_line_arguments)

    return build_training_config(arguments)


def test_missing_ticker_raises_system_exit() -> None:
    """Check that argparse rejects a call without --ticker."""
    parser = build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_only_ticker_gives_default_configuration() -> None:
    """Check that the defaults match AutoencoderTrainingConfig()."""
    training_config = _parse_configuration(["--ticker", "AAPL"])

    assert training_config == AutoencoderTrainingConfig()


def test_cli_options_override_the_configuration() -> None:
    """Check that the epochs, window, learning rate and device are set."""
    training_config = _parse_configuration(
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

    assert training_config.optimization.number_of_epochs == 50
    assert training_config.market_data.window_size == 20
    assert training_config.optimization.learning_rate == 0.01
    assert training_config.device == "cpu"


def test_numeric_options_are_converted_to_their_types() -> None:
    """Check that float and int options are converted correctly."""
    training_config = _parse_configuration(
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

    assert isinstance(training_config.market_data.train_split_ratio, float)
    assert isinstance(training_config.contamination, float)
    assert isinstance(training_config.market_data.batch_size, int)
    assert training_config.market_data.batch_size == 16


@patch(TRAINER_PATH)
def test_main_creates_trainer_and_returns_its_result(
    mock_trainer_class,
) -> None:
    """Check that main builds the trainer and returns the train result."""
    mock_trainer = mock_trainer_class.return_value
    mock_trainer.train.return_value.training_loss_history = [0.5, 0.25]
    mock_trainer.train.return_value.validation_loss = 0.3
    mock_trainer.train.return_value.anomaly_threshold = 0.9
    mock_trainer.train.return_value.model_file_path = "model.pt"

    result = main(["--ticker", "AAPL", "--epochs", "2"])

    trainer_kwargs = mock_trainer_class.call_args.kwargs
    assert trainer_kwargs["ticker"] == "AAPL"
    assert trainer_kwargs["config"].optimization.number_of_epochs == 2
    mock_trainer.train.assert_called_once()
    assert result is mock_trainer.train.return_value


@patch(TRAINER_PATH)
def test_main_rejects_invalid_epochs_before_creating_trainer(
    mock_trainer_class,
) -> None:
    """Check that an invalid epoch count fails before any training."""
    with pytest.raises(ValueError, match="number_of_epochs"):
        main(["--ticker", "AAPL", "--epochs", "0"])

    mock_trainer_class.assert_not_called()


@patch(TRAINER_PATH)
def test_main_prints_a_summary_line(mock_trainer_class, capsys) -> None:
    """Check that the summary line contains the ticker and the path."""
    mock_trainer = MagicMock()
    mock_trainer.train.return_value.training_loss_history = [0.5]
    mock_trainer.train.return_value.validation_loss = 0.3
    mock_trainer.train.return_value.anomaly_threshold = 0.9
    mock_trainer.train.return_value.model_file_path = "model.pt"
    mock_trainer_class.return_value = mock_trainer

    main(["--ticker", "AAPL", "--epochs", "1"])

    printed_output = capsys.readouterr().out
    assert "ticker=AAPL" in printed_output
    assert "model_file_path=model.pt" in printed_output
