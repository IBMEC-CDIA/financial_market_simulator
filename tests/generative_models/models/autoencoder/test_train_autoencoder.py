"""Unit tests for the autoencoder training command line script."""

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

SCRIPT_MODULE = "generative_models.models.autoencoder.train_autoencoder"


def parse_arguments(command_line_arguments: list[str]):
    """Parse command line arguments with the script parser.

    Parameters
    ----------
    command_line_arguments : list[str]
        Arguments to parse, without the program name.

    Returns
    -------
    argparse.Namespace
        Namespace with the parsed arguments.
    """
    return build_argument_parser().parse_args(command_line_arguments)


def test_missing_ticker_exits() -> None:
    """CA-01: parsing an empty argument list raises SystemExit."""
    with pytest.raises(SystemExit):
        parse_arguments([])


def test_only_ticker_uses_every_default() -> None:
    """CA-02: passing only the ticker reproduces the default config."""
    arguments = parse_arguments(["--ticker", "AAPL"])

    config = build_training_config(arguments)

    assert config == AutoencoderTrainingConfig()


def test_overridden_arguments_reach_the_configuration() -> None:
    """CA-03: explicit arguments override the configuration defaults."""
    arguments = parse_arguments(
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

    config = build_training_config(arguments)

    assert config.optimization.number_of_epochs == 50
    assert config.market_data.window_size == 20
    assert config.optimization.learning_rate == 0.01
    assert config.device == "cpu"


def test_numeric_arguments_are_converted() -> None:
    """CA-04: ratios are parsed as float and the batch size as int."""
    arguments = parse_arguments(
        [
            "--ticker",
            "AAPL",
            "--train-split-ratio",
            "0.7",
            "--contamination",
            "0.1",
            "--batch-size",
            "64",
        ]
    )

    config = build_training_config(arguments)

    assert isinstance(config.market_data.train_split_ratio, float)
    assert config.market_data.train_split_ratio == 0.7
    assert isinstance(config.contamination, float)
    assert config.contamination == 0.1
    assert isinstance(config.market_data.batch_size, int)
    assert config.market_data.batch_size == 64


def test_main_builds_the_trainer_and_returns_its_result() -> None:
    """CA-05: main wires the trainer, trains once and returns the result."""
    training_result = MagicMock()
    training_result.training_loss_history = [0.5, 0.25]
    training_result.validation_loss = 0.3
    training_result.anomaly_threshold = 0.9
    training_result.model_file_path = "artifacts/simple-autoencoder-aapl.pt"

    with patch(f"{SCRIPT_MODULE}.AutoencoderTrainer") as trainer_class:
        trainer_class.return_value.train.return_value = training_result
        result = main(["--ticker", "AAPL", "--epochs", "2"])

    keyword_arguments = trainer_class.call_args.kwargs

    assert keyword_arguments["ticker"] == "AAPL"
    assert keyword_arguments["config"].optimization.number_of_epochs == 2
    trainer_class.return_value.train.assert_called_once()
    assert result is training_result


def test_main_rejects_invalid_configuration() -> None:
    """CA-06: an invalid epoch count fails before the trainer is built."""
    with patch(f"{SCRIPT_MODULE}.AutoencoderTrainer") as trainer_class:
        with pytest.raises(ValueError, match="number_of_epochs"):
            main(["--ticker", "AAPL", "--epochs", "0"])

    trainer_class.assert_not_called()
