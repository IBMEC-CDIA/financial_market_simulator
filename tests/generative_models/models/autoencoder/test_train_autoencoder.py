"""Unit tests for the autoencoder training command line script."""

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

TRAINER_CLASS_PATH = (
    "generative_models.models.autoencoder.train_autoencoder."
    "AutoencoderTrainer"
)


def test_parser_requires_ticker() -> None:
    """CA-01: check that parsing without --ticker exits."""
    argument_parser = build_argument_parser()

    with pytest.raises(SystemExit):
        argument_parser.parse_args([])


def test_only_ticker_builds_default_config() -> None:
    """CA-02: check that only --ticker yields the default config."""
    arguments = build_argument_parser().parse_args(["--ticker", "AAPL"])

    training_config = build_training_config(arguments)

    assert training_config == AutoencoderTrainingConfig()


def test_arguments_override_config_fields() -> None:
    """CA-03: check that arguments override the matching fields."""
    arguments = build_argument_parser().parse_args(
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

    training_config = build_training_config(arguments)

    assert training_config.optimization.number_of_epochs == 50
    assert training_config.market_data.window_size == 20
    assert training_config.optimization.learning_rate == 0.01
    assert training_config.device == "cpu"


def test_arguments_are_converted_to_numeric_types() -> None:
    """CA-04: check the float and int conversion of numeric arguments."""
    arguments = build_argument_parser().parse_args(
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

    training_config = build_training_config(arguments)

    assert isinstance(training_config.market_data.train_split_ratio, float)
    assert training_config.market_data.train_split_ratio == 0.7
    assert isinstance(training_config.contamination, float)
    assert training_config.contamination == 0.1
    assert isinstance(training_config.market_data.batch_size, int)
    assert training_config.market_data.batch_size == 64


@patch(TRAINER_CLASS_PATH)
def test_main_trains_with_parsed_config(mock_trainer_class) -> None:
    """CA-05: check that main builds the trainer and returns its result."""
    training_result = main(["--ticker", "AAPL", "--epochs", "2"])

    trainer_kwargs = mock_trainer_class.call_args.kwargs
    mock_train = mock_trainer_class.return_value.train
    assert trainer_kwargs["ticker"] == "AAPL"
    assert trainer_kwargs["config"].optimization.number_of_epochs == 2
    mock_train.assert_called_once()
    assert training_result is mock_train.return_value


@patch(TRAINER_CLASS_PATH)
def test_main_rejects_invalid_config_before_training(
    mock_trainer_class,
) -> None:
    """CA-06: check that --epochs 0 raises before building the trainer."""
    with pytest.raises(ValueError, match="number_of_epochs"):
        main(["--ticker", "AAPL", "--epochs", "0"])

    mock_trainer_class.assert_not_called()
