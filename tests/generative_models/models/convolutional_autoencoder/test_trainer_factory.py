"""Unit tests for the CNN autoencoder trainer factory."""

# pylint: disable=duplicate-code

from unittest.mock import patch

import pandas as pd
import pytest

from generative_models.models.convolutional_autoencoder.config import (
    ConvolutionalAutoencoderTrainingConfig,
)
from generative_models.models.convolutional_autoencoder import (
    trainer_factory,
)
from generative_models.models.training_options import TrainingOptions

TRAINER_PATH = (
    "generative_models.models.convolutional_autoencoder.trainer_factory."
    "ConvolutionalAutoencoderTrainer"
)


@patch(TRAINER_PATH)
def test_build_trainer_passes_ticker_and_prices(mock_trainer_class) -> None:
    """Check that the trainer receives the ticker and the prepared prices."""
    price_data = pd.DataFrame({"close_price": [1.0, 2.0]})

    trainer = trainer_factory.build_trainer("AAPL", price_data)

    assert trainer is mock_trainer_class.return_value
    kwargs = mock_trainer_class.call_args.kwargs
    assert kwargs["ticker"] == "AAPL"
    assert kwargs["price_data"] is price_data


@patch(TRAINER_PATH)
def test_build_trainer_without_prices_lets_the_trainer_download(
    mock_trainer_class,
) -> None:
    """Check that no prices are given when the caller passes none."""
    trainer_factory.build_trainer("AAPL")

    assert mock_trainer_class.call_args.kwargs["price_data"] is None


@patch(TRAINER_PATH)
def test_build_trainer_forwards_the_options_to_the_config(
    mock_trainer_class,
) -> None:
    """Check that the shared options reach the trainer configuration."""
    options = TrainingOptions(epochs=7, window_size=15)

    trainer_factory.build_trainer("AAPL", None, options)

    config = mock_trainer_class.call_args.kwargs["config"]
    assert config.optimization.number_of_epochs == 7
    assert config.market_data.window_size == 15


def test_build_convolutional_config_maps_the_shared_options() -> None:
    """Check that the shared options set the model configuration."""
    options = TrainingOptions(
        start_date="2021-01-01",
        epochs=12,
        learning_rate=0.01,
        window_size=20,
        output_directory="out",
        device="cpu",
    )

    config = trainer_factory.build_convolutional_config(options)

    assert config.market_data.start_date == "2021-01-01"
    assert config.market_data.window_size == 20
    assert config.optimization.number_of_epochs == 12
    assert config.optimization.learning_rate == pytest.approx(0.01)
    assert config.tracking.output_directory == "out"
    assert config.device == "cpu"


def test_build_convolutional_config_keeps_the_experiment_name() -> None:
    """Check that the experiment of the options reaches the tracking."""
    options = TrainingOptions(experiment_name="convolutional-autoencoder")

    config = trainer_factory.build_convolutional_config(options)

    assert config.tracking.experiment_name == "convolutional-autoencoder"


def test_build_convolutional_config_without_options_is_default() -> None:
    """Check that no options give the default model configuration."""
    assert (
        trainer_factory.build_convolutional_config(None)
        == ConvolutionalAutoencoderTrainingConfig()
    )


def test_build_trainer_follows_the_shared_contract() -> None:
    """Check the signature the shared training command relies on."""
    trainer = trainer_factory.build_trainer(
        "AAPL",
        pd.DataFrame({"close_price": [1.0, 2.0]}),
        TrainingOptions(epochs=1, device="cpu"),
    )

    assert hasattr(trainer, "train")
    assert trainer.ticker == "AAPL"
