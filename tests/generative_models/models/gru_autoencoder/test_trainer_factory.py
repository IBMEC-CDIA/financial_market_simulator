"""Unit tests for the autoencoder trainer factory."""

# pylint: disable=duplicate-code

from unittest.mock import patch

import pandas as pd
import pytest

from generative_models.models.gru_autoencoder.config import (
    GRUAutoencoderTrainingConfig,
)
from generative_models.models.gru_autoencoder.trainer_factory import (
    build_autoencoder_config,
    build_trainer,
)
from generative_models.models.training_options import TrainingOptions

TRAINER_PATH = (
    "generative_models.models.gru_autoencoder.trainer_factory."
    "GRUAutoencoderTrainer"
)


@patch(TRAINER_PATH)
def test_build_trainer_passes_ticker_and_prices(mock_trainer_class) -> None:
    """Check that the trainer receives the ticker and the prepared prices."""
    price_data = pd.DataFrame({"close_price": [1.0, 2.0]})

    trainer = build_trainer("AAPL", price_data)

    assert trainer is mock_trainer_class.return_value
    kwargs = mock_trainer_class.call_args.kwargs
    assert kwargs["ticker"] == "AAPL"
    assert kwargs["price_data"] is price_data


@patch(TRAINER_PATH)
def test_build_trainer_without_prices_lets_the_trainer_download(
    mock_trainer_class,
) -> None:
    """Check that no prices are given when the caller passes none."""
    build_trainer("AAPL")

    assert mock_trainer_class.call_args.kwargs["price_data"] is None


def test_build_autoencoder_config_maps_the_shared_options() -> None:
    """Check that the shared options set the autoencoder configuration."""
    options = TrainingOptions(
        start_date="2021-01-01",
        epochs=12,
        learning_rate=0.01,
        window_size=20,
        output_directory="out",
        device="cpu",
    )

    config = build_autoencoder_config(options)

    assert config.market_data.start_date == "2021-01-01"
    assert config.market_data.window_size == 20
    assert config.optimization.number_of_epochs == 12
    assert config.optimization.learning_rate == pytest.approx(0.01)
    assert config.tracking.output_directory == "out"
    assert config.device == "cpu"


def test_build_autoencoder_config_without_options_is_the_default() -> None:
    """Check that no options give the default autoencoder configuration."""
    assert build_autoencoder_config(None) == GRUAutoencoderTrainingConfig()
