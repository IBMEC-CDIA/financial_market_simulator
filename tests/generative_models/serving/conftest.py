"""Shared factories for the serving unit tests."""

import numpy as np
import pandas as pd
import pytest
import torch

from generative_models.models.autoencoder.autoencoder import (
    SimpleAutoencoder,
)
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPredictor,
)


def _build_predictor(
    ticker: str = "PETR4.SA",
    version: str = "v1",
    anomaly_threshold: float = 1.0,
    return_train_mean: float = 0.0,
    return_train_std: float = 1.0,
) -> AutoencoderPredictor:
    """Build a predictor with an untrained model and window size 5."""
    torch.manual_seed(0)
    metadata = AutoencoderModelMetadata(
        ticker=ticker,
        window_size=5,
        return_train_mean=return_train_mean,
        return_train_std=return_train_std,
        anomaly_threshold=anomaly_threshold,
        artifact_name=f"simple-autoencoder-{ticker.lower()}:{version}",
    )

    return AutoencoderPredictor(
        model=SimpleAutoencoder(input_dim=5),
        metadata=metadata,
    )


def _build_price_frame(number_of_prices: int) -> pd.DataFrame:
    """Build synthetic prices indexed by business days from 2026-09-01."""
    dates = pd.bdate_range("2026-09-01", periods=number_of_prices)
    prices = 100 * np.exp(np.cumsum(np.full(number_of_prices, 0.01)))

    return pd.DataFrame({"close_price": prices}, index=dates)


@pytest.fixture(name="make_predictor")
def make_predictor_fixture():
    """Provide a factory that builds predictors with window size 5.

    Returns
    -------
    callable
        Function that accepts `ticker`, `version`, `anomaly_threshold`,
        `return_train_mean` and `return_train_std` and returns an
        `AutoencoderPredictor`.
    """
    return _build_predictor


@pytest.fixture(name="make_price_frame")
def make_price_frame_fixture():
    """Provide a factory that builds synthetic daily closing prices.

    Returns
    -------
    callable
        Function that accepts `number_of_prices` and returns a data
        frame with a `close_price` column, indexed by business days.
    """
    return _build_price_frame
