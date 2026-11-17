"""Unit tests for the serving hooks of the GRU autoencoder."""

# pylint: disable=duplicate-code

import numpy as np
import torch

from generative_models.models.gru_autoencoder import serving
from generative_models.models.gru_autoencoder.autoencoder import (
    GRUAutoencoder,
)


def test_build_model_creates_a_gru_autoencoder() -> None:
    """Check that the architecture is the constructor of the GRU model."""
    model = serving.build_model(
        {"input_dim": 10, "hidden_dim": 8, "latent_dim": 4}
    )

    assert isinstance(model, GRUAutoencoder)


def test_score_windows_returns_one_score_per_window() -> None:
    """Check that every window gets a reconstruction score."""
    model = serving.build_model(
        {"input_dim": 10, "hidden_dim": 8, "latent_dim": 4}
    )

    scores = serving.score_windows(model, torch.zeros(5, 10))

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (5,)
