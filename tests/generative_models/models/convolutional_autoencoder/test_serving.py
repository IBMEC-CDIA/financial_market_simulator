"""Unit tests for the serving hooks of the convolutional autoencoder."""

import numpy as np
import torch

from generative_models.models.convolutional_autoencoder import serving
from generative_models.models.convolutional_autoencoder.autoencoder import (
    ConvolutionalAutoencoder,
)


def test_build_model_creates_a_convolutional_autoencoder() -> None:
    """Check that the architecture is the constructor of the CNN model."""
    model = serving.build_model(
        {"input_dim": 10, "hidden_channels": 16, "latent_dim": 4}
    )

    assert isinstance(model, ConvolutionalAutoencoder)


def test_score_windows_returns_one_score_per_window() -> None:
    """Check that every window gets a reconstruction score."""
    model = serving.build_model(
        {"input_dim": 10, "hidden_channels": 16, "latent_dim": 4}
    )

    scores = serving.score_windows(model, torch.zeros(5, 10))

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (5,)
