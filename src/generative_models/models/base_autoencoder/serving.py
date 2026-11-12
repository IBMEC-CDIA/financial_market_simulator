"""Serving hooks of the base autoencoder used by the serving layer.

The serving layer rebuilds a stored model and scores its windows through
the two functions of this module. Every model family exposes the same two
names in its own `serving` module.
"""

import numpy as np
import torch
from torch import nn

from generative_models.models.base_autoencoder.autoencoder import (
    SimpleAutoencoder,
)
from generative_models.models.base_autoencoder.trainer import (
    compute_reconstruction_scores,
)


def build_model(architecture: dict[str, int]) -> nn.Module:
    """Create the untrained network of the base autoencoder.

    Parameters
    ----------
    architecture : dict[str, int]
        Constructor arguments of `SimpleAutoencoder`: `input_dim`,
        `hidden_dim` and `latent_dim`.

    Returns
    -------
    nn.Module
        Untrained `SimpleAutoencoder`.

    Example
    -------
    >>> model = build_model(
    ...     {"input_dim": 10, "hidden_dim": 16, "latent_dim": 4}
    ... )
    """
    return SimpleAutoencoder(**architecture)


def score_windows(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    """Return the reconstruction score of each window.

    Parameters
    ----------
    model : nn.Module
        Trained autoencoder built by `build_model`.
    input_tensor : torch.Tensor
        Normalized windows with shape `(number_of_windows, window_size)`.

    Returns
    -------
    np.ndarray
        One reconstruction error per window.
    """
    return compute_reconstruction_scores(model, input_tensor)
