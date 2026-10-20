"""Unit tests for SimpleAutoencoder."""

import torch

from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder


def test_forward_preserves_input_shape() -> None:
    """Check that the reconstruction has the same shape as the input."""
    model = SimpleAutoencoder(
        input_dim=10,
        hidden_dim=16,
        latent_dim=4,
    )
    input_windows = torch.randn(8, 10)

    reconstruction = model(input_windows)

    assert reconstruction.shape == input_windows.shape


def test_encode_returns_latent_dimension() -> None:
    """Check that encode maps (batch, input_dim) to (batch, latent_dim)."""
    model = SimpleAutoencoder(
        input_dim=12,
        hidden_dim=8,
        latent_dim=3,
    )

    latent_representation = model.encode(torch.randn(5, 12))

    assert latent_representation.shape == torch.Size([5, 3])


def test_forward_is_deterministic_in_eval_mode() -> None:
    """Check that two forward passes with the same input match."""
    model = SimpleAutoencoder(input_dim=6)
    model.eval()
    input_windows = torch.randn(4, 6)

    with torch.no_grad():
        first_reconstruction = model(input_windows)
        second_reconstruction = model(input_windows)

    assert torch.equal(first_reconstruction, second_reconstruction)
