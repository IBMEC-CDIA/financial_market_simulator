"""Unit tests for SimpleAutoencoder."""

import torch

from generative_models.models.autoencoder.simple_autoencoder import (
    SimpleAutoencoder,
)


def test_forward_preserves_input_shape() -> None:
    """Check that the reconstruction has the same shape as the input."""
    model = SimpleAutoencoder(input_dim=10)

    reconstruction = model(torch.randn(8, 10))

    assert reconstruction.shape == (8, 10)


def test_encoder_output_has_latent_dimension() -> None:
    """Check that the encoder compresses windows to latent_dim values."""
    model = SimpleAutoencoder(input_dim=10, hidden_dim=12, latent_dim=3)

    latent_representation = model.encoder(torch.randn(5, 10))

    assert latent_representation.shape == (5, 3)


def test_backward_pass_produces_gradients() -> None:
    """Check that the reconstruction loss is differentiable."""
    model = SimpleAutoencoder(input_dim=6)
    input_tensor = torch.randn(4, 6)

    reconstruction = model(input_tensor)
    loss_value = torch.nn.functional.mse_loss(reconstruction, input_tensor)
    loss_value.backward()

    assert all(
        parameter.grad is not None for parameter in model.parameters()
    )
