"""Unit tests for the GRUAutoencoder architecture."""

import torch

from generative_models.models.gru_autoencoder.autoencoder import (
    GRUAutoencoder,
)


def test_forward_preserves_input_shape() -> None:
    """Check that forward reconstructs windows with the input shape."""
    model = GRUAutoencoder(input_dim=10)
    input_tensor = torch.randn(8, 10)

    output_tensor = model(input_tensor)

    assert output_tensor.shape == (8, 10)


def test_encode_returns_latent_shape() -> None:
    """Check that encode returns one latent vector per window."""
    model = GRUAutoencoder(input_dim=10, latent_dim=3)
    input_tensor = torch.randn(8, 10)

    latent_tensor = model.encode(input_tensor)

    assert latent_tensor.shape == (8, 3)


def test_forward_is_deterministic_in_eval_mode() -> None:
    """Check that repeated eval-mode passes reconstruct the same values."""
    gru_model = GRUAutoencoder(input_dim=6).eval()
    window_batch = torch.randn(5, 6)

    reconstructions = [gru_model(window_batch) for _ in range(2)]

    assert torch.equal(reconstructions[0], reconstructions[1])


def test_default_sizes() -> None:
    """Check that hidden_dim defaults to 16 and latent_dim to 4."""
    model = GRUAutoencoder(input_dim=10)

    assert model.encoder_gru.hidden_size == 16
    assert model.encoder_projection.out_features == 4
    assert model.decoder_gru.hidden_size == 16


def test_reconstruction_can_be_negative() -> None:
    """Check that the output has no final activation."""
    model = GRUAutoencoder(input_dim=10)
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.fill_(-1.0)

    output_tensor = model(torch.randn(4, 10))

    assert torch.equal(output_tensor, torch.full((4, 10), -1.0))


def test_gradients_reach_every_parameter() -> None:
    """Check that a reconstruction loss trains every layer."""
    model = GRUAutoencoder(input_dim=10)
    input_tensor = torch.randn(4, 10)

    loss = ((model(input_tensor) - input_tensor) ** 2).mean()
    loss.backward()

    assert all(
        parameter.grad is not None for parameter in model.parameters()
    )


def test_state_dict_can_be_reloaded() -> None:
    """Check that saved weights rebuild an identical model."""
    model = GRUAutoencoder(input_dim=10, hidden_dim=8, latent_dim=2)
    reloaded_model = GRUAutoencoder(input_dim=10, hidden_dim=8, latent_dim=2)
    reloaded_model.load_state_dict(model.state_dict())
    model.eval()
    reloaded_model.eval()
    input_tensor = torch.randn(3, 10)

    assert torch.equal(model(input_tensor), reloaded_model(input_tensor))
