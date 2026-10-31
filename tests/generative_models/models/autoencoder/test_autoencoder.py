"""Unit tests for the SimpleAutoencoder architecture."""

import torch

from generative_models.models.autoencoder.autoencoder import (
    SimpleAutoencoder,
)


def test_forward_preserves_input_shape() -> None:
    """Check that forward reconstructs windows with the input shape."""
    model = SimpleAutoencoder(input_dim=10)
    input_tensor = torch.randn(8, 10)

    output_tensor = model(input_tensor)

    assert output_tensor.shape == (8, 10)


def test_encode_returns_latent_shape() -> None:
    """Check that encode returns one latent vector per window."""
    model = SimpleAutoencoder(input_dim=10, latent_dim=3)
    input_tensor = torch.randn(8, 10)

    latent_tensor = model.encode(input_tensor)

    assert latent_tensor.shape == (8, 3)


def test_encode_output_is_never_negative() -> None:
    """Check that the latent representation is non-negative (ReLU)."""
    model = SimpleAutoencoder(input_dim=10)
    input_tensor = torch.randn(64, 10)

    latent_tensor = model.encode(input_tensor)

    assert bool((latent_tensor >= 0).all())


def test_forward_is_deterministic_in_eval_mode() -> None:
    """Check that two forward passes in eval mode give identical output."""
    model = SimpleAutoencoder(input_dim=10)
    model.eval()
    input_tensor = torch.randn(8, 10)

    first_output = model(input_tensor)
    second_output = model(input_tensor)

    assert torch.equal(first_output, second_output)


def test_default_sizes_match_the_specification() -> None:
    """Check that hidden_dim defaults to 16 and latent_dim to 4."""
    model = SimpleAutoencoder(input_dim=10)

    assert model.encoder[0].out_features == 16
    assert model.encoder[2].out_features == 4


def test_state_dict_keys_match_the_specification() -> None:
    """Check that the state_dict keys are the expected layer names."""
    model = SimpleAutoencoder(input_dim=10)

    expected_keys = {
        "encoder.0.weight",
        "encoder.0.bias",
        "encoder.2.weight",
        "encoder.2.bias",
        "decoder.0.weight",
        "decoder.0.bias",
        "decoder.2.weight",
        "decoder.2.bias",
    }
    assert set(model.state_dict().keys()) == expected_keys
