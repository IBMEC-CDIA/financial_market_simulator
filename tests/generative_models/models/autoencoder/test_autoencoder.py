"""Unit tests for SimpleAutoencoder."""

import torch

from generative_models.models.autoencoder.autoencoder import (
    SimpleAutoencoder,
)


def test_forward_preserves_input_shape() -> None:
    """CA-01: input (8, 10) produces output (8, 10) in forward."""
    model = SimpleAutoencoder(input_dim=10)
    input_tensor = torch.randn(8, 10)

    reconstruction = model(input_tensor)

    assert reconstruction.shape == torch.Size([8, 10])


def test_encode_returns_latent_dimension() -> None:
    """CA-02: encode of an (8, 10) input produces (8, 3)."""
    model = SimpleAutoencoder(input_dim=10, latent_dim=3)
    input_tensor = torch.randn(8, 10)

    latent_representation = model.encode(input_tensor)

    assert latent_representation.shape == torch.Size([8, 3])


def test_encode_returns_non_negative_values() -> None:
    """CA-03: every value returned by encode is greater or equal to 0."""
    model = SimpleAutoencoder(input_dim=10)
    input_tensor = torch.randn(32, 10)

    latent_representation = model.encode(input_tensor)

    assert bool((latent_representation >= 0).all())


def test_forward_is_deterministic_in_eval_mode() -> None:
    """CA-04: in eval mode two forward calls return identical tensors."""
    model = SimpleAutoencoder(input_dim=10)
    model.eval()
    input_tensor = torch.randn(8, 10)

    first_reconstruction = model(input_tensor)
    second_reconstruction = model(input_tensor)

    assert torch.equal(first_reconstruction, second_reconstruction)


def test_default_layer_sizes() -> None:
    """CA-05: default hidden_dim is 16 and default latent_dim is 4."""
    model = SimpleAutoencoder(input_dim=10)

    assert model.encoder[0].out_features == 16
    assert model.encoder[2].out_features == 4


def test_state_dict_keys_are_stable() -> None:
    """CA-06: the state_dict keys are exactly the eight expected keys."""
    model = SimpleAutoencoder(input_dim=10)

    expected_keys = [
        "encoder.0.weight",
        "encoder.0.bias",
        "encoder.2.weight",
        "encoder.2.bias",
        "decoder.0.weight",
        "decoder.0.bias",
        "decoder.2.weight",
        "decoder.2.bias",
    ]

    assert list(model.state_dict().keys()) == expected_keys
