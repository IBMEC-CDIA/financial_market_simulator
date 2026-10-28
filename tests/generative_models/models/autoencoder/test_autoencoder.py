"""Unit tests for SimpleAutoencoder."""

import torch

from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder


def test_forward_returns_same_shape_as_input() -> None:
    """CA-01: check that forward maps (8, 10) to (8, 10)."""
    model = SimpleAutoencoder(input_dim=10)
    input_tensor = torch.randn(8, 10)

    reconstructed_tensor = model(input_tensor)

    assert reconstructed_tensor.shape == torch.Size([8, 10])


def test_encode_returns_latent_shape() -> None:
    """CA-02: check that encode maps (8, 10) to (8, latent_dim)."""
    model = SimpleAutoencoder(input_dim=10, latent_dim=3)
    input_tensor = torch.randn(8, 10)

    latent_tensor = model.encode(input_tensor)

    assert latent_tensor.shape == torch.Size([8, 3])


def test_encode_returns_non_negative_values() -> None:
    """CA-03: check that the latent representation is never negative."""
    model = SimpleAutoencoder(input_dim=10)
    input_tensor = torch.randn(64, 10) * 5

    latent_tensor = model.encode(input_tensor)

    assert bool((latent_tensor >= 0).all())


def test_forward_is_deterministic_in_eval_mode() -> None:
    """CA-04: check that two forward passes match in eval mode."""
    model = SimpleAutoencoder(input_dim=10)
    model.eval()
    input_tensor = torch.randn(4, 10)

    with torch.no_grad():
        first_reconstruction = model(input_tensor)
        second_reconstruction = model(input_tensor)

    assert torch.equal(first_reconstruction, second_reconstruction)


def test_default_hidden_and_latent_dimensions() -> None:
    """CA-05: check the default hidden_dim=16 and latent_dim=4."""
    model = SimpleAutoencoder(input_dim=10)

    assert model.encoder[0].out_features == 16
    assert model.encoder[2].out_features == 4


def test_state_dict_keys_match_expected_layout() -> None:
    """CA-06: check the exact state_dict keys of the architecture."""
    model = SimpleAutoencoder(input_dim=10)

    assert list(model.state_dict().keys()) == [
        "encoder.0.weight",
        "encoder.0.bias",
        "encoder.2.weight",
        "encoder.2.bias",
        "decoder.0.weight",
        "decoder.0.bias",
        "decoder.2.weight",
        "decoder.2.bias",
    ]
