"""Unit tests for SimpleAutoencoder."""

import torch

from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder


def test_forward_preserves_input_shape() -> None:
    """CA-01: forward maps (8, 10) inputs to (8, 10) outputs."""
    model = SimpleAutoencoder(input_dim=10)

    output_tensor = model(torch.randn(8, 10))

    assert output_tensor.shape == (8, 10)


def test_encode_returns_latent_shape() -> None:
    """CA-02: encode maps (8, 10) inputs to (8, latent_dim)."""
    model = SimpleAutoencoder(input_dim=10, latent_dim=3)

    latent_tensor = model.encode(torch.randn(8, 10))

    assert latent_tensor.shape == (8, 3)


def test_encode_returns_non_negative_values() -> None:
    """CA-03: the latent representation is never negative."""
    torch.manual_seed(0)
    model = SimpleAutoencoder(input_dim=10)

    latent_tensor = model.encode(torch.randn(64, 10))

    assert torch.all(latent_tensor >= 0)


def test_forward_is_deterministic_in_eval_mode() -> None:
    """CA-04: two eval-mode forward calls return identical tensors."""
    model = SimpleAutoencoder(input_dim=10)
    model.eval()
    input_tensor = torch.randn(8, 10)

    first_output = model(input_tensor)
    second_output = model(input_tensor)

    assert torch.equal(first_output, second_output)


def test_default_layer_sizes() -> None:
    """CA-05: hidden_dim defaults to 16 and latent_dim to 4."""
    model = SimpleAutoencoder(input_dim=10)

    assert model.encoder[0].out_features == 16
    assert model.encoder[2].out_features == 4


def test_state_dict_keys() -> None:
    """CA-06: state_dict keys match the serving contract."""
    model = SimpleAutoencoder(input_dim=10)

    assert set(model.state_dict().keys()) == {
        "encoder.0.weight",
        "encoder.0.bias",
        "encoder.2.weight",
        "encoder.2.bias",
        "decoder.0.weight",
        "decoder.0.bias",
        "decoder.2.weight",
        "decoder.2.bias",
    }
