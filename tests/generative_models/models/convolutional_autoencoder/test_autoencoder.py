"""Unit tests for the ConvolutionalAutoencoder architecture."""

# pylint: disable=duplicate-code

import pytest
import torch

from generative_models.models.convolutional_autoencoder.autoencoder import (
    CONVOLUTION_STRIDE,
    ConvolutionalAutoencoder,
    compute_convolution_output_length,
    compute_transposed_output_padding,
)

WINDOW_SIZES = [2, 3, 4, 5, 7, 10, 16, 20, 30, 63, 100]


def test_compute_convolution_output_length_halves_the_window() -> None:
    """Check the output length of the strided convolution."""
    assert compute_convolution_output_length(10) == 5
    assert compute_convolution_output_length(5) == 3
    assert compute_convolution_output_length(4) == 2


def test_compute_transposed_output_padding_restores_the_length() -> None:
    """Check the output padding that rebuilds the original length."""
    assert compute_transposed_output_padding(5, 10) == 1
    assert compute_transposed_output_padding(3, 5) == 0


@pytest.mark.parametrize("window_size", WINDOW_SIZES)
def test_forward_preserves_input_shape(window_size: int) -> None:
    """Check that forward reconstructs windows with the input shape."""
    model = ConvolutionalAutoencoder(input_dim=window_size)
    input_tensor = torch.randn(8, window_size)

    output_tensor = model(input_tensor)

    assert output_tensor.shape == (8, window_size)


@pytest.mark.parametrize("window_size", WINDOW_SIZES)
def test_output_padding_stays_below_the_stride(window_size: int) -> None:
    """Check that torch accepts the computed output padding."""
    first_length = compute_convolution_output_length(window_size)
    second_length = compute_convolution_output_length(first_length)

    first_padding = compute_transposed_output_padding(
        second_length,
        first_length,
    )
    second_padding = compute_transposed_output_padding(
        first_length,
        window_size,
    )

    assert 0 <= first_padding < CONVOLUTION_STRIDE
    assert 0 <= second_padding < CONVOLUTION_STRIDE


def test_encode_returns_latent_shape() -> None:
    """Check that encode returns one latent vector per window."""
    model = ConvolutionalAutoencoder(input_dim=10, latent_dim=3)
    input_tensor = torch.randn(8, 10)

    latent_tensor = model.encode(input_tensor)

    assert latent_tensor.shape == (8, 3)


def test_encode_output_is_never_negative() -> None:
    """Check that the latent representation is non-negative (ReLU)."""
    model = ConvolutionalAutoencoder(input_dim=10)
    input_tensor = torch.randn(64, 10)

    latent_tensor = model.encode(input_tensor)

    assert bool((latent_tensor >= 0).all())


def test_decode_rebuilds_the_window_shape() -> None:
    """Check that decode maps a latent batch back to windows."""
    model = ConvolutionalAutoencoder(input_dim=10, latent_dim=3)
    latent_tensor = torch.randn(8, 3)

    output_tensor = model.decode(latent_tensor)

    assert output_tensor.shape == (8, 10)


def test_forward_is_deterministic_in_eval_mode() -> None:
    """Check that two forward passes in eval mode give identical output."""
    model = ConvolutionalAutoencoder(input_dim=10)
    model.eval()
    input_tensor = torch.randn(8, 10)

    first_output = model(input_tensor)
    second_output = model(input_tensor)

    assert torch.equal(first_output, second_output)


def test_default_sizes_match_the_specification() -> None:
    """Check that hidden_channels defaults to 16 and latent_dim to 4."""
    model = ConvolutionalAutoencoder(input_dim=10)

    assert model.encoder_convolutions[0].out_channels == 16
    assert model.encoder_projection[1].out_features == 4


def test_encoded_length_matches_two_convolutions() -> None:
    """Check the stored length of the encoded feature maps."""
    model = ConvolutionalAutoencoder(input_dim=10)

    assert model.encoded_length == 3
    assert model.hidden_channels == 16


def test_decoder_ends_with_a_single_channel() -> None:
    """Check that the decoder outputs one channel, as the input has."""
    model = ConvolutionalAutoencoder(input_dim=10)

    assert model.decoder_convolutions[2].out_channels == 1


def test_state_dict_keys_match_the_specification() -> None:
    """Check that the state_dict keys are the expected layer names."""
    model = ConvolutionalAutoencoder(input_dim=10)

    expected_keys = {
        "encoder_convolutions.0.weight",
        "encoder_convolutions.0.bias",
        "encoder_convolutions.2.weight",
        "encoder_convolutions.2.bias",
        "encoder_projection.1.weight",
        "encoder_projection.1.bias",
        "decoder_projection.0.weight",
        "decoder_projection.0.bias",
        "decoder_convolutions.0.weight",
        "decoder_convolutions.0.bias",
        "decoder_convolutions.2.weight",
        "decoder_convolutions.2.bias",
    }
    assert set(model.state_dict().keys()) == expected_keys


def test_filters_are_shared_across_time_steps() -> None:
    """Check that the encoder has one kernel per filter, not per step."""
    model = ConvolutionalAutoencoder(input_dim=50, hidden_channels=8)

    first_convolution = model.encoder_convolutions[0]

    assert first_convolution.weight.shape == (8, 1, 3)
