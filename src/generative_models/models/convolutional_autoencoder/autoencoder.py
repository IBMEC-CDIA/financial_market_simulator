"""Convolutional autoencoder for reconstruction-based anomaly scoring."""

import torch
from torch import nn

CONVOLUTION_KERNEL_SIZE = 3
CONVOLUTION_STRIDE = 2
CONVOLUTION_PADDING = 1


def compute_convolution_output_length(input_length: int) -> int:
    """Return the length a 1D convolution produces for an input length.

    The kernel size, the stride and the padding are the module
    constants shared by the encoder and the decoder.

    Parameters
    ----------
    input_length : int
        Number of time steps that enter the convolution.

    Returns
    -------
    int
        Number of time steps that leave the convolution.

    Example
    -------
    >>> compute_convolution_output_length(10)
    5
    """
    padded_length = (
        input_length + 2 * CONVOLUTION_PADDING - CONVOLUTION_KERNEL_SIZE
    )

    return padded_length // CONVOLUTION_STRIDE + 1


def compute_transposed_output_padding(
    input_length: int,
    target_length: int,
) -> int:
    """Return the output padding that restores an exact length.

    A transposed convolution is ambiguous: several input lengths map to
    the same output length. The output padding removes the ambiguity, so
    the decoder rebuilds windows with the same length as the encoder
    input, for any window size.

    Parameters
    ----------
    input_length : int
        Number of time steps that enter the transposed convolution.
    target_length : int
        Number of time steps the transposed convolution must produce.

    Returns
    -------
    int
        Output padding to give to `nn.ConvTranspose1d`. It is always
        smaller than the stride.

    Example
    -------
    >>> compute_transposed_output_padding(5, 10)
    1
    """
    natural_length = (
        (input_length - 1) * CONVOLUTION_STRIDE
        - 2 * CONVOLUTION_PADDING
        + CONVOLUTION_KERNEL_SIZE
    )

    return target_length - natural_length


class ConvolutionalAutoencoder(nn.Module):
    """1D convolutional autoencoder over windows of normalized returns.

    The reconstruction error of a window is used as an anomaly score:
    windows that the model cannot reproduce well are unusual with
    respect to the training data. Unlike a dense autoencoder, the
    encoder slides shared filters over the window, so it looks for
    local shapes in the series instead of learning one weight per time
    step.

    The input and the output are two-dimensional, of shape
    `(batch_size, input_dim)`, like the dense autoencoder of
    `base_autoencoder`. The channel dimension that the convolutions
    need is added and removed inside `forward`, so the training
    pipeline and the anomaly scoring do not change.

    Parameters
    ----------
    input_dim : int
        Number of time steps in each input window. The decoder outputs
        the same number of values.
    hidden_channels : int
        Number of filters of each of the two convolutional layers of
        the encoder and of the decoder.
    latent_dim : int
        Size of the latent representation produced by the encoder.

    Attributes
    ----------
    hidden_channels : int
        Number of filters of the convolutional layers.
    encoded_length : int
        Number of time steps that remain after the two encoder
        convolutions.
    encoder_convolutions : nn.Sequential
        Two strided convolutions that halve the window length and read
        local patterns.
    encoder_projection : nn.Sequential
        Flattens the feature maps and projects them to the latent
        vector. It ends with a ReLU, so the latent values are never
        negative.
    decoder_projection : nn.Sequential
        Projects the latent vector back to the flattened feature maps.
    decoder_convolutions : nn.Sequential
        Two transposed convolutions that restore the window length. The
        last one has no activation, so the reconstructed values may be
        negative.

    Example
    -------
    >>> model = ConvolutionalAutoencoder(input_dim=10)
    >>> window_batch = torch.randn(8, 10)
    >>> model(window_batch).shape
    torch.Size([8, 10])
    >>> model.encode(window_batch).shape
    torch.Size([8, 4])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 16,
        latent_dim: int = 4,
    ) -> None:
        """Build the convolutional encoder and decoder layers."""
        super().__init__()

        first_length = compute_convolution_output_length(input_dim)
        second_length = compute_convolution_output_length(first_length)
        flattened_size = hidden_channels * second_length

        self.hidden_channels = hidden_channels
        self.encoded_length = second_length

        self.encoder_convolutions = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=hidden_channels,
                kernel_size=CONVOLUTION_KERNEL_SIZE,
                stride=CONVOLUTION_STRIDE,
                padding=CONVOLUTION_PADDING,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=CONVOLUTION_KERNEL_SIZE,
                stride=CONVOLUTION_STRIDE,
                padding=CONVOLUTION_PADDING,
            ),
            nn.ReLU(),
        )
        self.encoder_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, latent_dim),
            nn.ReLU(),
        )
        self.decoder_projection = nn.Sequential(
            nn.Linear(latent_dim, flattened_size),
            nn.ReLU(),
        )
        self.decoder_convolutions = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=CONVOLUTION_KERNEL_SIZE,
                stride=CONVOLUTION_STRIDE,
                padding=CONVOLUTION_PADDING,
                output_padding=compute_transposed_output_padding(
                    second_length,
                    first_length,
                ),
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=CONVOLUTION_KERNEL_SIZE,
                stride=CONVOLUTION_STRIDE,
                padding=CONVOLUTION_PADDING,
                output_padding=compute_transposed_output_padding(
                    first_length,
                    input_dim,
                ),
            ),
        )

    def encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return the latent representation of a batch of windows.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor of shape `(batch_size, input_dim)` with the windows
            to encode.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_size, latent_dim)` with
            non-negative values.
        """
        channel_tensor = input_tensor.unsqueeze(1)
        feature_tensor = self.encoder_convolutions(channel_tensor)

        return self.encoder_projection(feature_tensor)

    def decode(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """Rebuild a batch of windows from their latent representation.

        Parameters
        ----------
        latent_tensor : torch.Tensor
            Tensor of shape `(batch_size, latent_dim)` produced by
            `encode`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_size, input_dim)` with the
            reconstructed windows.
        """
        flattened_tensor = self.decoder_projection(latent_tensor)
        feature_tensor = flattened_tensor.view(
            -1,
            self.hidden_channels,
            self.encoded_length,
        )

        return self.decoder_convolutions(feature_tensor).squeeze(1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of windows.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor of shape `(batch_size, input_dim)` with the windows
            to reconstruct.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_size, input_dim)` with the
            reconstructed windows.
        """
        latent_tensor = self.encode(input_tensor)

        return self.decode(latent_tensor)
