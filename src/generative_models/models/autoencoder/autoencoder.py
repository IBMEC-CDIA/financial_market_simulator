"""Dense autoencoder used to reconstruct log-return windows."""

import torch
from torch import nn


class SimpleAutoencoder(nn.Module):
    """A simple fully connected (dense) autoencoder.

    The network is composed of an encoder that compresses the input
    window into a lower-dimensional latent representation, and a
    decoder that reconstructs the original window from that
    representation. Unlike a recurrent autoencoder, every position of
    the window is treated as an independent feature.

    The reconstruction error of a window is used as an anomaly score:
    the harder a window is to reconstruct, the more unusual its
    pattern of returns.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input window (number of returns per
        window).
    hidden_dim : int
        Number of units in the hidden layers. Defaults to `16`.
    latent_dim : int
        Dimensionality of the latent representation. Defaults to `4`.

    Attributes
    ----------
    encoder : nn.Sequential
        Module that encodes the input into the latent space. Because
        it ends in a `ReLU`, the latent representation is never
        negative.
    decoder : nn.Sequential
        Module that reconstructs the input from the latent space. It
        has no final activation, so the output can be negative, just
        like the normalized log-returns it reconstructs.

    Example
    -------
    >>> model = SimpleAutoencoder(input_dim=10, latent_dim=3)
    >>> windows = torch.randn(8, 10)
    >>> model.encode(windows).shape
    torch.Size([8, 3])
    >>> model(windows).shape
    torch.Size([8, 10])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        latent_dim: int = 4,
    ) -> None:
        """Initialize the encoder and decoder layers."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Compress the input windows into the latent space.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor of shape `(batch_size, input_dim)` with the input
            windows.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_size, latent_dim)` with the latent
            representation of each window. Every value is greater than
            or equal to zero.
        """
        return self.encoder(input_tensor)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run the forward pass through the autoencoder.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor of shape `(batch_size, input_dim)` with the input
            windows.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_size, input_dim)` with the
            reconstructed windows.
        """
        latent_representation = self.encode(input_tensor)

        return self.decoder(latent_representation)
