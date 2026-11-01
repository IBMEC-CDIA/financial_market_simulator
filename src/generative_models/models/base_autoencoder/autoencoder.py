"""Fully connected autoencoder for reconstruction-based anomaly scoring."""

import torch
from torch import nn


class SimpleAutoencoder(nn.Module):
    """Dense autoencoder that reconstructs windows of normalized returns.

    The reconstruction error of a window is used as an anomaly score:
    windows that the model cannot reproduce well are unusual with
    respect to the training data.

    Parameters
    ----------
    input_dim : int
        Number of time steps in each input window. The decoder outputs
        the same number of values.
    hidden_dim : int
        Number of units in the hidden layers of the encoder and decoder.
    latent_dim : int
        Size of the latent representation produced by the encoder.

    Attributes
    ----------
    encoder : nn.Sequential
        Maps an input window to its latent representation. It ends with
        a ReLU, so the latent values are never negative.
    decoder : nn.Sequential
        Maps a latent representation back to a window. It has no final
        activation, so the reconstructed values may be negative.

    Example
    -------
    >>> model = SimpleAutoencoder(input_dim=10)
    >>> window_batch = torch.randn(8, 10)
    >>> model(window_batch).shape
    torch.Size([8, 10])
    >>> model.encode(window_batch).shape
    torch.Size([8, 4])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        latent_dim: int = 4,
    ) -> None:
        """Build the encoder and decoder layers."""
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
        """Return the latent representation of a batch of windows.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor of shape `(batch_size, input_dim)` with the windows
            to encode.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_size, latent_dim)` with non-negative
            values.
        """
        return self.encoder(input_tensor)

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

        return self.decoder(latent_tensor)
