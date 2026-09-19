"""Simple fully connected autoencoder for time series windows."""

import torch
from torch import nn


class SimpleAutoencoder(nn.Module):
    """Fully connected (dense) autoencoder for sliding windows.

    The encoder compresses each input window into a lower-dimensional
    latent representation and the decoder reconstructs the original
    window from it. Unlike a recurrent autoencoder, every position of
    the window is treated as an independent feature.

    Parameters
    ----------
    input_dim : int
        Number of values in each input window.
    hidden_dim : int
        Number of units in the hidden layers.
    latent_dim : int
        Dimensionality of the latent representation.

    Example
    -------
    >>> model = SimpleAutoencoder(input_dim=10)
    >>> model(torch.randn(8, 10)).shape
    torch.Size([8, 10])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        latent_dim: int = 4,
    ) -> None:
        """Build the encoder and decoder stacks."""
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

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Encode and decode a batch of windows.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor with shape `(batch_size, input_dim)`.

        Returns
        -------
        torch.Tensor
            Reconstructed windows with shape `(batch_size, input_dim)`.
        """
        latent_representation = self.encoder(input_tensor)
        return self.decoder(latent_representation)
