"""Simple fully connected autoencoder for time series windows."""

import torch
from torch import nn


class SimpleAutoencoder(nn.Module):
    """A simple fully connected (dense) autoencoder.

    The network is composed of an encoder that compresses the input
    window into a lower-dimensional latent representation, and a
    decoder that reconstructs the original window from that
    representation. Unlike a recurrent autoencoder, every position of
    the window is treated as an independent feature.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input window (number of values per
        window).
    hidden_dim : int
        Number of units in the hidden layers of the encoder and the
        decoder. Defaults to `16`.
    latent_dim : int
        Dimensionality of the latent representation. Defaults to `4`.

    Example
    -------
    >>> model = SimpleAutoencoder(
    ...     input_dim=10,
    ...     hidden_dim=16,
    ...     latent_dim=4,
    ... )
    >>> input_windows = torch.randn(8, 10)
    >>> model(input_windows).shape
    torch.Size([8, 10])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        latent_dim: int = 4,
    ) -> None:
        """Build the encoder and decoder layer stacks."""
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
        """Encode the input windows into the latent space.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor of shape `(batch_size, input_dim)` with the input
            windows.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_size, latent_dim)` with the latent
            representation of each window.
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
