"""Fully connected autoencoder for reconstructing log-return windows."""

import torch
from torch import nn


class SimpleAutoencoder(nn.Module):
    """Dense autoencoder that reconstructs windows of a time series.

    The encoder compresses each input window into a small latent
    representation and the decoder reconstructs the original window
    from it. The reconstruction error of a window is used as its
    anomaly score: windows that the model reconstructs poorly are
    treated as outliers. The decoder has no final activation, so the
    reconstruction can be negative, as normalized log-returns are.

    Parameters
    ----------
    input_dim : int
        Number of values in each input window.
    hidden_dim : int
        Number of units in the hidden layers of the encoder and the
        decoder. Defaults to 16.
    latent_dim : int
        Size of the latent representation. Defaults to 4.

    Example
    -------
    >>> model = SimpleAutoencoder(input_dim=10)
    >>> input_windows = torch.randn(8, 10)
    >>> model(input_windows).shape
    torch.Size([8, 10])
    >>> model.encode(input_windows).shape
    torch.Size([8, 4])
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

    def encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Compute the latent representation of the input windows.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor of shape `(batch_size, input_dim)` with the windows
            to encode.

        Returns
        -------
        torch.Tensor
            Non-negative tensor of shape `(batch_size, latent_dim)`.
        """
        return self.encoder(input_tensor)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input windows through the latent space.

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
        return self.decoder(self.encode(input_tensor))
