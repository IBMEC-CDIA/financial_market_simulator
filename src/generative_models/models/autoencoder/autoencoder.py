"""Dense autoencoder used for reconstruction-based anomaly detection."""

import torch
from torch import nn


class SimpleAutoencoder(nn.Module):
    """Fully connected autoencoder for windows of normalized log-returns.

    The encoder compresses each input window into a non-negative latent
    representation, and the decoder reconstructs the original window
    from it. The reconstruction error is used as an anomaly score.

    Parameters
    ----------
    input_dim : int
        Number of values in each input window (the window size).
    hidden_dim : int
        Number of units in the hidden layer of the encoder and the
        decoder. Defaults to 16.
    latent_dim : int
        Number of units in the latent representation. Defaults to 4.

    Example
    -------
    >>> model = SimpleAutoencoder(input_dim=10)
    >>> model(torch.zeros(8, 10)).shape
    torch.Size([8, 10])
    >>> model.encode(torch.zeros(8, 10)).shape
    torch.Size([8, 4])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        latent_dim: int = 4,
    ) -> None:
        """Build the encoder and decoder blocks."""
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
        """Map input windows to their latent representation.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor of shape `(batch_size, input_dim)`.

        Returns
        -------
        torch.Tensor
            Non-negative tensor of shape `(batch_size, latent_dim)`.
        """
        return self.encoder(input_tensor)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input windows.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor of shape `(batch_size, input_dim)`.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of shape `(batch_size, input_dim)`.
        """
        return self.decoder(self.encode(input_tensor))
