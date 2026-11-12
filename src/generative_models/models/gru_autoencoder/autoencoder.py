"""Recurrent GRU autoencoder for reconstruction-based anomaly scoring."""

import torch
from torch import nn


class GRUAutoencoder(nn.Module):
    """Sequence autoencoder that reconstructs windows with GRU layers.

    Each window of normalized log-returns is read as a sequence with one
    feature per time step. The encoder GRU summarizes the sequence in its
    last hidden state, which is projected to the latent representation.
    The decoder repeats the latent vector over every time step, runs a
    second GRU and projects each hidden state back to one value. The
    reconstruction error of a window is used as an anomaly score.

    Parameters
    ----------
    input_dim : int
        Number of time steps in each input window. The decoder outputs
        the same number of values.
    hidden_dim : int
        Size of the hidden state of the encoder and decoder GRUs.
    latent_dim : int
        Size of the latent representation produced by the encoder.

    Attributes
    ----------
    input_dim : int
        Number of time steps in each window.
    encoder_gru : nn.GRU
        Reads the window, one log-return per time step.
    encoder_projection : nn.Linear
        Maps the last hidden state of the encoder to the latent vector.
    decoder_projection : nn.Linear
        Maps the latent vector to the input of every decoder time step.
    decoder_gru : nn.GRU
        Produces one hidden state per time step of the reconstruction.
    output_projection : nn.Linear
        Maps each decoder hidden state to one reconstructed value. It has
        no activation, so the reconstructed values may be negative.

    Example
    -------
    >>> model = GRUAutoencoder(input_dim=10)
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

        self.input_dim = input_dim
        self.encoder_gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.encoder_projection = nn.Linear(hidden_dim, latent_dim)
        self.decoder_projection = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output_projection = nn.Linear(hidden_dim, 1)

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
            Tensor of shape `(batch_size, latent_dim)`.
        """
        sequence_tensor = input_tensor.unsqueeze(-1)
        _, last_hidden_state = self.encoder_gru(sequence_tensor)

        return self.encoder_projection(last_hidden_state[-1])

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
        decoder_input_tensor = (
            self.decoder_projection(latent_tensor)
            .unsqueeze(1)
            .repeat(1, self.input_dim, 1)
        )
        decoder_output_tensor, _ = self.decoder_gru(decoder_input_tensor)

        return self.output_projection(decoder_output_tensor).squeeze(-1)
