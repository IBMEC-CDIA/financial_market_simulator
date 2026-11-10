"""Recurrent (LSTM) autoencoder for reconstruction-based anomaly scoring."""

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """LSTM autoencoder that reconstructs windows of normalized returns.

    Each window is read as a sequence with one feature per time step. The
    encoder LSTM summarizes the sequence in its last hidden state, which
    a linear layer projects to the latent representation. The decoder
    repeats the latent representation at every time step, runs a second
    LSTM over it and projects each output back to one value.

    The reconstruction error of a window is used as an anomaly score:
    windows that the model cannot reproduce well are unusual with
    respect to the training data.

    Parameters
    ----------
    window_size : int
        Number of time steps in each input window. The decoder outputs
        the same number of values.
    hidden_dim : int
        Number of units in the hidden state of the encoder and decoder
        LSTM layers.
    latent_dim : int
        Size of the latent representation produced by the encoder.

    Attributes
    ----------
    window_size : int
        Number of time steps the decoder generates.
    encoder_lstm : nn.LSTM
        Reads the window, one value per time step.
    encoder_projection : nn.Linear
        Maps the last hidden state of the encoder to the latent
        representation. It has no activation, so the latent values may
        be negative.
    decoder_lstm : nn.LSTM
        Reads the latent representation repeated at every time step.
    decoder_projection : nn.Linear
        Maps each decoder output to one reconstructed value. It has no
        activation, so the reconstructed values may be negative.

    Example
    -------
    >>> model = LSTMAutoencoder(window_size=10)
    >>> window_batch = torch.randn(8, 10)
    >>> model(window_batch).shape
    torch.Size([8, 10])
    >>> model.encode(window_batch).shape
    torch.Size([8, 4])
    """

    def __init__(
        self,
        window_size: int,
        hidden_dim: int = 16,
        latent_dim: int = 4,
    ) -> None:
        """Build the encoder and decoder layers."""
        super().__init__()

        self.window_size = window_size
        self.encoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.encoder_projection = nn.Linear(hidden_dim, latent_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.decoder_projection = nn.Linear(hidden_dim, 1)

    def encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return the latent representation of a batch of windows.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor of shape `(batch_size, window_size)` with the windows
            to encode.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_size, latent_dim)`.
        """
        sequence_tensor = input_tensor.unsqueeze(-1)
        _, (hidden_state, _) = self.encoder_lstm(sequence_tensor)

        return self.encoder_projection(hidden_state[-1])

    def decode(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of windows from latent representations.

        Parameters
        ----------
        latent_tensor : torch.Tensor
            Tensor of shape `(batch_size, latent_dim)`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_size, window_size)` with the
            reconstructed windows.
        """
        repeated_latent_tensor = latent_tensor.unsqueeze(1).repeat(
            1,
            self.window_size,
            1,
        )
        decoded_sequence_tensor, _ = self.decoder_lstm(repeated_latent_tensor)

        return self.decoder_projection(decoded_sequence_tensor).squeeze(-1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of windows.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor of shape `(batch_size, window_size)` with the windows
            to reconstruct.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_size, window_size)` with the
            reconstructed windows.
        """
        latent_tensor = self.encode(input_tensor)

        return self.decode(latent_tensor)
