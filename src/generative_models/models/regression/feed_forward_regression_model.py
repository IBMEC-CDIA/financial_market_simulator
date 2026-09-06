"""Feed-forward regression model for next-value time series prediction."""

import torch
from torch import nn


class FeedForwardRegressionModel(nn.Module):
    """Feed-forward network that predicts the next value in a series.

    The network is composed of two hidden layers with `ReLU`
    activations, followed by a linear output layer that produces a
    single predicted value. The hidden layers are what differentiate
    this model from a plain linear regression: they let the model
    learn non-linear combinations of the values in the input window.

    Parameters
    ----------
    input_window_size : int
        Number of consecutive past values given as input to the
        network.
    hidden_layer_size : int
        Number of units in each of the two hidden layers.

    Example
    -------
    >>> model = FeedForwardRegressionModel(
    ...     input_window_size=30,
    ...     hidden_layer_size=64,
    ... )
    >>> input_window = torch.randn(8, 30)
    >>> model(input_window).shape
    torch.Size([8, 1])
    """

    def __init__(
        self,
        input_window_size: int,
        hidden_layer_size: int,
    ) -> None:
        """Build the sequential stack of linear layers and activations."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_window_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
        )

    def forward(self, input_window: torch.Tensor) -> torch.Tensor:
        """Compute the predicted next value from the input window.

        Parameters
        ----------
        input_window : torch.Tensor
            Tensor of shape `(batch_size, input_window_size)` with the
            past values used as input.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(batch_size, 1)` with the predicted next
            value for each sample in the batch.
        """
        return self.network(input_window)
