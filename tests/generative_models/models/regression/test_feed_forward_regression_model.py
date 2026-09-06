"""Unit tests for FeedForwardRegressionModel."""

import torch

from generative_models.models.regression.feed_forward_regression_model import (
    FeedForwardRegressionModel,
)


def test_forward_returns_expected_output_shape() -> None:
    """Check that the forward pass maps (batch, window) to (batch, 1)."""
    model = FeedForwardRegressionModel(
        input_window_size=30,
        hidden_layer_size=16,
    )
    input_window = torch.randn(8, 30)

    predicted_values = model(input_window)

    assert predicted_values.shape == torch.Size([8, 1])


def test_forward_is_deterministic_in_eval_mode() -> None:
    """Check that two forward passes with the same input match in eval mode."""
    model = FeedForwardRegressionModel(
        input_window_size=10,
        hidden_layer_size=8,
    )
    model.eval()
    input_window = torch.randn(4, 10)

    with torch.no_grad():
        first_prediction = model(input_window)
        second_prediction = model(input_window)

    assert torch.equal(first_prediction, second_prediction)
