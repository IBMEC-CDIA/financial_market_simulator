"""Unit tests for SlidingWindowReconstructionDataset."""

import numpy as np
import pytest
import torch

from generative_models.data.datasets import SlidingWindowReconstructionDataset


def test_len_returns_number_of_valid_windows() -> None:
    """Check that the dataset length equals series_length - window_size + 1."""
    series_values = np.arange(10, dtype=np.float32)
    dataset = SlidingWindowReconstructionDataset(
        series_values=series_values,
        window_size=3,
    )

    assert len(dataset) == 8


def test_getitem_returns_expected_window_from_numpy_array() -> None:
    """Check that indexing returns the correct contiguous window."""
    series_values = np.arange(10, dtype=np.float32)
    dataset = SlidingWindowReconstructionDataset(
        series_values=series_values,
        window_size=3,
    )

    window = dataset[2]

    assert torch.equal(window, torch.tensor([2.0, 3.0, 4.0]))


def test_getitem_returns_expected_window_from_tensor() -> None:
    """Check that a torch.Tensor input is stored and sliced correctly."""
    series_values = torch.arange(10, dtype=torch.float32)
    dataset = SlidingWindowReconstructionDataset(
        series_values=series_values,
        window_size=4,
    )

    window = dataset[0]

    assert torch.equal(window, torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_window_shape_matches_window_size() -> None:
    """Check that every returned window has the requested window_size."""
    series_values = np.arange(30, dtype=np.float32)
    window_size = 30
    dataset = SlidingWindowReconstructionDataset(
        series_values=series_values,
        window_size=window_size,
    )

    window = dataset[0]

    assert window.shape == torch.Size([window_size])


def test_invalid_series_values_type_raises_type_error() -> None:
    """Check that a non-array, non-tensor input raises TypeError."""
    with pytest.raises(TypeError):
        SlidingWindowReconstructionDataset(
            series_values=[0.0, 1.0, 2.0],
            window_size=2,
        )
