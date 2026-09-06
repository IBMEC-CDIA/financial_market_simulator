"""Dataset utilities for reconstruction-based generative models."""

import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowReconstructionDataset(Dataset):
    """Dataset that generates sliding windows of a time series for
    reconstruction-based models such as autoencoders and VAEs.

    Each sample is a single window extracted from the underlying
    series. There is no separate output window: the window itself is
    both the input given to the encoder and the target that the
    decoder must reconstruct.

    Parameters
    ----------
    series_values : np.ndarray or torch.Tensor
        One-dimensional array or tensor with the time series values
        (for example, normalized log-returns). Accepted types are
        `numpy.ndarray` and `torch.Tensor`; any other type raises a
        `TypeError`.
    window_size : int
        Number of consecutive time steps contained in each sample.

    Raises
    ------
    TypeError
        If `series_values` is neither a `numpy.ndarray` nor a
        `torch.Tensor`.

    Example
    -------
    >>> dataset = SlidingWindowReconstructionDataset(
    ...     series_values=normalized_train_return_tensor.numpy(),
    ...     window_size=30,
    ... )
    >>> window = dataset[0]
    >>> window.shape
    torch.Size([30])
    """

    def __init__(
        self,
        series_values: np.ndarray | torch.Tensor,
        window_size: int,
    ) -> None:
        """Validate the series type and store it as a torch.Tensor."""
        if isinstance(series_values, torch.Tensor):
            self.series_values = series_values
        elif isinstance(series_values, np.ndarray):
            self.series_values = torch.from_numpy(series_values)
        else:
            raise TypeError(
                "series_values must be a numpy.ndarray or a "
                f"torch.Tensor, got {type(series_values).__name__}."
            )

        self.window_size = window_size

    def __len__(self) -> int:
        """Return how many windows fit in the stored series."""
        return len(self.series_values) - self.window_size + 1

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the window starting at the given index as a tensor."""
        window_start_index = index
        window_end_index = window_start_index + self.window_size

        window_tensor = self.series_values[
            window_start_index:window_end_index
        ]

        return window_tensor
