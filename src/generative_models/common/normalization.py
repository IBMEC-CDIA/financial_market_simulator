"""Normalization utilities shared across generative models."""

import torch


def normalize_series(
    series_values: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Standardize a 1D tensor using its own mean and standard deviation.

    Parameters
    ----------
    series_values : torch.Tensor
        One-dimensional tensor to be standardized (z-score).

    Returns
    -------
    tuple[torch.Tensor, float, float]
        A tuple with, in order: the standardized tensor, the mean used
        and the standard deviation used. The mean and standard
        deviation are returned so that another split of the data (for
        example, a validation set) can be standardized with the same
        statistics, avoiding data leakage.

    Example
    -------
    >>> normalized, mean, std = normalize_series(train_log_return_tensor)
    >>> round(normalized.mean().item(), 4)
    0.0
    """
    series_mean = series_values.mean().item()
    series_std = series_values.std().item()
    normalized_series_values = (series_values - series_mean) / series_std

    return normalized_series_values, series_mean, series_std
