"""Normalization utilities shared across generative models."""

from typing import Optional

import torch


def normalize_series(
    series_values: torch.Tensor,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> tuple[torch.Tensor, float, float]:
    """Standardize a 1D tensor using a mean and standard deviation.

    Parameters
    ----------
    series_values : torch.Tensor
        One-dimensional tensor to be standardized (z-score).
    mean : float, optional
        Mean used to standardize `series_values`. If not provided, it
        is computed from `series_values` itself.
    std : float, optional
        Standard deviation used to standardize `series_values`. If not
        provided, it is computed from `series_values` itself.

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
    >>> normalized_validation, _, _ = normalize_series(
    ...     validation_log_return_tensor, mean=mean, std=std,
    ... )
    """
    if mean is None:
        mean = series_values.mean().item()
    if std is None:
        std = series_values.std().item()
    normalized_series_values = (series_values - mean) / std

    return normalized_series_values, mean, std
