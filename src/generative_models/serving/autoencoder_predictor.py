"""Inference wrapper around a trained log-return autoencoder."""

from dataclasses import dataclass

import numpy as np
import torch

from generative_models.common.normalization import normalize_series
from generative_models.models.autoencoder.autoencoder import (
    SimpleAutoencoder,
)
from generative_models.models.autoencoder.trainer import (
    compute_reconstruction_scores,
)


@dataclass
class AutoencoderModelMetadata:
    """Everything inference needs to know about a trained model.

    Parameters
    ----------
    ticker : str
        Ticker symbol the model was trained on.
    window_size : int
        Number of consecutive log-returns per window, which is also
        the input dimension of the model.
    return_train_mean : float
        Mean of the training log-returns, used to normalize new data
        with the same statistics seen during training.
    return_train_std : float
        Standard deviation of the training log-returns.
    anomaly_threshold : float
        Reconstruction error above which a window is an outlier.
    artifact_name : str
        Versioned name of the artifact the model came from.
    """

    ticker: str
    window_size: int
    return_train_mean: float
    return_train_std: float
    anomaly_threshold: float
    artifact_name: str


@dataclass
class AutoencoderPrediction:
    """Outcome of scoring a log-return series window by window.

    Parameters
    ----------
    reconstruction_scores : np.ndarray
        One score per window, in chronological order.
    is_outlier : np.ndarray
        Boolean array, `True` where the score is strictly above the
        anomaly threshold.
    window_end_indices : np.ndarray
        Index, in the log-return series, of the last value of each
        window.
    """

    reconstruction_scores: np.ndarray
    is_outlier: np.ndarray
    window_end_indices: np.ndarray


class AutoencoderPredictor:
    """Apply a trained autoencoder to new log-returns.

    The predictor normalizes the incoming series with the statistics
    stored in the metadata, builds every sliding window, scores them
    by reconstruction error and flags the windows above the anomaly
    threshold. Unlike the training notebook, the threshold is never
    recomputed: it comes from the training metadata, so that outliers
    are judged against what the model learned.

    This class does not know where the model came from. It receives
    the model and its metadata already built.

    Parameters
    ----------
    model : SimpleAutoencoder
        Trained autoencoder, already placed on `device`.
    metadata : AutoencoderModelMetadata
        Window size, normalization statistics and anomaly threshold of
        the trained model.
    device : str
        Device where the windows are scored. Defaults to `"cpu"`.

    Attributes
    ----------
    model : SimpleAutoencoder
        Trained autoencoder, kept in evaluation mode.
    metadata : AutoencoderModelMetadata
        Metadata of the trained model.
    device : str
        Device where the windows are scored.

    Example
    -------
    >>> predictor = AutoencoderPredictor(model, metadata)
    >>> prediction = predictor.predict_from_prices(price_values)
    >>> int(prediction.is_outlier.sum())
    3
    """

    def __init__(
        self,
        model: SimpleAutoencoder,
        metadata: AutoencoderModelMetadata,
        device: str = "cpu",
    ) -> None:
        """Store the model in evaluation mode with its metadata."""
        self.model = model
        self.model.eval()
        self.metadata = metadata
        self.device = device

    def predict(
        self,
        log_return_values: np.ndarray | torch.Tensor,
    ) -> AutoencoderPrediction:
        """Score every sliding window of a log-return series.

        Parameters
        ----------
        log_return_values : np.ndarray or torch.Tensor
            One-dimensional series of log-returns. The input is copied
            before use, so read-only arrays and tensors that require
            gradients are accepted and never modified.

        Returns
        -------
        AutoencoderPrediction
            The reconstruction score of each window, the outlier flags
            and the index of the last log-return of each window.

        Raises
        ------
        ValueError
            If the series holds fewer values than the window size, so
            that no window can be built.

        Example
        -------
        >>> prediction = predictor.predict(np.zeros(12))
        >>> prediction.window_end_indices[0]
        4
        """
        series_tensor = _to_float32_tensor(log_return_values)
        window_size = self.metadata.window_size
        number_of_returns = len(series_tensor)

        if number_of_returns < window_size:
            raise ValueError(
                f"At least {window_size} log-returns are required, "
                f"got {number_of_returns}."
            )

        normalized_tensor, _, _ = normalize_series(
            series_tensor,
            mean=self.metadata.return_train_mean,
            std=self.metadata.return_train_std,
        )
        window_tensor = normalized_tensor.unfold(
            dimension=0,
            size=window_size,
            step=1,
        ).to(self.device)

        reconstruction_scores = compute_reconstruction_scores(
            self.model,
            window_tensor,
        )

        return AutoencoderPrediction(
            reconstruction_scores=reconstruction_scores,
            is_outlier=(
                reconstruction_scores > self.metadata.anomaly_threshold
            ),
            window_end_indices=np.arange(
                window_size - 1,
                number_of_returns,
            ),
        )

    def predict_from_prices(
        self,
        price_values: np.ndarray | torch.Tensor,
    ) -> AutoencoderPrediction:
        """Score every sliding window of a price series.

        Parameters
        ----------
        price_values : np.ndarray or torch.Tensor
            One-dimensional series of consecutive prices. With `m`
            prices there are `m - 1` log-returns, so a window ending
            at index `i` refers to the return between the prices `i`
            and `i + 1`.

        Returns
        -------
        AutoencoderPrediction
            The same structure returned by `predict`, computed over the
            log-returns of `price_values`.

        Raises
        ------
        ValueError
            If there are fewer than `window_size + 1` prices.

        Example
        -------
        >>> prediction = predictor.predict_from_prices(price_values)
        >>> prediction.reconstruction_scores.shape
        (7,)
        """
        price_tensor = _to_float32_tensor(price_values)
        log_return_tensor = torch.log(price_tensor[1:]) - torch.log(
            price_tensor[:-1]
        )

        return self.predict(log_return_tensor)


def _to_float32_tensor(
    series_values: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Copy a series into a detached one-dimensional float32 tensor.

    Parameters
    ----------
    series_values : np.ndarray or torch.Tensor
        Series to convert. Read-only arrays and tensors that require
        gradients are accepted, and neither is modified.

    Returns
    -------
    torch.Tensor
        One-dimensional `float32` tensor that shares no memory with
        `series_values`.
    """
    if isinstance(series_values, torch.Tensor):
        return series_values.detach().clone().to(torch.float32)

    return torch.tensor(np.asarray(series_values), dtype=torch.float32)
