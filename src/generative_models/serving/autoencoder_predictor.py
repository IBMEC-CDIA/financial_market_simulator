"""Inference with a trained dense autoencoder outlier detector."""

from dataclasses import dataclass

import numpy as np
import torch

from generative_models.common.normalization import normalize_series
from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder
from generative_models.models.autoencoder.trainer import (
    compute_reconstruction_scores,
)


@dataclass
class AutoencoderModelMetadata:
    """Training information required to run inference.

    Parameters
    ----------
    ticker : str
        Ticker symbol the model was trained on.
    window_size : int
        Number of log-returns in each window.
    return_train_mean : float
        Mean of the training log-returns.
    return_train_std : float
        Standard deviation of the training log-returns.
    anomaly_threshold : float
        Reconstruction score above which a window is an outlier.
    artifact_name : str
        Versioned name of the model artifact (for example,
        `"simple-autoencoder-aapl:v3"`).
    """

    ticker: str
    window_size: int
    return_train_mean: float
    return_train_std: float
    anomaly_threshold: float
    artifact_name: str


@dataclass
class AutoencoderPrediction:
    """Per-window outcome of an inference.

    Parameters
    ----------
    reconstruction_scores : np.ndarray
        Mean squared reconstruction error of each window.
    is_outlier : np.ndarray
        Boolean flags, `True` when the score exceeds the threshold.
    window_end_indices : np.ndarray
        Index of the last log-return of each window.
    """

    reconstruction_scores: np.ndarray
    is_outlier: np.ndarray
    window_end_indices: np.ndarray


class AutoencoderPredictor:
    """Score windows of new data with a trained autoencoder.

    Parameters
    ----------
    model : SimpleAutoencoder
        Trained autoencoder, already on `device`.
    metadata : AutoencoderModelMetadata
        Training statistics and anomaly threshold.
    device : str
        Device where the windows are scored. Defaults to `"cpu"`.

    Example
    -------
    >>> predictor = AutoencoderPredictor(model=model, metadata=metadata)
    >>> prediction = predictor.predict_from_prices(price_values)
    >>> prediction.is_outlier.sum()
    3
    """

    def __init__(
        self,
        model: SimpleAutoencoder,
        metadata: AutoencoderModelMetadata,
        device: str = "cpu",
    ) -> None:
        """Store the model in evaluation mode with its metadata."""
        self.model = model.eval()
        self.metadata = metadata
        self.device = device

    def predict(
        self,
        log_return_values: np.ndarray | torch.Tensor,
    ) -> AutoencoderPrediction:
        """Score every sliding window of a log-return series.

        Parameters
        ----------
        log_return_values : np.ndarray | torch.Tensor
            One-dimensional series of raw (not normalized) log-returns.
            The input is copied and never modified.

        Returns
        -------
        AutoencoderPrediction
            Scores, outlier flags and window end indices.

        Raises
        ------
        ValueError
            If the series has fewer values than `window_size`.

        Example
        -------
        >>> prediction = predictor.predict(np.random.randn(12) * 0.01)
        >>> prediction.window_end_indices
        array([ 4,  5,  6,  7,  8,  9, 10, 11])
        """
        if isinstance(log_return_values, torch.Tensor):
            log_return_tensor = (
                log_return_values.detach().clone().to(torch.float32)
            )
        else:
            log_return_tensor = torch.tensor(
                np.array(log_return_values, dtype=np.float32)
            )
        log_return_tensor = log_return_tensor.reshape(-1).cpu()

        window_size = self.metadata.window_size
        number_of_returns = len(log_return_tensor)
        if number_of_returns < window_size:
            raise ValueError(
                f"At least {window_size} log-returns are required, "
                f"got {number_of_returns}."
            )

        normalized_tensor, _, _ = normalize_series(
            log_return_tensor,
            mean=self.metadata.return_train_mean,
            std=self.metadata.return_train_std,
        )
        window_tensor = normalized_tensor.unfold(0, window_size, 1).to(
            self.device
        )
        reconstruction_scores = compute_reconstruction_scores(
            self.model, window_tensor
        )

        return AutoencoderPrediction(
            reconstruction_scores=reconstruction_scores,
            is_outlier=(
                reconstruction_scores > self.metadata.anomaly_threshold
            ),
            window_end_indices=np.arange(window_size - 1, number_of_returns),
        )

    def predict_from_prices(
        self,
        price_values: np.ndarray | torch.Tensor,
    ) -> AutoencoderPrediction:
        """Score every window of the log-returns of a price series.

        Parameters
        ----------
        price_values : np.ndarray | torch.Tensor
            One-dimensional series of positive prices. Window end index
            `i` refers to the return between prices `i` and `i + 1`.

        Returns
        -------
        AutoencoderPrediction
            Scores, outlier flags and window end indices.

        Raises
        ------
        ValueError
            If there are fewer than `window_size + 1` prices.
        """
        if isinstance(price_values, torch.Tensor):
            price_tensor = price_values.detach().clone().to(torch.float32)
        else:
            price_tensor = torch.tensor(
                np.array(price_values, dtype=np.float32)
            )
        log_price_tensor = torch.log(price_tensor.reshape(-1))
        log_return_tensor = log_price_tensor[1:] - log_price_tensor[:-1]

        return self.predict(log_return_tensor)
