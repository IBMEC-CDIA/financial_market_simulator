"""Inference of a trained SimpleAutoencoder on new log-return series."""

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
    """Training information needed to run inference with a model.

    Parameters
    ----------
    ticker : str
        Ticker symbol the model was trained on.
    window_size : int
        Number of consecutive log-returns in each window.
    return_train_mean : float
        Mean of the training log-returns, used for normalization.
    return_train_std : float
        Standard deviation of the training log-returns, used for
        normalization.
    anomaly_threshold : float
        Reconstruction score above which a window is an outlier.
    artifact_name : str
        Versioned name of the artifact the model was read from, for
        example `"simple-autoencoder-aapl:v3"`.
    """

    ticker: str
    window_size: int
    return_train_mean: float
    return_train_std: float
    anomaly_threshold: float
    artifact_name: str


@dataclass
class AutoencoderPrediction:
    """Reconstruction scores and outlier flags of every window.

    Parameters
    ----------
    reconstruction_scores : np.ndarray
        One-dimensional array with the mean squared reconstruction
        error of each window.
    is_outlier : np.ndarray
        Boolean array, `True` when the score of the window is strictly
        greater than the anomaly threshold.
    window_end_indices : np.ndarray
        Index of the last log-return of each window.
    """

    reconstruction_scores: np.ndarray
    is_outlier: np.ndarray
    window_end_indices: np.ndarray


class AutoencoderPredictor:
    """Score log-return windows with a trained autoencoder.

    The predictor normalizes the log-returns with the training
    statistics, builds every sliding window of the series, computes the
    reconstruction error of each window and flags as outliers the
    windows above the anomaly threshold saved during training.

    Parameters
    ----------
    model : SimpleAutoencoder
        Trained autoencoder, already on `device`. It is put in
        evaluation mode.
    metadata : AutoencoderModelMetadata
        Training information of the model.
    device : str
        Torch device where the windows are scored. Defaults to
        `"cpu"`.

    Example
    -------
    >>> predictor = AutoencoderPredictor(
    ...     model=trained_model,
    ...     metadata=model_metadata,
    ... )
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
            One-dimensional series of log-returns. It is not modified.

        Returns
        -------
        AutoencoderPrediction
            Score, outlier flag and end index of each of the
            `n - window_size + 1` windows of the `n` log-returns.

        Raises
        ------
        ValueError
            If the series has fewer values than `window_size`.
        """
        if isinstance(log_return_values, torch.Tensor):
            log_return_tensor = log_return_values.detach().clone()
        else:
            log_return_tensor = torch.tensor(np.array(log_return_values))
        log_return_tensor = log_return_tensor.to(torch.float32).flatten()

        window_size = self.metadata.window_size
        number_of_log_returns = len(log_return_tensor)
        if number_of_log_returns < window_size:
            raise ValueError(
                f"At least {window_size} log-returns are required, "
                f"got {number_of_log_returns}."
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
            self.model,
            window_tensor,
        )

        return AutoencoderPrediction(
            reconstruction_scores=reconstruction_scores,
            is_outlier=reconstruction_scores
            > self.metadata.anomaly_threshold,
            window_end_indices=np.arange(
                window_size - 1,
                number_of_log_returns,
            ),
        )

    def predict_from_prices(
        self,
        price_values: np.ndarray | torch.Tensor,
    ) -> AutoencoderPrediction:
        """Score every window of the log-returns of a price series.

        Parameters
        ----------
        price_values : np.ndarray | torch.Tensor
            One-dimensional series of positive prices. With `m` prices
            there are `m - 1` log-returns, and the window end index
            `i` refers to the return between prices `i` and `i + 1`.

        Returns
        -------
        AutoencoderPrediction
            Prediction of `predict` over the log-returns.

        Raises
        ------
        ValueError
            If there are fewer than `window_size + 1` prices.
        """
        if isinstance(price_values, torch.Tensor):
            price_tensor = price_values.detach().clone()
        else:
            price_tensor = torch.tensor(np.array(price_values))
        log_price_tensor = torch.log(price_tensor.to(torch.float32).flatten())

        return self.predict(log_price_tensor[1:] - log_price_tensor[:-1])
