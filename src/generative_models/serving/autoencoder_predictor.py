"""Outlier prediction with a trained simple autoencoder."""

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
    """Information required to run a trained autoencoder on new data.

    Parameters
    ----------
    ticker : str
        Ticker symbol the model was trained on.
    window_size : int
        Number of consecutive log-returns in each input window.
    return_train_mean : float
        Mean of the training log-returns used for normalization.
    return_train_std : float
        Standard deviation of the training log-returns used for
        normalization.
    anomaly_threshold : float
        Reconstruction error above which a window is classified as an
        outlier.
    artifact_name : str
        Name and version of the source artifact (for example,
        `"simple-autoencoder-aapl:v3"`), kept for traceability.
    """

    ticker: str
    window_size: int
    return_train_mean: float
    return_train_std: float
    anomaly_threshold: float
    artifact_name: str


@dataclass
class AutoencoderPrediction:
    """Outlier detection output for a log-return series.

    Parameters
    ----------
    reconstruction_scores : np.ndarray
        Mean squared reconstruction error of each window.
    is_outlier : np.ndarray
        Boolean array flagging the windows whose score is above the
        anomaly threshold.
    window_end_indices : np.ndarray
        Index, in the log-return series, of the last element of each
        window. Use it to align each score with its date.
    """

    reconstruction_scores: np.ndarray
    is_outlier: np.ndarray
    window_end_indices: np.ndarray


class AutoencoderPredictor:
    """Score log-return windows and flag outliers with a trained model.

    New data is normalized with the training statistics stored in the
    model metadata, split into sliding windows and reconstructed by
    the autoencoder. Windows whose reconstruction error exceeds the
    stored anomaly threshold are classified as outliers.

    Parameters
    ----------
    model : SimpleAutoencoder
        Trained autoencoder, already on `device`.
    metadata : AutoencoderModelMetadata
        Normalization statistics, window size and anomaly threshold
        that belong to `model`.
    device : str
        Torch device where `model` lives. Defaults to `"cpu"`.

    Example
    -------
    >>> predictor = registry.load_latest_model()
    >>> prediction = predictor.predict_from_prices(price_values)
    >>> int(prediction.is_outlier.sum())
    12
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
        """Detect outlier windows in a log-return series.

        Parameters
        ----------
        log_return_values : np.ndarray | torch.Tensor
            One-dimensional series of raw (not normalized) daily
            log-returns, with at least `window_size` elements.

        Returns
        -------
        AutoencoderPrediction
            Reconstruction score, outlier flag and end index of each
            window.

        Raises
        ------
        ValueError
            If the series is shorter than the model window size.
        """
        log_return_tensor = _to_float_tensor(log_return_values)
        window_size = self.metadata.window_size
        if len(log_return_tensor) < window_size:
            raise ValueError(
                f"At least {window_size} log-returns are required, got "
                f"{len(log_return_tensor)}."
            )

        normalized_tensor, _, _ = normalize_series(
            log_return_tensor,
            mean=self.metadata.return_train_mean,
            std=self.metadata.return_train_std,
        )
        windows_tensor = normalized_tensor.unfold(0, window_size, 1)
        reconstruction_scores = compute_reconstruction_scores(
            self.model,
            windows_tensor.to(self.device),
        )

        return AutoencoderPrediction(
            reconstruction_scores=reconstruction_scores,
            is_outlier=(
                reconstruction_scores > self.metadata.anomaly_threshold
            ),
            window_end_indices=np.arange(
                window_size - 1,
                len(log_return_tensor),
            ),
        )

    def predict_from_prices(
        self,
        price_values: np.ndarray | torch.Tensor,
    ) -> AutoencoderPrediction:
        """Detect outlier windows directly from a price series.

        Parameters
        ----------
        price_values : np.ndarray | torch.Tensor
            One-dimensional series of positive closing prices, with at
            least `window_size + 1` elements.

        Returns
        -------
        AutoencoderPrediction
            Prediction for the log-returns of `price_values`. Window
            end index `i` refers to the return between prices `i` and
            `i + 1`.
        """
        price_tensor = _to_float_tensor(price_values)
        log_return_tensor = torch.log(price_tensor[1:]) - torch.log(
            price_tensor[:-1]
        )
        return self.predict(log_return_tensor)


def _to_float_tensor(
    series_values: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Copy a series into a one-dimensional float32 tensor.

    Parameters
    ----------
    series_values : np.ndarray | torch.Tensor
        Series to convert.

    Returns
    -------
    torch.Tensor
        Flattened float32 copy of `series_values`, detached from the
        source memory so read-only arrays are safe to use.
    """
    if isinstance(series_values, torch.Tensor):
        return series_values.detach().to(torch.float32).flatten().clone()
    return torch.tensor(
        np.asarray(series_values),
        dtype=torch.float32,
    ).flatten()
