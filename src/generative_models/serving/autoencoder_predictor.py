"""Inference wrapper that scores new data with a trained autoencoder."""

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
    """Information saved with a trained autoencoder and needed for inference.

    Parameters
    ----------
    ticker : str
        Ticker symbol the model was trained on.
    window_size : int
        Number of log-returns in each window.
    return_train_mean : float
        Mean of the training log-returns, used for normalization.
    return_train_std : float
        Standard deviation of the training log-returns, used for
        normalization.
    anomaly_threshold : float
        Reconstruction score above which a window is an outlier.
    artifact_name : str
        Versioned name of the wandb artifact, for example
        `"simple-autoencoder-petr4.sa:v2"`.
    """

    ticker: str
    window_size: int
    return_train_mean: float
    return_train_std: float
    anomaly_threshold: float
    artifact_name: str


@dataclass
class AutoencoderPrediction:
    """Scores and outlier flags of every window of a series.

    Parameters
    ----------
    reconstruction_scores : np.ndarray
        One-dimensional array with the reconstruction score of each
        window.
    is_outlier : np.ndarray
        Boolean array, `True` where the score is strictly greater than
        the anomaly threshold.
    window_end_indices : np.ndarray
        Index of the last log-return of each window, in the input
        series.
    """

    reconstruction_scores: np.ndarray
    is_outlier: np.ndarray
    window_end_indices: np.ndarray


class AutoencoderPredictor:
    """Score the windows of a series with a trained autoencoder.

    The predictor normalizes the input with the statistics saved at
    training time, builds every sliding window and compares each
    reconstruction score with the saved anomaly threshold. The threshold
    is never recomputed here.

    Parameters
    ----------
    model : SimpleAutoencoder
        Trained autoencoder, already on `device`. It is put in
        evaluation mode.
    metadata : AutoencoderModelMetadata
        Training metadata of the model.
    device : str
        Device where the windows are scored.

    Attributes
    ----------
    model : SimpleAutoencoder
        Autoencoder in evaluation mode.
    metadata : AutoencoderModelMetadata
        Training metadata of the model.
    device : str
        Device where the windows are scored.

    Example
    -------
    >>> predictor = AutoencoderPredictor(model=model, metadata=metadata)
    >>> prediction = predictor.predict(log_return_values)
    >>> prediction.is_outlier.sum()
    3
    """

    def __init__(
        self,
        model: SimpleAutoencoder,
        metadata: AutoencoderModelMetadata,
        device: str = "cpu",
    ) -> None:
        """Store the model and metadata and put the model in eval mode."""
        self.model = model
        self.metadata = metadata
        self.device = device
        self.model.eval()

    def predict(
        self,
        log_return_values: np.ndarray | torch.Tensor,
    ) -> AutoencoderPrediction:
        """Score every sliding window of a series of log-returns.

        Parameters
        ----------
        log_return_values : np.ndarray | torch.Tensor
            One-dimensional series of log-returns. The input is copied
            and never modified.

        Returns
        -------
        AutoencoderPrediction
            Scores, outlier flags and window end indices. There are
            `n - window_size + 1` windows for `n` log-returns.

        Raises
        ------
        ValueError
            If the series is not one-dimensional or has fewer values
            than `window_size`.

        Example
        -------
        >>> prediction = predictor.predict(np.zeros(12))
        >>> prediction.window_end_indices
        array([ 4,  5,  6,  7,  8,  9, 10, 11])
        """
        log_return_tensor = _to_float_tensor(log_return_values)
        window_size = self.metadata.window_size
        number_of_values = len(log_return_tensor)
        if number_of_values < window_size:
            raise ValueError(
                f"At least {window_size} log-returns are required, "
                f"got {number_of_values}."
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
            is_outlier=reconstruction_scores > self.metadata.anomaly_threshold,
            window_end_indices=np.arange(window_size - 1, number_of_values),
        )

    def predict_from_prices(
        self,
        price_values: np.ndarray | torch.Tensor,
    ) -> AutoencoderPrediction:
        """Score the windows of a series of closing prices.

        The log-returns are computed as `log(p[t + 1]) - log(p[t])`, so
        `m` prices give `m - 1` log-returns. The window end index `i`
        refers to the return between the prices `i` and `i + 1`.

        Parameters
        ----------
        price_values : np.ndarray | torch.Tensor
            One-dimensional series of positive closing prices.

        Returns
        -------
        AutoencoderPrediction
            Scores, outlier flags and window end indices, as in
            `predict`.

        Raises
        ------
        ValueError
            If there are fewer than `window_size + 1` prices.
        """
        price_tensor = _to_float_tensor(price_values)
        window_size = self.metadata.window_size
        number_of_prices = len(price_tensor)
        if number_of_prices < window_size + 1:
            raise ValueError(
                f"At least {window_size + 1} prices are required, got "
                f"{number_of_prices}. The model window size is "
                f"{window_size}."
            )
        log_price_tensor = torch.log(price_tensor)
        log_return_tensor = log_price_tensor[1:] - log_price_tensor[:-1]

        return self.predict(log_return_tensor)


def _to_float_tensor(values: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Return a detached float32 copy of a one-dimensional series.

    Parameters
    ----------
    values : np.ndarray | torch.Tensor
        One-dimensional series to convert. Read-only arrays and tensors
        that require gradients are accepted.

    Returns
    -------
    torch.Tensor
        New float32 tensor that does not share memory with `values`.

    Raises
    ------
    ValueError
        If `values` is not one-dimensional.
    """
    if isinstance(values, torch.Tensor):
        tensor = values.detach().clone().to(torch.float32)
    else:
        tensor = torch.tensor(np.asarray(values), dtype=torch.float32)

    if tensor.ndim != 1:
        raise ValueError("The input series must be one-dimensional.")

    return tensor
