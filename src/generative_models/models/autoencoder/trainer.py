"""Training pipeline of the simple autoencoder for one ticker."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from generative_models.common.normalization import normalize_series
from generative_models.data.datasets import SlidingWindowReconstructionDataset
from generative_models.data.market_data import fetch_price_series
from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder
from generative_models.models.autoencoder.config import (
    AutoencoderTrainingConfig,
)
from generative_models.tracking.wandb_tracker import WandbExperimentTracker

MODEL_NAME_PREFIX = "simple-autoencoder-"


@dataclass
class PreparedAutoencoderData:
    """Datasets and normalization statistics ready for training.

    Parameters
    ----------
    train_dataset : SlidingWindowReconstructionDataset
        Windows of the normalized training log-returns.
    validation_dataset : SlidingWindowReconstructionDataset
        Windows of the validation log-returns, normalized with the
        training statistics.
    full_dataset : SlidingWindowReconstructionDataset
        Windows of the whole log-return series, normalized with the
        training statistics.
    return_train_mean : float
        Mean of the training log-returns.
    return_train_std : float
        Standard deviation of the training log-returns.
    """

    train_dataset: SlidingWindowReconstructionDataset
    validation_dataset: SlidingWindowReconstructionDataset
    full_dataset: SlidingWindowReconstructionDataset
    return_train_mean: float
    return_train_std: float


@dataclass
class AutoencoderTrainingResult:
    """Outcome of a complete autoencoder training run.

    Parameters
    ----------
    model : SimpleAutoencoder
        Trained autoencoder.
    training_loss_history : list[float]
        Mean training MSE of each epoch, in order.
    validation_loss : float
        Mean MSE over the validation windows.
    anomaly_threshold : float
        Reconstruction score above which a window is an outlier.
    return_train_mean : float
        Mean of the training log-returns, needed for inference.
    return_train_std : float
        Standard deviation of the training log-returns, needed for
        inference.
    model_file_path : Path
        Local path of the saved model weights.
    """

    model: SimpleAutoencoder
    training_loss_history: list[float]
    validation_loss: float
    anomaly_threshold: float
    return_train_mean: float
    return_train_std: float
    model_file_path: Path


def compute_reconstruction_scores(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    """Compute the mean squared reconstruction error of each window.

    Parameters
    ----------
    model : nn.Module
        Autoencoder used to reconstruct the windows. It is put in
        evaluation mode.
    input_tensor : torch.Tensor
        Tensor of shape `(num_windows, window_size)`, on the same
        device as `model`.

    Returns
    -------
    np.ndarray
        One-dimensional array of shape `(num_windows,)` with the mean
        squared error of each window.
    """
    model.eval()
    with torch.no_grad():
        reconstructed_tensor = model(input_tensor)
        squared_errors = (reconstructed_tensor - input_tensor) ** 2

    return squared_errors.mean(dim=1).cpu().numpy()


def build_ticker_slug(ticker: str) -> str:
    """Convert a ticker into a name safe for files and wandb artifacts.

    Parameters
    ----------
    ticker : str
        Ticker symbol, for example `"PETR4.SA"` or `"^BVSP"`.

    Returns
    -------
    str
        Lowercase ticker in which every character other than a letter,
        a digit, `.`, `_` or `-` is replaced by `-`.

    Example
    -------
    >>> build_ticker_slug("^BVSP")
    '-bvsp'
    """
    return re.sub(r"[^a-z0-9._-]", "-", ticker.lower())


class AutoencoderTrainer:
    """Train, evaluate and register a SimpleAutoencoder for one ticker.

    The trainer downloads the closing prices, computes and normalizes
    the log-returns, trains the autoencoder on sliding windows, measures
    the validation loss, computes the anomaly threshold and registers
    the trained weights in wandb together with everything inference
    needs (threshold and normalization statistics).

    Parameters
    ----------
    ticker : str
        Ticker symbol whose prices are used for training.
    config : AutoencoderTrainingConfig
        Training configuration.
    experiment_tracker : WandbExperimentTracker | None
        Tracker used to log metrics and the model. When `None`, a
        `WandbExperimentTracker` is created from `config.tracking`.

    Example
    -------
    >>> trainer = AutoencoderTrainer(
    ...     ticker="AAPL",
    ...     config=AutoencoderTrainingConfig(),
    ... )
    >>> training_result = trainer.train()
    >>> training_result.anomaly_threshold > 0
    True
    """

    def __init__(
        self,
        ticker: str,
        config: AutoencoderTrainingConfig,
        experiment_tracker: WandbExperimentTracker | None = None,
    ) -> None:
        """Store the settings, pick the device and build the tracker."""
        self.ticker = ticker
        self.config = config
        if config.device is not None:
            self.device = config.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if experiment_tracker is None:
            experiment_tracker = self._build_experiment_tracker()
        self.experiment_tracker = experiment_tracker

    @property
    def model_name(self) -> str:
        """Return the wandb artifact name of the model for the ticker."""
        return MODEL_NAME_PREFIX + build_ticker_slug(self.ticker)

    @property
    def model_file_path(self) -> Path:
        """Return the local path where the model weights are saved."""
        output_directory = Path(self.config.tracking.output_directory)
        return output_directory / f"{self.model_name}.pt"

    def _build_experiment_tracker(self) -> WandbExperimentTracker:
        """Create the default wandb tracker for this training run.

        Returns
        -------
        WandbExperimentTracker
            Tracker whose run `config` holds the ticker, the flattened
            training configuration and the optimizer, loss and model
            names.
        """
        run_config = {
            "ticker": self.ticker,
            **self.config.to_flat_dict(),
            "optimizer": "Adam",
            "loss_function": "MSE",
            "model": "SimpleAutoencoder",
        }
        return WandbExperimentTracker(
            project_name=self.config.tracking.project_name,
            config=run_config,
            env_file_path=self.config.tracking.env_file_path,
        )

    def prepare_data(self) -> PreparedAutoencoderData:
        """Download the prices and build the normalized window datasets.

        Returns
        -------
        PreparedAutoencoderData
            Training, validation and full-series datasets, plus the
            mean and standard deviation of the training log-returns.

        Raises
        ------
        ValueError
            If the training or the validation split has fewer
            log-returns than `window_size`.
        """
        market_data_config = self.config.market_data
        price_frame = fetch_price_series(
            self.ticker,
            market_data_config.start_date,
            market_data_config.end_date,
        )
        price_tensor = torch.tensor(
            price_frame["close_price"].to_numpy(),
            dtype=torch.float32,
        ).flatten()
        log_price_tensor = torch.log(price_tensor)
        log_return_tensor = log_price_tensor[1:] - log_price_tensor[:-1]

        train_size = int(
            len(log_return_tensor) * market_data_config.train_split_ratio
        )
        train_return_tensor = log_return_tensor[:train_size]
        validation_return_tensor = log_return_tensor[train_size:]

        window_size = market_data_config.window_size
        if (
            len(train_return_tensor) < window_size
            or len(validation_return_tensor) < window_size
        ):
            raise ValueError(
                f"Not enough data for ticker '{self.ticker}': "
                f"{len(train_return_tensor)} training and "
                f"{len(validation_return_tensor)} validation log-returns "
                f"for window_size={window_size}."
            )

        normalized_train_tensor, return_train_mean, return_train_std = (
            normalize_series(train_return_tensor)
        )
        normalized_validation_tensor, _, _ = normalize_series(
            validation_return_tensor,
            mean=return_train_mean,
            std=return_train_std,
        )
        normalized_full_tensor, _, _ = normalize_series(
            log_return_tensor,
            mean=return_train_mean,
            std=return_train_std,
        )

        return PreparedAutoencoderData(
            train_dataset=SlidingWindowReconstructionDataset(
                series_values=normalized_train_tensor,
                window_size=window_size,
            ),
            validation_dataset=SlidingWindowReconstructionDataset(
                series_values=normalized_validation_tensor,
                window_size=window_size,
            ),
            full_dataset=SlidingWindowReconstructionDataset(
                series_values=normalized_full_tensor,
                window_size=window_size,
            ),
            return_train_mean=return_train_mean,
            return_train_std=return_train_std,
        )

    def build_model(self) -> SimpleAutoencoder:
        """Create the autoencoder described by the configuration.

        Returns
        -------
        SimpleAutoencoder
            Untrained autoencoder already moved to `device`.
        """
        model = SimpleAutoencoder(
            input_dim=self.config.market_data.window_size,
            hidden_dim=self.config.architecture.hidden_layer_size,
            latent_dim=self.config.architecture.latent_layer_size,
        )
        return model.to(self.device)

    def _fit(
        self,
        model: SimpleAutoencoder,
        train_dataset: SlidingWindowReconstructionDataset,
    ) -> list[float]:
        """Run the training epochs and log the loss of each one.

        Parameters
        ----------
        model : SimpleAutoencoder
            Autoencoder to train in place.
        train_dataset : SlidingWindowReconstructionDataset
            Windows of the normalized training log-returns.

        Returns
        -------
        list[float]
            Mean training MSE of each epoch, weighted by batch size.
        """
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(self.config.optimization.random_seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.market_data.batch_size,
            shuffle=True,
            generator=shuffle_generator,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.optimization.learning_rate,
        )
        loss_function = nn.MSELoss()

        training_loss_history = []
        for epoch in range(1, self.config.optimization.number_of_epochs + 1):
            model.train()
            epoch_loss_sum = 0.0
            for window_batch in train_loader:
                window_batch = window_batch.to(self.device)
                optimizer.zero_grad()
                batch_loss = loss_function(model(window_batch), window_batch)
                batch_loss.backward()
                optimizer.step()
                epoch_loss_sum += batch_loss.item() * len(window_batch)

            epoch_loss = epoch_loss_sum / len(train_dataset)
            training_loss_history.append(epoch_loss)
            self.experiment_tracker.log_metrics(
                {"epoch": epoch, "train/loss_mse": epoch_loss}
            )

        return training_loss_history

    def _evaluate(
        self,
        model: SimpleAutoencoder,
        validation_dataset: SlidingWindowReconstructionDataset,
    ) -> float:
        """Compute and log the mean MSE over the validation windows.

        Parameters
        ----------
        model : SimpleAutoencoder
            Trained autoencoder.
        validation_dataset : SlidingWindowReconstructionDataset
            Windows of the normalized validation log-returns.

        Returns
        -------
        float
            Mean validation MSE, weighted by batch size.
        """
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.config.market_data.batch_size,
            shuffle=False,
        )
        loss_function = nn.MSELoss()

        model.eval()
        validation_loss_sum = 0.0
        with torch.no_grad():
            for window_batch in validation_loader:
                window_batch = window_batch.to(self.device)
                batch_loss = loss_function(model(window_batch), window_batch)
                validation_loss_sum += batch_loss.item() * len(window_batch)

        validation_loss = validation_loss_sum / len(validation_dataset)
        self.experiment_tracker.log_metrics(
            {"validation/loss_mse": validation_loss}
        )
        return validation_loss

    def _compute_anomaly_threshold(
        self,
        model: SimpleAutoencoder,
        full_dataset: SlidingWindowReconstructionDataset,
    ) -> float:
        """Compute the reconstruction score that marks an outlier.

        Parameters
        ----------
        model : SimpleAutoencoder
            Trained autoencoder.
        full_dataset : SlidingWindowReconstructionDataset
            Windows of the whole normalized log-return series.

        Returns
        -------
        float
            Percentile `100 * (1 - contamination)` of the
            reconstruction scores of every window of the series.
        """
        full_window_tensor = full_dataset.series_values.unfold(
            0,
            full_dataset.window_size,
            1,
        ).to(self.device)
        reconstruction_scores = compute_reconstruction_scores(
            model,
            full_window_tensor,
        )
        anomaly_percentile = 100 * (1 - self.config.contamination)

        return float(np.percentile(reconstruction_scores, anomaly_percentile))

    def train(self) -> AutoencoderTrainingResult:
        """Run the full training pipeline and register the model.

        The wandb run is always finished, even when training,
        evaluation or model registration raises an exception; the
        original exception is propagated.

        Returns
        -------
        AutoencoderTrainingResult
            Trained model, loss history, validation loss, anomaly
            threshold, normalization statistics and weights path.

        Raises
        ------
        ValueError
            If there is not enough data for the configured window.
        """
        prepared_data = self.prepare_data()
        torch.manual_seed(self.config.optimization.random_seed)
        model = self.build_model()

        self.experiment_tracker.start_run()
        try:
            training_loss_history = self._fit(
                model,
                prepared_data.train_dataset,
            )
            validation_loss = self._evaluate(
                model,
                prepared_data.validation_dataset,
            )
            anomaly_threshold = self._compute_anomaly_threshold(
                model,
                prepared_data.full_dataset,
            )
            self.model_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.experiment_tracker.log_model(
                model=model,
                model_name=self.model_name,
                model_file_path=str(self.model_file_path),
                description=(
                    "SimpleAutoencoder trained on normalized log-return "
                    f"windows of {self.ticker} for outlier detection."
                ),
                metadata={
                    "final_train_loss": training_loss_history[-1],
                    "validation_loss": validation_loss,
                    "anomaly_threshold": anomaly_threshold,
                    "return_train_mean": prepared_data.return_train_mean,
                    "return_train_std": prepared_data.return_train_std,
                },
            )
        finally:
            self.experiment_tracker.finish_run()

        return AutoencoderTrainingResult(
            model=model,
            training_loss_history=training_loss_history,
            validation_loss=validation_loss,
            anomaly_threshold=anomaly_threshold,
            return_train_mean=prepared_data.return_train_mean,
            return_train_std=prepared_data.return_train_std,
            model_file_path=self.model_file_path,
        )
