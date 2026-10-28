"""Training pipeline for the dense autoencoder anomaly detector."""

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
        Windows of the normalized validation log-returns.
    full_dataset : SlidingWindowReconstructionDataset
        Windows of the whole normalized log-return series.
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
    """Outcome of a training run.

    Parameters
    ----------
    model : SimpleAutoencoder
        Trained autoencoder.
    training_loss_history : list[float]
        Mean training loss of each epoch.
    validation_loss : float
        Mean squared error on the validation windows.
    anomaly_threshold : float
        Reconstruction score above which a window is an outlier.
    return_train_mean : float
        Mean of the training log-returns.
    return_train_std : float
        Standard deviation of the training log-returns.
    model_file_path : Path
        Local path of the saved weights.
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
        Autoencoder used to reconstruct the windows.
    input_tensor : torch.Tensor
        Tensor of shape `(num_windows, window_size)`.

    Returns
    -------
    np.ndarray
        One-dimensional array of shape `(num_windows,)` with the mean
        squared error of each window.

    Example
    -------
    >>> scores = compute_reconstruction_scores(
    ...     model=SimpleAutoencoder(input_dim=5),
    ...     input_tensor=torch.randn(7, 5),
    ... )
    >>> scores.shape
    (7,)
    """
    model.eval()
    with torch.no_grad():
        reconstruction = model(input_tensor)
        squared_error = (reconstruction - input_tensor) ** 2
        window_scores = squared_error.mean(dim=1)

    return window_scores.cpu().numpy()


def build_ticker_slug(ticker: str) -> str:
    """Build a name-safe slug from a ticker symbol.

    Parameters
    ----------
    ticker : str
        Ticker symbol (for example, `"PETR4.SA"`).

    Returns
    -------
    str
        Lowercase ticker where every character other than a letter,
        digit, `.`, `_` or `-` is replaced by `-`.

    Example
    -------
    >>> build_ticker_slug("^BVSP")
    '-bvsp'
    """
    return re.sub(r"[^a-z0-9._-]", "-", ticker.lower())


class AutoencoderTrainer:
    """Train a `SimpleAutoencoder` on the log-returns of one ticker.

    The trainer collects the prices, builds normalized sliding-window
    datasets, trains the model, evaluates it on the validation split,
    computes the anomaly threshold and registers the model with the
    experiment tracker.

    Parameters
    ----------
    ticker : str
        Ticker symbol to train on.
    config : AutoencoderTrainingConfig
        Training configuration.
    experiment_tracker : WandbExperimentTracker | None
        Tracker used to log the run. When `None`, a
        `WandbExperimentTracker` is built from `config.tracking`.

    Example
    -------
    >>> trainer = AutoencoderTrainer(
    ...     ticker="AAPL",
    ...     config=AutoencoderTrainingConfig(),
    ... )
    >>> result = trainer.train()
    >>> result.model_file_path
    PosixPath('artifacts/simple-autoencoder-aapl.pt')
    """

    def __init__(
        self,
        ticker: str,
        config: AutoencoderTrainingConfig,
        experiment_tracker: WandbExperimentTracker | None = None,
    ) -> None:
        """Store the settings and resolve the device and tracker."""
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
        """Return the artifact name of the model for this ticker."""
        return MODEL_NAME_PREFIX + build_ticker_slug(self.ticker)

    @property
    def model_file_path(self) -> Path:
        """Return the local path where the weights are saved."""
        return (
            Path(self.config.tracking.output_directory)
            / f"{self.model_name}.pt"
        )

    def _build_experiment_tracker(self) -> WandbExperimentTracker:
        """Build the default wandb tracker for this run.

        Returns
        -------
        WandbExperimentTracker
            Tracker configured with the project name, the `.env` path
            and the flattened run configuration.
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
        """Fetch prices and build the normalized window datasets.

        Returns
        -------
        PreparedAutoencoderData
            Training, validation and full-series datasets, plus the
            training normalization statistics.

        Raises
        ------
        ValueError
            If the training or validation split has fewer log-returns
            than `window_size`.
        """
        market_data_config = self.config.market_data
        window_size = market_data_config.window_size

        price_data = fetch_price_series(
            self.ticker,
            market_data_config.start_date,
            market_data_config.end_date,
        )
        price_tensor = torch.tensor(
            np.asarray(price_data["close_price"].values).reshape(-1),
            dtype=torch.float32,
        )
        log_price_tensor = torch.log(price_tensor)
        log_return_tensor = log_price_tensor[1:] - log_price_tensor[:-1]

        train_size = int(
            len(log_return_tensor) * market_data_config.train_split_ratio
        )
        train_return_tensor = log_return_tensor[:train_size]
        validation_return_tensor = log_return_tensor[train_size:]
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
                normalized_train_tensor, window_size
            ),
            validation_dataset=SlidingWindowReconstructionDataset(
                normalized_validation_tensor, window_size
            ),
            full_dataset=SlidingWindowReconstructionDataset(
                normalized_full_tensor, window_size
            ),
            return_train_mean=return_train_mean,
            return_train_std=return_train_std,
        )

    def build_model(self) -> SimpleAutoencoder:
        """Instantiate the autoencoder on the training device.

        Returns
        -------
        SimpleAutoencoder
            Untrained autoencoder sized according to the configuration.
        """
        return SimpleAutoencoder(
            input_dim=self.config.market_data.window_size,
            hidden_dim=self.config.architecture.hidden_layer_size,
            latent_dim=self.config.architecture.latent_layer_size,
        ).to(self.device)

    def _stack_windows(
        self,
        dataset: SlidingWindowReconstructionDataset,
    ) -> torch.Tensor:
        """Stack every window of a dataset into one tensor.

        Parameters
        ----------
        dataset : SlidingWindowReconstructionDataset
            Dataset whose windows are stacked.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(num_windows, window_size)` on the device.
        """
        return torch.stack(
            [dataset[index] for index in range(len(dataset))]
        ).to(self.device)

    def _fit(
        self,
        model: SimpleAutoencoder,
        train_dataset: SlidingWindowReconstructionDataset,
    ) -> list[float]:
        """Run the training loop and log the loss of each epoch.

        Parameters
        ----------
        model : SimpleAutoencoder
            Model to train in place.
        train_dataset : SlidingWindowReconstructionDataset
            Training windows.

        Returns
        -------
        list[float]
            Mean training loss of each epoch.
        """
        optimization_config = self.config.optimization
        data_generator = torch.Generator()
        data_generator.manual_seed(optimization_config.random_seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.market_data.batch_size,
            shuffle=True,
            generator=data_generator,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimization_config.learning_rate,
        )
        loss_function = nn.MSELoss()

        training_loss_history = []
        for epoch_index in range(optimization_config.number_of_epochs):
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
                {"epoch": epoch_index + 1, "train/loss_mse": epoch_loss}
            )

        return training_loss_history

    def _evaluate(
        self,
        model: SimpleAutoencoder,
        validation_dataset: SlidingWindowReconstructionDataset,
    ) -> float:
        """Compute and log the mean squared error on validation windows.

        Parameters
        ----------
        model : SimpleAutoencoder
            Trained model.
        validation_dataset : SlidingWindowReconstructionDataset
            Validation windows.

        Returns
        -------
        float
            Mean squared reconstruction error on the validation set.
        """
        validation_tensor = self._stack_windows(validation_dataset)
        model.eval()
        with torch.no_grad():
            validation_loss = nn.functional.mse_loss(
                model(validation_tensor), validation_tensor
            ).item()

        self.experiment_tracker.log_metrics(
            {"validation/loss_mse": validation_loss}
        )
        return validation_loss

    def _compute_anomaly_threshold(
        self,
        model: SimpleAutoencoder,
        full_dataset: SlidingWindowReconstructionDataset,
    ) -> float:
        """Compute the reconstruction score threshold for outliers.

        Parameters
        ----------
        model : SimpleAutoencoder
            Trained model.
        full_dataset : SlidingWindowReconstructionDataset
            Windows of the whole series.

        Returns
        -------
        float
            Percentile `100 * (1 - contamination)` of the scores.
        """
        reconstruction_scores = compute_reconstruction_scores(
            model, self._stack_windows(full_dataset)
        )
        return float(
            np.percentile(
                reconstruction_scores,
                100 * (1 - self.config.contamination),
            )
        )

    def train(self) -> AutoencoderTrainingResult:
        """Run the full pipeline: data, training, evaluation, registry.

        The tracker run is always finished, even when a step fails;
        the original exception is propagated.

        Returns
        -------
        AutoencoderTrainingResult
            Trained model and the training summary.
        """
        prepared_data = self.prepare_data()
        torch.manual_seed(self.config.optimization.random_seed)
        model = self.build_model()

        self.experiment_tracker.start_run()
        try:
            training_loss_history = self._fit(
                model, prepared_data.train_dataset
            )
            validation_loss = self._evaluate(
                model, prepared_data.validation_dataset
            )
            anomaly_threshold = self._compute_anomaly_threshold(
                model, prepared_data.full_dataset
            )

            self.model_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.experiment_tracker.log_model(
                model=model,
                model_name=self.model_name,
                model_file_path=str(self.model_file_path),
                description=(
                    "Dense autoencoder for log-return outlier detection "
                    f"on {self.ticker}."
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
