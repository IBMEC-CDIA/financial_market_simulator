"""Training pipeline for the simple autoencoder on a single ticker."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
    """Windowed datasets and normalization statistics for one ticker.

    Parameters
    ----------
    train_dataset : SlidingWindowReconstructionDataset
        Windows built from the normalized training log-returns.
    validation_dataset : SlidingWindowReconstructionDataset
        Windows built from the normalized validation log-returns.
    full_dataset : SlidingWindowReconstructionDataset
        Windows built from the whole normalized log-return series, in
        chronological order, used to compute the anomaly threshold.
    return_train_mean : float
        Mean of the training log-returns used for normalization.
    return_train_std : float
        Standard deviation of the training log-returns used for
        normalization.
    """

    train_dataset: SlidingWindowReconstructionDataset
    validation_dataset: SlidingWindowReconstructionDataset
    full_dataset: SlidingWindowReconstructionDataset
    return_train_mean: float
    return_train_std: float


@dataclass
class AutoencoderTrainingResult:
    """Outputs produced by a completed training run.

    Parameters
    ----------
    model : SimpleAutoencoder
        Trained autoencoder, in evaluation mode.
    training_loss_history : list[float]
        Mean training MSE of each epoch.
    validation_loss : float
        Mean reconstruction MSE on the validation windows.
    anomaly_threshold : float
        Reconstruction error above which a window is classified as an
        outlier, computed over the full series.
    return_train_mean : float
        Mean of the training log-returns used for normalization.
    return_train_std : float
        Standard deviation of the training log-returns used for
        normalization.
    model_file_path : Path
        Local path where the model weights are saved.
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
    """Compute the per-window reconstruction error scores.

    Parameters
    ----------
    model : nn.Module
        Trained autoencoder model.
    input_tensor : torch.Tensor
        Tensor of shape `(num_windows, window_size)` with the windows
        to be scored, on the same device as `model`.

    Returns
    -------
    np.ndarray
        One-dimensional array with the mean squared reconstruction
        error of each window.
    """
    criterion = nn.MSELoss(reduction="none")
    model.eval()
    with torch.no_grad():
        reconstruction = model(input_tensor)
        errors = criterion(reconstruction, input_tensor).mean(dim=1)
    return errors.cpu().numpy()


def build_ticker_slug(ticker: str) -> str:
    """Convert a ticker symbol into a name safe for files and artifacts.

    Parameters
    ----------
    ticker : str
        Ticker symbol (for example, `"PETR4.SA"` or `"^BVSP"`).

    Returns
    -------
    str
        Lowercase version of `ticker` in which every character other
        than letters, digits, `.`, `_` and `-` is replaced by `-`.

    Example
    -------
    >>> build_ticker_slug("^BVSP")
    '-bvsp'
    """
    return re.sub(r"[^a-z0-9._-]", "-", ticker.lower())


class AutoencoderTrainer:
    """Train a `SimpleAutoencoder` on the log-returns of one ticker.

    The pipeline mirrors the one used in the autoencoder outlier
    notebooks: daily closing prices are fetched with
    `fetch_price_series`, converted into log-returns, split
    chronologically into training and validation sets, standardized
    with the training statistics and turned into sliding windows. The
    autoencoder is trained with Adam on the reconstruction MSE, the
    experiment is tracked with `WandbExperimentTracker`, and the
    anomaly threshold is computed over the full series.

    Parameters
    ----------
    ticker : str
        Ticker symbol of the asset whose returns are modeled (for
        example, `"AAPL"` or `"PETR4.SA"`).
    config : AutoencoderTrainingConfig
        Every other setting required for data preparation, training
        and tracking.
    experiment_tracker : WandbExperimentTracker | None
        Tracker used to log the run. When `None`, a tracker is built
        from `config.tracking`, with the flattened configuration and
        the ticker as the run `config`.

    Example
    -------
    >>> config = AutoencoderTrainingConfig(
    ...     optimization=OptimizationConfig(number_of_epochs=100),
    ... )
    >>> trainer = AutoencoderTrainer(ticker="AAPL", config=config)
    >>> result = trainer.train()
    >>> result.model_file_path.name
    'simple-autoencoder-aapl.pt'
    """

    def __init__(
        self,
        ticker: str,
        config: AutoencoderTrainingConfig,
        experiment_tracker: WandbExperimentTracker | None = None,
    ) -> None:
        """Resolve the device and build the experiment tracker."""
        self.ticker = ticker
        self.config = config
        self.device = config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.experiment_tracker = (
            experiment_tracker or self._build_experiment_tracker()
        )

    @property
    def model_name(self) -> str:
        """Name of the wandb model artifact for this ticker.

        Returns
        -------
        str
            Artifact name in the form `simple-autoencoder-<ticker>`.
        """
        return f"{MODEL_NAME_PREFIX}{build_ticker_slug(self.ticker)}"

    @property
    def model_file_path(self) -> Path:
        """Local path where the trained model weights are saved.

        Returns
        -------
        Path
            Path inside `config.tracking.output_directory`, named
            after the ticker.
        """
        return (
            Path(self.config.tracking.output_directory)
            / f"{self.model_name}.pt"
        )

    def prepare_data(self) -> PreparedAutoencoderData:
        """Fetch prices and build the normalized window datasets.

        Returns
        -------
        PreparedAutoencoderData
            Training, validation and full-series window datasets,
            together with the training normalization statistics.

        Raises
        ------
        ValueError
            If the training or validation split is shorter than the
            configured window size.
        """
        market_data_config = self.config.market_data
        price_data = fetch_price_series(
            self.ticker,
            market_data_config.start_date,
            market_data_config.end_date,
        )
        price_tensor = torch.tensor(
            price_data["close_price"].to_numpy(),
            dtype=torch.float32,
        ).flatten()
        log_return_tensor = torch.log(price_tensor[1:]) - torch.log(
            price_tensor[:-1]
        )

        train_split_index = int(
            len(log_return_tensor) * market_data_config.train_split_ratio
        )
        train_log_return_tensor = log_return_tensor[:train_split_index]
        validation_log_return_tensor = log_return_tensor[train_split_index:]

        window_size = market_data_config.window_size
        if min(
            len(train_log_return_tensor),
            len(validation_log_return_tensor),
        ) < window_size:
            raise ValueError(
                f"Not enough data for ticker '{self.ticker}': both the "
                "training and validation splits need at least "
                f"{window_size} log-returns."
            )

        normalized_train_tensor, return_train_mean, return_train_std = (
            normalize_series(train_log_return_tensor)
        )
        normalized_validation_tensor, _, _ = normalize_series(
            validation_log_return_tensor,
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
        """Instantiate the autoencoder on the configured device.

        Returns
        -------
        SimpleAutoencoder
            Untrained autoencoder sized according to the
            configuration.
        """
        return SimpleAutoencoder(
            input_dim=self.config.market_data.window_size,
            hidden_dim=self.config.architecture.hidden_layer_size,
            latent_dim=self.config.architecture.latent_layer_size,
        ).to(self.device)

    def train(self) -> AutoencoderTrainingResult:
        """Run the full training pipeline for the configured ticker.

        Prepares the data, trains the autoencoder while logging the
        training loss of every epoch, evaluates it on the validation
        windows, computes the anomaly threshold, registers the model
        as a wandb artifact and finishes the run.

        Returns
        -------
        AutoencoderTrainingResult
            Trained model, loss history, validation loss, anomaly
            threshold, normalization statistics and model file path.
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
            self.experiment_tracker.log_metrics(
                {"validation/loss_mse": validation_loss}
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
                    "Simple dense autoencoder trained on the normalized "
                    f"daily log-returns of {self.ticker}."
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

    def _build_experiment_tracker(self) -> WandbExperimentTracker:
        """Build the default wandb tracker from the configuration.

        Returns
        -------
        WandbExperimentTracker
            Tracker whose run `config` holds the ticker, every
            configuration value and the optimizer, loss and model
            names.
        """
        return WandbExperimentTracker(
            project_name=self.config.tracking.project_name,
            config={
                "ticker": self.ticker,
                **self.config.to_flat_dict(),
                "optimizer": "Adam",
                "loss_function": "MSE",
                "model": SimpleAutoencoder.__name__,
            },
            env_file_path=self.config.tracking.env_file_path,
        )

    def _fit(
        self,
        model: SimpleAutoencoder,
        train_dataset: SlidingWindowReconstructionDataset,
    ) -> list[float]:
        """Optimize the model on the training windows.

        Parameters
        ----------
        model : SimpleAutoencoder
            Autoencoder to be trained in place.
        train_dataset : SlidingWindowReconstructionDataset
            Training windows.

        Returns
        -------
        list[float]
            Mean training MSE of each epoch.
        """
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(
            self.config.optimization.random_seed
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.market_data.batch_size,
            shuffle=True,
            generator=dataloader_generator,
            pin_memory=self.device.startswith("cuda"),
        )

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.optimization.learning_rate,
        )

        training_loss_history = []
        epoch_progress_bar = tqdm(
            range(self.config.optimization.number_of_epochs),
            desc=f"Training {self.ticker}",
            unit="epoch",
        )
        for epoch_index in epoch_progress_bar:
            model.train()
            epoch_loss_total = torch.zeros((), device=self.device)

            for batch_windows in train_loader:
                batch_windows = batch_windows.to(
                    self.device,
                    non_blocking=True,
                )
                optimizer.zero_grad(set_to_none=True)
                reconstruction = model(batch_windows)
                loss_value = loss_function(reconstruction, batch_windows)
                loss_value.backward()
                optimizer.step()
                epoch_loss_total += (
                    loss_value.detach() * batch_windows.shape[0]
                )

            epoch_average_loss = (
                epoch_loss_total / len(train_dataset)
            ).item()
            training_loss_history.append(epoch_average_loss)

            self.experiment_tracker.log_metrics(
                {
                    "epoch": epoch_index + 1,
                    "train/loss_mse": epoch_average_loss,
                }
            )
            epoch_progress_bar.set_postfix(loss=f"{epoch_average_loss:.6f}")

        return training_loss_history

    def _evaluate(
        self,
        model: SimpleAutoencoder,
        dataset: SlidingWindowReconstructionDataset,
    ) -> float:
        """Compute the mean reconstruction MSE over a dataset.

        Parameters
        ----------
        model : SimpleAutoencoder
            Trained autoencoder.
        dataset : SlidingWindowReconstructionDataset
            Windows to evaluate.

        Returns
        -------
        float
            Mean squared reconstruction error across every window.
        """
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.market_data.batch_size,
            shuffle=False,
        )
        loss_function = nn.MSELoss()

        model.eval()
        loss_total = torch.zeros((), device=self.device)
        with torch.no_grad():
            for batch_windows in data_loader:
                batch_windows = batch_windows.to(self.device)
                reconstruction = model(batch_windows)
                loss_value = loss_function(reconstruction, batch_windows)
                loss_total += loss_value * batch_windows.shape[0]

        return (loss_total / len(dataset)).item()

    def _compute_anomaly_threshold(
        self,
        model: SimpleAutoencoder,
        full_dataset: SlidingWindowReconstructionDataset,
    ) -> float:
        """Compute the reconstruction error cut-off for outliers.

        Parameters
        ----------
        model : SimpleAutoencoder
            Trained autoencoder.
        full_dataset : SlidingWindowReconstructionDataset
            Windows covering the whole series, in chronological
            order.

        Returns
        -------
        float
            The `(1 - contamination)` percentile of the per-window
            reconstruction errors.
        """
        full_windows_tensor = torch.stack(
            [full_dataset[index] for index in range(len(full_dataset))]
        ).to(self.device)
        reconstruction_scores = compute_reconstruction_scores(
            model,
            full_windows_tensor,
        )
        return float(
            np.percentile(
                reconstruction_scores,
                100 * (1 - self.config.contamination),
            )
        )
