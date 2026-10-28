"""Training pipeline of the dense log-return autoencoder."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from generative_models.common.normalization import normalize_series
from generative_models.data.datasets import (
    SlidingWindowReconstructionDataset,
)
from generative_models.data.market_data import fetch_price_series
from generative_models.models.autoencoder.autoencoder import (
    SimpleAutoencoder,
)
from generative_models.models.autoencoder.config import (
    AutoencoderTrainingConfig,
)
from generative_models.tracking.wandb_tracker import WandbExperimentTracker

MODEL_NAME_PREFIX = "simple-autoencoder-"
UNSAFE_SLUG_CHARACTERS = re.compile(r"[^a-z0-9._-]")


@dataclass
class PreparedAutoencoderData:
    """Datasets and normalization statistics ready for training.

    Parameters
    ----------
    train_dataset : SlidingWindowReconstructionDataset
        Windows of the training split.
    validation_dataset : SlidingWindowReconstructionDataset
        Windows of the validation split.
    full_dataset : SlidingWindowReconstructionDataset
        Windows of the whole chronological series, used to compute the
        anomaly threshold.
    return_train_mean : float
        Mean of the training log-returns, needed to normalize new data
        at inference time.
    return_train_std : float
        Standard deviation of the training log-returns, needed to
        normalize new data at inference time.
    """

    train_dataset: SlidingWindowReconstructionDataset
    validation_dataset: SlidingWindowReconstructionDataset
    full_dataset: SlidingWindowReconstructionDataset
    return_train_mean: float
    return_train_std: float


@dataclass
class AutoencoderTrainingResult:
    """Outcome of one autoencoder training run.

    Parameters
    ----------
    model : SimpleAutoencoder
        Trained autoencoder.
    training_loss_history : list[float]
        Average training loss of each epoch, in order.
    validation_loss : float
        Mean reconstruction error over the validation windows.
    anomaly_threshold : float
        Reconstruction error above which a window is an outlier.
    return_train_mean : float
        Mean of the training log-returns.
    return_train_std : float
        Standard deviation of the training log-returns.
    model_file_path : Path
        Local path where the model weights were written.
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
    """Compute the per-window reconstruction error of a model.

    Parameters
    ----------
    model : nn.Module
        Trained autoencoder used to reconstruct the windows.
    input_tensor : torch.Tensor
        Tensor of shape `(num_windows, window_size)` with the windows
        to be scored.

    Returns
    -------
    np.ndarray
        One-dimensional array of shape `(num_windows,)` with the mean
        squared reconstruction error of each window. The higher the
        error, the harder that window was to reconstruct, which
        suggests an unusual pattern of returns.

    Example
    -------
    >>> scores = compute_reconstruction_scores(model, windows)
    >>> scores.shape
    (96,)
    """
    criterion = nn.MSELoss(reduction="none")
    model.eval()
    with torch.no_grad():
        reconstruction = model(input_tensor)
        squared_errors = criterion(reconstruction, input_tensor)
        window_scores = squared_errors.mean(dim=1)

    return window_scores.cpu().numpy()


def build_ticker_slug(ticker: str) -> str:
    """Turn a ticker symbol into a slug safe for names and paths.

    Parameters
    ----------
    ticker : str
        Ticker symbol, as given by the user (for example,
        `"PETR4.SA"`).

    Returns
    -------
    str
        Lowercase version of `ticker` in which every character other
        than a letter, a digit, `.`, `_` or `-` is replaced by `-`.

    Example
    -------
    >>> build_ticker_slug("PETR4.SA")
    'petr4.sa'
    >>> build_ticker_slug("^BVSP")
    '-bvsp'
    """
    return UNSAFE_SLUG_CHARACTERS.sub("-", ticker.lower())


class AutoencoderTrainer:
    """Full training pipeline of the autoencoder for a single ticker.

    The trainer reproduces, end to end, what the reference notebook
    does cell by cell: it collects the closing prices, turns them into
    log-returns, splits and normalizes them, builds the sliding
    windows, trains the autoencoder, evaluates it on the validation
    split, derives the anomaly threshold from the reconstruction error
    and registers the trained weights as a wandb artifact.

    Parameters
    ----------
    ticker : str
        Ticker symbol of the asset to train on.
    config : AutoencoderTrainingConfig
        Configuration of the run.
    experiment_tracker : WandbExperimentTracker | None
        Tracker used to log metrics and register the model. When
        `None`, one is built from `config.tracking`.

    Attributes
    ----------
    ticker : str
        Ticker symbol of the asset to train on.
    config : AutoencoderTrainingConfig
        Configuration of the run.
    device : str
        Device used for training, either taken from `config.device` or
        decided automatically.
    experiment_tracker : WandbExperimentTracker
        Tracker used to log metrics and register the model.

    Example
    -------
    >>> trainer = AutoencoderTrainer(
    ...     ticker="PETR4.SA",
    ...     config=AutoencoderTrainingConfig(),
    ... )
    >>> result = trainer.train()
    >>> result.anomaly_threshold > 0
    True
    """

    def __init__(
        self,
        ticker: str,
        config: AutoencoderTrainingConfig,
        experiment_tracker: WandbExperimentTracker | None = None,
    ) -> None:
        """Store the ticker and configuration and resolve the device."""
        self.ticker = ticker
        self.config = config

        if config.device is not None:
            self.device = config.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if experiment_tracker is not None:
            self.experiment_tracker = experiment_tracker
        else:
            self.experiment_tracker = self._build_experiment_tracker()

    def _build_experiment_tracker(self) -> WandbExperimentTracker:
        """Build the default tracker from the tracking configuration.

        Returns
        -------
        WandbExperimentTracker
            Tracker pointing at the configured project, whose run
            `config` holds the ticker, every flattened configuration
            field and the fixed choices of optimizer, loss function
            and model.
        """
        return WandbExperimentTracker(
            project_name=self.config.tracking.project_name,
            config={
                "ticker": self.ticker,
                **self.config.to_flat_dict(),
                "optimizer": "Adam",
                "loss_function": "MSE",
                "model": "SimpleAutoencoder",
            },
            env_file_path=self.config.tracking.env_file_path,
        )

    @property
    def model_name(self) -> str:
        """Return the name used for the model artifact in wandb."""
        return MODEL_NAME_PREFIX + build_ticker_slug(self.ticker)

    @property
    def model_file_path(self) -> Path:
        """Return the local path of the trained model weights."""
        output_directory = Path(self.config.tracking.output_directory)

        return output_directory / f"{self.model_name}.pt"

    def prepare_data(self) -> PreparedAutoencoderData:
        """Collect the prices and build the normalized window datasets.

        Returns
        -------
        PreparedAutoencoderData
            The training, validation and full-series datasets, plus the
            mean and standard deviation of the training log-returns.

        Raises
        ------
        ValueError
            If either split holds fewer log-returns than the window
            size, so that no window can be built.
        """
        market_data = self.config.market_data
        window_size = market_data.window_size

        price_data = fetch_price_series(
            self.ticker,
            market_data.start_date,
            market_data.end_date,
        )
        price_tensor = torch.tensor(
            price_data["close_price"].to_numpy(),
            dtype=torch.float32,
        )
        log_return_tensor = torch.log(price_tensor[1:]) - torch.log(
            price_tensor[:-1]
        )

        train_split_index = int(
            len(log_return_tensor) * market_data.train_split_ratio
        )
        train_return_tensor = log_return_tensor[:train_split_index]
        validation_return_tensor = log_return_tensor[train_split_index:]

        if (
            len(train_return_tensor) < window_size
            or len(validation_return_tensor) < window_size
        ):
            raise ValueError(
                f"Not enough data for ticker '{self.ticker}': both the "
                f"training and the validation split need at least "
                f"{window_size} log-returns."
            )

        (
            normalized_train_tensor,
            return_train_mean,
            return_train_std,
        ) = normalize_series(train_return_tensor)
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
        """Build the autoencoder described by the configuration.

        Returns
        -------
        SimpleAutoencoder
            Autoencoder sized after the window and the architecture
            configuration, already moved to `device`.
        """
        return SimpleAutoencoder(
            input_dim=self.config.market_data.window_size,
            hidden_dim=self.config.architecture.hidden_layer_size,
            latent_dim=self.config.architecture.latent_layer_size,
        ).to(self.device)

    def _fit(
        self,
        model: SimpleAutoencoder,
        train_dataset: SlidingWindowReconstructionDataset,
    ) -> list[float]:
        """Run the optimization loop over the training windows.

        Parameters
        ----------
        model : SimpleAutoencoder
            Autoencoder to be optimized in place.
        train_dataset : SlidingWindowReconstructionDataset
            Windows used for training.

        Returns
        -------
        list[float]
            Average training loss of each epoch, in order. Each value
            is also logged to the experiment tracker.
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
        )
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.optimization.learning_rate,
        )

        training_loss_history = []
        for epoch_index in range(self.config.optimization.number_of_epochs):
            model.train()
            epoch_loss_total = 0.0

            for batch_windows in train_loader:
                device_windows = batch_windows.to(self.device)

                reconstruction = model(device_windows)
                loss_value = loss_function(reconstruction, device_windows)
                loss_value.backward()

                optimizer.step()
                optimizer.zero_grad()

                epoch_loss_total += (
                    loss_value.item() * device_windows.shape[0]
                )

            epoch_average_loss = epoch_loss_total / len(train_dataset)
            training_loss_history.append(epoch_average_loss)

            self.experiment_tracker.log_metrics(
                {
                    "epoch": epoch_index + 1,
                    "train/loss_mse": epoch_average_loss,
                }
            )

        return training_loss_history

    def _evaluate(
        self,
        model: SimpleAutoencoder,
        validation_dataset: SlidingWindowReconstructionDataset,
    ) -> float:
        """Measure the reconstruction error on the validation windows.

        Parameters
        ----------
        model : SimpleAutoencoder
            Trained autoencoder.
        validation_dataset : SlidingWindowReconstructionDataset
            Windows never seen during training.

        Returns
        -------
        float
            Mean squared reconstruction error over every validation
            window, also logged to the experiment tracker.
        """
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.config.market_data.batch_size,
            shuffle=False,
        )
        loss_function = nn.MSELoss()

        model.eval()
        validation_loss_total = 0.0
        with torch.no_grad():
            for batch_windows in validation_loader:
                device_windows = batch_windows.to(self.device)
                reconstruction = model(device_windows)
                loss_value = loss_function(reconstruction, device_windows)
                validation_loss_total += (
                    loss_value.item() * device_windows.shape[0]
                )

        validation_loss = validation_loss_total / len(validation_dataset)
        self.experiment_tracker.log_metrics(
            {"validation/loss_mse": validation_loss}
        )

        return validation_loss

    def _compute_anomaly_threshold(
        self,
        model: SimpleAutoencoder,
        full_dataset: SlidingWindowReconstructionDataset,
    ) -> float:
        """Derive the outlier cut-off from the whole series.

        Parameters
        ----------
        model : SimpleAutoencoder
            Trained autoencoder.
        full_dataset : SlidingWindowReconstructionDataset
            Windows of the whole chronological series.

        Returns
        -------
        float
            The `100 * (1 - contamination)` percentile of the
            reconstruction scores, above which a window is considered
            an outlier.
        """
        full_loader = DataLoader(
            full_dataset,
            batch_size=len(full_dataset),
            shuffle=False,
        )
        full_windows_tensor = next(iter(full_loader)).to(self.device)
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

    def train(self) -> AutoencoderTrainingResult:
        """Run the whole pipeline and register the trained model.

        The wandb run is always finished, even when one of the training
        steps fails. In that case the original exception keeps
        propagating.

        Returns
        -------
        AutoencoderTrainingResult
            The trained model, the loss history, the validation loss,
            the anomaly threshold, the normalization statistics and the
            path of the saved weights.

        Example
        -------
        >>> result = trainer.train()
        >>> len(result.training_loss_history)
        300
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

            model_file_path = self.model_file_path
            model_file_path.parent.mkdir(parents=True, exist_ok=True)

            self.experiment_tracker.log_model(
                model=model,
                model_name=self.model_name,
                model_file_path=str(model_file_path),
                description=(
                    "Dense autoencoder trained on normalized log-return "
                    f"windows of {self.ticker}, whose reconstruction "
                    "error is used as an outlier score."
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
            model_file_path=model_file_path,
        )
