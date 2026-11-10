"""Training pipeline for the per-ticker LSTM autoencoder anomaly detector."""

# pylint: disable=duplicate-code

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from generative_models.data.datasets import (
    SlidingWindowReconstructionDataset,
)
from generative_models.data.market_data import (
    add_log_return_column,
    add_split_column,
    fetch_price_series,
)
from generative_models.data.normalization import normalize_series
from generative_models.models.lstm_autoencoder.autoencoder import (
    LSTMAutoencoder,
)
from generative_models.models.lstm_autoencoder.config import (
    LSTMAutoencoderTrainingConfig,
)
from generative_models.tracking.wandb_tracker import (
    WandbExperimentTracker,
)

MODEL_NAME_PREFIX = "lstm-autoencoder-"
REQUIRED_PRICE_COLUMNS = ("close_price", "split")
TRAIN_SPLIT_LABEL = "train"
VALIDATION_SPLIT_LABEL = "validation"


@dataclass
class PreparedLSTMAutoencoderData:
    """Datasets and normalization statistics used for training.

    Parameters
    ----------
    train_dataset : SlidingWindowReconstructionDataset
        Windows of the normalized training log-returns.
    validation_dataset : SlidingWindowReconstructionDataset
        Windows of the validation log-returns, normalized with the
        training statistics.
    full_dataset : SlidingWindowReconstructionDataset
        Windows of the complete log-return series, normalized with the
        training statistics. It is used to compute the anomaly
        threshold.
    return_train_mean : float
        Mean of the training log-returns.
    return_train_std : float
        Standard deviation of the training log-returns.
    test_dataset : SlidingWindowReconstructionDataset | None
        Windows of the held-out test log-returns, normalized with the
        training statistics. It is `None` when there is no test split.
    """

    train_dataset: SlidingWindowReconstructionDataset
    validation_dataset: SlidingWindowReconstructionDataset
    full_dataset: SlidingWindowReconstructionDataset
    return_train_mean: float
    return_train_std: float
    test_dataset: SlidingWindowReconstructionDataset | None = None


@dataclass
# pylint: disable-next=too-many-instance-attributes
class LSTMAutoencoderTrainingResult:
    """Outcome of one call to `LSTMAutoencoderTrainer.train`.

    Parameters
    ----------
    model : LSTMAutoencoder
        Trained LSTM autoencoder.
    training_loss_history : list[float]
        Mean squared reconstruction error of the training set, one value
        per epoch.
    validation_loss : float
        Mean squared reconstruction error of the validation set.
    train_evaluation_loss : float
        Mean squared reconstruction error of the training set, computed
        after training with the same metric as `validation_loss`. Comparing
        the two shows whether the model overfits.
    anomaly_threshold : float
        Reconstruction score above which a window is flagged as an
        anomaly.
    return_train_mean : float
        Mean of the training log-returns used for normalization.
    return_train_std : float
        Standard deviation of the training log-returns used for
        normalization.
    model_file_path : Path
        Local file where the trained `state_dict` was saved.
    test_loss : float | None
        Mean squared reconstruction error of the held-out test set. It is
        `None` when there is no test split.
    baseline_test_loss : float | None
        Error of the naive zero prediction on the test set. The model
        should have a smaller `test_loss` than this value.
    window_size : int | None
        Number of log-returns in each window of the model.
    hidden_layer_size : int | None
        Number of units in the hidden state of the LSTM layers.
    latent_layer_size : int | None
        Size of the latent representation.
    """

    model: LSTMAutoencoder
    training_loss_history: list[float]
    validation_loss: float
    train_evaluation_loss: float
    anomaly_threshold: float
    return_train_mean: float
    return_train_std: float
    model_file_path: Path
    test_loss: float | None = None
    baseline_test_loss: float | None = None
    window_size: int | None = None
    hidden_layer_size: int | None = None
    latent_layer_size: int | None = None


def compute_reconstruction_scores(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    """Compute the mean squared reconstruction error of each window.

    The model is set to evaluation mode and no gradients are computed.

    Parameters
    ----------
    model : nn.Module
        Autoencoder that reconstructs windows of shape
        `(num_windows, window_size)`.
    input_tensor : torch.Tensor
        Tensor with the windows to score.

    Returns
    -------
    np.ndarray
        One-dimensional array of shape `(num_windows,)` with the mean
        squared error of each window.

    Example
    -------
    >>> window_tensor = torch.randn(7, 5)
    >>> compute_reconstruction_scores(model, window_tensor).shape
    (7,)
    """
    model.eval()
    with torch.no_grad():
        reconstructed_tensor = model(input_tensor)
        squared_error_tensor = (reconstructed_tensor - input_tensor) ** 2
        window_score_tensor = squared_error_tensor.mean(dim=1)

    return window_score_tensor.cpu().numpy()


def stack_dataset_windows(
    dataset: SlidingWindowReconstructionDataset,
) -> torch.Tensor:
    """Stack every window of a dataset into a single tensor.

    Parameters
    ----------
    dataset : SlidingWindowReconstructionDataset
        Dataset of sliding windows.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(num_windows, window_size)`, in the order of the
        dataset.
    """
    return torch.stack([dataset[index] for index in range(len(dataset))])


def compute_mean_baseline_loss(
    dataset: SlidingWindowReconstructionDataset,
) -> float:
    """Compute the error of a naive model that always predicts zero.

    The series is normalized with the training statistics, so zero is the
    training mean. A useful autoencoder must reconstruct the windows with
    a smaller mean squared error than this baseline.

    Parameters
    ----------
    dataset : SlidingWindowReconstructionDataset
        Normalized windows to evaluate.

    Returns
    -------
    float
        Mean of the squared values over all windows.

    Example
    -------
    >>> compute_mean_baseline_loss(prepared_data.test_dataset)
    0.98
    """
    window_tensor = stack_dataset_windows(dataset)

    return float((window_tensor**2).mean().item())


def build_ticker_slug(ticker: str) -> str:
    """Convert a ticker into a slug that is safe in file names.

    Parameters
    ----------
    ticker : str
        Ticker symbol, for example `"PETR4.SA"` or `"^BVSP"`.

    Returns
    -------
    str
        Lowercase ticker in which every character that is not a letter,
        a digit, `.`, `_` or `-` is replaced by `-`.

    Example
    -------
    >>> build_ticker_slug("PETR4.SA")
    'petr4.sa'
    >>> build_ticker_slug("^BVSP")
    '-bvsp'
    """
    return re.sub(r"[^\w.-]", "-", ticker.lower())


def _build_window_dataset(
    return_tensor: torch.Tensor,
    return_train_mean: float,
    return_train_std: float,
    window_size: int,
) -> SlidingWindowReconstructionDataset:
    """Normalize log-returns with the training statistics and window them.

    Parameters
    ----------
    return_tensor : torch.Tensor
        Log-returns in chronological order.
    return_train_mean : float
        Mean of the training log-returns, used for normalization.
    return_train_std : float
        Standard deviation of the training log-returns, used for
        normalization.
    window_size : int
        Number of log-returns in each window.

    Returns
    -------
    SlidingWindowReconstructionDataset
        Windows of the normalized log-returns.
    """
    normalized_tensor, _, _ = normalize_series(
        return_tensor,
        mean=return_train_mean,
        std=return_train_std,
    )

    return SlidingWindowReconstructionDataset(
        series_values=normalized_tensor,
        window_size=window_size,
    )


class LSTMAutoencoderTrainer:
    """Train an LSTM autoencoder that flags anomalous log-returns.

    The trainer covers the whole workflow for one ticker: it obtains the
    prices, builds the normalized windows from the `split` labels of the
    price data, trains the LSTM autoencoder, evaluates it on the
    validation and test splits, computes the anomaly threshold and
    registers the trained model in wandb.

    Parameters
    ----------
    ticker : str
        Ticker symbol to train on, for example `"AAPL"`.
    config : LSTMAutoencoderTrainingConfig
        Complete training configuration.
    experiment_tracker : WandbExperimentTracker, optional
        Tracker used to log the run. When `None`, a tracker is created
        from the `tracking` settings of `config`.
    price_data : pd.DataFrame, optional
        Prices already collected, with the `"close_price"` and `"split"`
        columns and, optionally, a `"log_return"` column, such as the
        ones saved by `collect_market_data`. When `None`, the prices are
        downloaded from yfinance and labeled with the split ratios of
        `config`.

    Attributes
    ----------
    ticker : str
        Ticker symbol being trained.
    config : LSTMAutoencoderTrainingConfig
        Training configuration.
    device : str
        Device used for training. It is `config.device` when defined,
        `"cuda"` when available and `"cpu"` otherwise.
    experiment_tracker : WandbExperimentTracker
        Tracker that receives the metrics and the model artifact.
    price_data : pd.DataFrame | None
        Prices given at construction, or `None` to download them.

    Example
    -------
    >>> trainer = LSTMAutoencoderTrainer(
    ...     ticker="AAPL",
    ...     config=LSTMAutoencoderTrainingConfig(),
    ... )
    >>> result = trainer.train()
    >>> result.model_file_path
    PosixPath('artifacts/lstm-autoencoder-aapl.pt')
    """

    def __init__(
        self,
        ticker: str,
        config: LSTMAutoencoderTrainingConfig,
        experiment_tracker: WandbExperimentTracker | None = None,
        price_data: pd.DataFrame | None = None,
    ) -> None:
        """Store the settings and prices, then prepare device and tracker."""
        self.ticker = ticker
        self.config = config
        self.price_data = price_data
        self.device = self._resolve_device()
        if experiment_tracker is None:
            experiment_tracker = self._build_experiment_tracker()
        self.experiment_tracker = experiment_tracker

    @property
    def model_name(self) -> str:
        """Return the name of the model artifact for this ticker.

        Returns
        -------
        str
            Name in the form `lstm-autoencoder-<ticker_slug>`.
        """
        return MODEL_NAME_PREFIX + build_ticker_slug(self.ticker)

    @property
    def model_file_path(self) -> Path:
        """Return the local path where the trained weights are saved.

        Returns
        -------
        Path
            Path in the form
            `<output_directory>/lstm-autoencoder-<ticker_slug>.pt`.
        """
        output_directory = Path(self.config.tracking.output_directory)

        return output_directory / f"{self.model_name}.pt"

    def load_price_data(self) -> pd.DataFrame:
        """Return the prices of the ticker with their split labels.

        Prices given at construction are used as they are, and their
        `split` column is never recomputed. Without them, the prices are
        downloaded and labeled with the split ratios of the configuration.

        Returns
        -------
        pd.DataFrame
            Data frame with the `"close_price"`, `"log_return"` and
            `"split"` columns. The `"log_return"` column may be absent
            when the prices were given at construction.

        Raises
        ------
        ValueError
            If the prices given at construction have no `"close_price"`
            or no `"split"` column.
        """
        if self.price_data is None:
            market_data_config = self.config.market_data
            raw_price_data = fetch_price_series(
                self.ticker,
                market_data_config.start_date,
                market_data_config.end_date,
            )

            return add_split_column(
                add_log_return_column(raw_price_data),
                market_data_config.train_split_ratio,
                market_data_config.validation_split_ratio,
            )

        for column_name in REQUIRED_PRICE_COLUMNS:
            if column_name not in self.price_data.columns:
                raise ValueError(
                    f"The price data of ticker '{self.ticker}' has no "
                    f"'{column_name}' column."
                )

        return self.price_data

    def prepare_data(self) -> PreparedLSTMAutoencoderData:
        """Obtain the prices and build the normalized training data.

        The log-returns are separated by the `split` labels of the price
        data, in chronological order. The training split is normalized
        with its own mean and standard deviation, and those statistics are
        reused for the other splits and for the complete series, to avoid
        data leakage.

        Returns
        -------
        PreparedLSTMAutoencoderData
            Datasets for training, validation, test and the complete
            series, together with the normalization statistics.

        Raises
        ------
        ValueError
            If the price data has no `"close_price"` or `"split"` column,
            if the number of split labels differs from the number of
            log-returns, or if a split has fewer values than
            `window_size`.
        """
        price_data = self.load_price_data()
        log_return_tensor = self._build_log_return_tensor(price_data)
        train_size, validation_size = self._count_split_sizes(
            price_data,
            len(log_return_tensor),
        )
        train_return_tensor = log_return_tensor[:train_size]
        validation_return_tensor = log_return_tensor[
            train_size : train_size + validation_size
        ]
        test_return_tensor = log_return_tensor[train_size + validation_size :]
        window_size = self.config.market_data.window_size
        self._check_split_lengths(
            len(train_return_tensor),
            len(validation_return_tensor),
            len(test_return_tensor),
        )

        _, return_train_mean, return_train_std = normalize_series(
            train_return_tensor
        )
        test_dataset = None
        if len(test_return_tensor) > 0:
            test_dataset = _build_window_dataset(
                test_return_tensor,
                return_train_mean,
                return_train_std,
                window_size,
            )

        return PreparedLSTMAutoencoderData(
            train_dataset=_build_window_dataset(
                train_return_tensor,
                return_train_mean,
                return_train_std,
                window_size,
            ),
            validation_dataset=_build_window_dataset(
                validation_return_tensor,
                return_train_mean,
                return_train_std,
                window_size,
            ),
            full_dataset=_build_window_dataset(
                log_return_tensor,
                return_train_mean,
                return_train_std,
                window_size,
            ),
            return_train_mean=return_train_mean,
            return_train_std=return_train_std,
            test_dataset=test_dataset,
        )

    def build_model(self) -> LSTMAutoencoder:
        """Create the LSTM autoencoder with the sizes of the config.

        Returns
        -------
        LSTMAutoencoder
            Untrained model, already moved to `device`.
        """
        architecture_config = self.config.architecture
        model = LSTMAutoencoder(
            window_size=self.config.market_data.window_size,
            hidden_dim=architecture_config.hidden_layer_size,
            latent_dim=architecture_config.latent_layer_size,
        )

        return model.to(self.device)

    def train(self) -> LSTMAutoencoderTrainingResult:
        """Run the full training workflow for the ticker.

        The wandb run is always finished, even when a step after the
        start of the run fails. The original exception is then
        propagated.

        Returns
        -------
        LSTMAutoencoderTrainingResult
            Trained model, loss history, evaluation losses, anomaly
            threshold, normalization statistics and the saved file path.
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
            train_evaluation_loss = self._evaluate(
                model,
                prepared_data.train_dataset,
            )
            validation_loss = self._evaluate(
                model,
                prepared_data.validation_dataset,
            )
            test_loss, baseline_test_loss = self._evaluate_test_split(
                model,
                prepared_data,
            )
            self.experiment_tracker.log_metrics(
                {
                    "train/loss_mse": train_evaluation_loss,
                    "validation/loss_mse": validation_loss,
                }
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
                    "LSTM autoencoder trained on the daily "
                    f"log-returns of {self.ticker}."
                ),
                metadata={
                    "final_train_loss": training_loss_history[-1],
                    "train_evaluation_loss": train_evaluation_loss,
                    "validation_loss": validation_loss,
                    "anomaly_threshold": anomaly_threshold,
                    "return_train_mean": prepared_data.return_train_mean,
                    "return_train_std": prepared_data.return_train_std,
                },
                aliases=[self.config.tracking.experiment_name],
            )
        finally:
            self.experiment_tracker.finish_run()

        return LSTMAutoencoderTrainingResult(
            model=model,
            training_loss_history=training_loss_history,
            validation_loss=validation_loss,
            train_evaluation_loss=train_evaluation_loss,
            anomaly_threshold=anomaly_threshold,
            return_train_mean=prepared_data.return_train_mean,
            return_train_std=prepared_data.return_train_std,
            model_file_path=self.model_file_path,
            test_loss=test_loss,
            baseline_test_loss=baseline_test_loss,
            window_size=self.config.market_data.window_size,
            hidden_layer_size=self.config.architecture.hidden_layer_size,
            latent_layer_size=self.config.architecture.latent_layer_size,
        )

    def _build_log_return_tensor(
        self,
        price_data: pd.DataFrame,
    ) -> torch.Tensor:
        """Return the log-returns of the prices, reusing stored ones.

        When the data frame has a `"log_return"` column, as the files of
        `collect_market_data` do, its values are used. Otherwise the
        log-returns are computed from the `"close_price"` column.

        Parameters
        ----------
        price_data : pd.DataFrame
            Prices with a `"close_price"` column and, optionally, a
            `"log_return"` column whose first row may be `NaN`.

        Returns
        -------
        torch.Tensor
            One-dimensional tensor of log-returns in chronological order.
        """
        if "log_return" in price_data.columns:
            log_return_values = price_data["log_return"].dropna().to_numpy()

            return torch.tensor(log_return_values, dtype=torch.float32)

        price_tensor = torch.tensor(
            price_data["close_price"].to_numpy(),
            dtype=torch.float32,
        )

        return torch.log(price_tensor[1:]) - torch.log(price_tensor[:-1])

    def _count_split_sizes(
        self,
        price_data: pd.DataFrame,
        return_count: int,
    ) -> tuple[int, int]:
        """Return the number of training and validation log-returns.

        The rows of the `"split"` column marked `"train"` and
        `"validation"` are counted. The log-returns after them are the
        test set. The split is never recomputed from ratios here.

        Parameters
        ----------
        price_data : pd.DataFrame
            Prices with a `"split"` column.
        return_count : int
            Number of log-returns in the series.

        Returns
        -------
        tuple[int, int]
            Number of log-returns for training and for validation. They
            are taken from the start of the series, in this order.

        Raises
        ------
        ValueError
            If the number of split labels differs from `return_count`.
        """
        split_values = price_data["split"].dropna()
        if len(split_values) != return_count:
            raise ValueError(
                f"The price data of ticker '{self.ticker}' has "
                f"{len(split_values)} split labels for {return_count} "
                "log-returns."
            )

        return (
            int((split_values == TRAIN_SPLIT_LABEL).sum()),
            int((split_values == VALIDATION_SPLIT_LABEL).sum()),
        )

    def _check_split_lengths(
        self,
        train_length: int,
        validation_length: int,
        test_length: int,
    ) -> None:
        """Check that every split has enough log-returns for one window.

        Parameters
        ----------
        train_length : int
            Number of training log-returns.
        validation_length : int
            Number of validation log-returns.
        test_length : int
            Number of test log-returns. Zero means there is no test split.

        Raises
        ------
        ValueError
            If the training or validation split has fewer values than
            `window_size`, or if a non-empty test split does.
        """
        window_size = self.config.market_data.window_size
        if train_length < window_size or validation_length < window_size:
            raise ValueError(
                f"Not enough data for ticker '{self.ticker}': the "
                "training and validation splits need at least "
                f"{window_size} log-returns each."
            )
        if 0 < test_length < window_size:
            raise ValueError(
                f"Not enough data for ticker '{self.ticker}': the test "
                f"split needs at least {window_size} log-returns."
            )

    def _evaluate_test_split(
        self,
        model: nn.Module,
        prepared_data: PreparedLSTMAutoencoderData,
    ) -> tuple[float | None, float | None]:
        """Compute the test error of the model and of the naive baseline.

        Parameters
        ----------
        model : nn.Module
            Trained autoencoder.
        prepared_data : PreparedLSTMAutoencoderData
            Data prepared by `prepare_data`, which may hold a test set.

        Returns
        -------
        tuple[float | None, float | None]
            Test error of the model and error of the naive zero prediction.
            Both are `None` when the data has no test set.
        """
        if prepared_data.test_dataset is None:
            return None, None

        test_loss = self._evaluate(model, prepared_data.test_dataset)
        baseline_test_loss = compute_mean_baseline_loss(
            prepared_data.test_dataset
        )

        return test_loss, baseline_test_loss

    def _resolve_device(self) -> str:
        """Return the configured device, or pick one automatically.

        Returns
        -------
        str
            `config.device` when it is defined, `"cuda"` when a GPU is
            available and `"cpu"` otherwise.
        """
        if self.config.device is not None:
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def _build_experiment_tracker(self) -> WandbExperimentTracker:
        """Create the wandb tracker for this ticker and configuration.

        Returns
        -------
        WandbExperimentTracker
            Tracker whose `config` contains the ticker, every
            hyperparameter and the names of the optimizer, the loss
            function and the model.
        """
        run_config = {
            "ticker": self.ticker,
            **self.config.to_flat_dict(),
            "optimizer": "Adam",
            "loss_function": "MSE",
            "model": "LSTMAutoencoder",
        }

        return WandbExperimentTracker(
            project_name=self.config.tracking.project_name,
            config=run_config,
            env_file_path=self.config.tracking.env_file_path,
        )

    def _fit(
        self,
        model: nn.Module,
        train_dataset: SlidingWindowReconstructionDataset,
    ) -> list[float]:
        """Train the model for the configured number of epochs.

        Each epoch logs its mean training loss, weighted by the size of
        every batch, to the experiment tracker.

        Parameters
        ----------
        model : nn.Module
            Autoencoder to train. It is updated in place.
        train_dataset : SlidingWindowReconstructionDataset
            Windows used for training.

        Returns
        -------
        list[float]
            Mean training loss of each epoch, in order.
        """
        optimization_config = self.config.optimization
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimization_config.learning_rate,
        )
        loss_function = nn.MSELoss()
        shuffle_generator = torch.Generator().manual_seed(
            optimization_config.random_seed
        )
        data_loader = DataLoader(
            train_dataset,
            batch_size=self.config.market_data.batch_size,
            shuffle=True,
            generator=shuffle_generator,
        )

        training_loss_history = []
        for epoch_index in range(optimization_config.number_of_epochs):
            model.train()
            epoch_loss_sum = 0.0
            for batch_tensor in data_loader:
                batch_tensor = batch_tensor.to(self.device)
                optimizer.zero_grad()
                reconstructed_tensor = model(batch_tensor)
                batch_loss = loss_function(reconstructed_tensor, batch_tensor)
                batch_loss.backward()
                optimizer.step()
                epoch_loss_sum += batch_loss.item() * len(batch_tensor)

            epoch_loss = epoch_loss_sum / len(train_dataset)
            training_loss_history.append(epoch_loss)
            self.experiment_tracker.log_metrics(
                {
                    "epoch": epoch_index + 1,
                    "train/loss_mse": epoch_loss,
                }
            )

        return training_loss_history

    def _evaluate(
        self,
        model: nn.Module,
        evaluation_dataset: SlidingWindowReconstructionDataset,
    ) -> float:
        """Compute the mean squared error on the windows of a dataset.

        The windows are not shuffled and no gradients are computed.

        Parameters
        ----------
        model : nn.Module
            Trained autoencoder.
        evaluation_dataset : SlidingWindowReconstructionDataset
            Windows used for the evaluation.

        Returns
        -------
        float
            Mean squared error over all values of all windows.
        """
        model.eval()
        squared_error_sum = 0.0
        loss_function = nn.MSELoss(reduction="sum")
        data_loader = DataLoader(
            evaluation_dataset,
            batch_size=self.config.market_data.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            for batch_tensor in data_loader:
                batch_tensor = batch_tensor.to(self.device)
                reconstructed_tensor = model(batch_tensor)
                squared_error_sum += loss_function(
                    reconstructed_tensor,
                    batch_tensor,
                ).item()

        window_size = self.config.market_data.window_size
        value_count = len(evaluation_dataset) * window_size

        return squared_error_sum / value_count

    def _compute_anomaly_threshold(
        self,
        model: nn.Module,
        full_dataset: SlidingWindowReconstructionDataset,
    ) -> float:
        """Compute the reconstruction score that flags anomalies.

        The threshold is the percentile
        `100 * (1 - contamination)` of the reconstruction scores of all
        windows of the complete series.

        Parameters
        ----------
        model : nn.Module
            Trained autoencoder.
        full_dataset : SlidingWindowReconstructionDataset
            Windows of the complete normalized series.

        Returns
        -------
        float
            Anomaly threshold, greater than zero for non-degenerate data.
        """
        window_tensor = stack_dataset_windows(full_dataset).to(self.device)
        reconstruction_scores = compute_reconstruction_scores(
            model,
            window_tensor,
        )
        percentile = 100 * (1 - self.config.contamination)

        return float(np.percentile(reconstruction_scores, percentile))
