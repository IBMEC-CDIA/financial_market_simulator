"""Training of one autoencoder per ticker symbol."""

from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from generative_models.common.normalization import normalize_series
from generative_models.data.datasets import SlidingWindowReconstructionDataset
from generative_models.data.market_data import fetch_price_series
from generative_models.models.autoencoder.simple_autoencoder import (
    SimpleAutoencoder,
)
from generative_models.models.ticker_model_manager import TickerModelManager
from generative_models.tracking.wandb_tracker import WandbExperimentTracker


@dataclass(frozen=True)
class AutoencoderTrainingConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters used to train the autoencoder of each ticker.

    Parameters
    ----------
    window_size : int
        Number of consecutive log-returns in each window.
    batch_size : int
        Number of windows per mini-batch.
    number_of_epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate of the Adam optimizer.
    hidden_layer_size : int
        Number of units in the hidden layers of the autoencoder.
    latent_layer_size : int
        Dimensionality of the latent representation.
    validation_fraction : float
        Fraction of the chronologically last log-returns reserved for
        validation.
    random_seed : int
        Seed used for the model initialization and the data loader.
    device : str
        Torch device where the model is trained (for example, `"cpu"`
        or `"cuda"`).
    """

    window_size: int = 10
    batch_size: int = 32
    number_of_epochs: int = 300
    learning_rate: float = 1e-3
    hidden_layer_size: int = 16
    latent_layer_size: int = 4
    validation_fraction: float = 0.2
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TickerTrainingResult:
    """Outputs of the training of the autoencoder of one ticker.

    Parameters
    ----------
    training_loss_history : list[float]
        Average training loss of each epoch.
    validation_loss : float
        Average reconstruction loss on the validation windows.
    normalization_statistics : tuple[float, float]
        Mean and standard deviation of the training log-returns, needed
        to normalize new data consistently with the training data.
    """

    training_loss_history: list[float]
    validation_loss: float
    normalization_statistics: tuple[float, float]


class AutoencoderTickerModelManager(TickerModelManager):
    """Train and store one dense autoencoder for each ticker symbol.

    The log-returns of each ticker are split chronologically into
    training and validation sets, normalized with the training
    statistics, turned into sliding windows and used to train a
    `SimpleAutoencoder` that reconstructs each window. Every training
    is tracked with `WandbExperimentTracker`.

    Parameters
    ----------
    tickers : dict[str, dict[str, str]]
        Dictionary mapping each ticker symbol to its `start_date` and
        `end_date`, both in the format `"YYYY-MM-DD"`.
    training_config : AutoencoderTrainingConfig, optional
        Hyperparameters used to train every autoencoder. Defaults to
        `AutoencoderTrainingConfig()`.
    project_name : str
        Name of the wandb project where the runs are logged.

    Example
    -------
    >>> manager = AutoencoderTickerModelManager(
    ...     tickers={
    ...         "AAPL": {
    ...             "start_date": "2020-01-01",
    ...             "end_date": "2026-01-01",
    ...         },
    ...     },
    ... )
    >>> manager.train_model("AAPL")
    >>> autoencoder = manager.get_model("AAPL")
    """

    def __init__(
        self,
        tickers: dict[str, dict[str, str]],
        training_config: AutoencoderTrainingConfig | None = None,
        project_name: str = "autoencoder-ticker-models",
    ) -> None:
        super().__init__(tickers=tickers)
        self.training_config = training_config or AutoencoderTrainingConfig()
        self.project_name = project_name
        self.training_results: dict[str, TickerTrainingResult] = {}

    def list_tickers(self) -> list[str]:
        """List every ticker symbol managed by this instance.

        Returns
        -------
        list[str]
            Ticker symbols, in the same order they were provided in
            the constructor.
        """
        return list(self.tickers)

    def get_date_range(self, ticker: str) -> tuple[str, str]:
        """Get the configured date range for a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol whose date range should be returned.

        Returns
        -------
        tuple[str, str]
            The `start_date` and `end_date` configured for `ticker`.
        """
        date_range = self.tickers[ticker]
        return date_range["start_date"], date_range["end_date"]

    def fetch_price_data(self, ticker: str) -> pd.DataFrame:
        """Fetch the historical closing prices for a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol whose price series should be fetched, using
            the date range configured in the constructor.

        Returns
        -------
        pandas.DataFrame
            Data frame indexed by date with the `"close_price"` column.
        """
        start_date, end_date = self.get_date_range(ticker)
        return fetch_price_series(ticker, start_date, end_date)

    def train_model(self, ticker: str) -> None:
        """Train and register the autoencoder for a ticker.

        The trained model is stored in `models` and its training
        outputs (loss history, validation loss and normalization
        statistics) in `training_results`, both keyed by ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol whose autoencoder should be trained.
        """
        config = self.training_config
        torch.manual_seed(config.random_seed)

        train_loader, validation_loader, normalization_statistics = (
            self._build_data_loaders(ticker)
        )
        autoencoder = SimpleAutoencoder(
            input_dim=config.window_size,
            hidden_dim=config.hidden_layer_size,
            latent_dim=config.latent_layer_size,
        ).to(config.device)

        experiment_tracker = WandbExperimentTracker(
            project_name=self.project_name,
            config={
                "ticker": ticker,
                **self.tickers[ticker],
                "window_size": config.window_size,
                "batch_size": config.batch_size,
                "epochs": config.number_of_epochs,
                "learning_rate": config.learning_rate,
                "hidden_layer_size": config.hidden_layer_size,
                "latent_layer_size": config.latent_layer_size,
                "validation_fraction": config.validation_fraction,
                "random_seed": config.random_seed,
                "optimizer": "Adam",
                "loss_function": "MSE",
                "model": "SimpleAutoencoder",
            },
        )
        experiment_tracker.start_run()

        training_loss_history = self._fit(
            autoencoder, train_loader, experiment_tracker
        )
        validation_loss = self._evaluate(autoencoder, validation_loader)
        experiment_tracker.log_metrics(
            {"validation/loss_mse": validation_loss}
        )

        experiment_tracker.log_model(
            model=autoencoder,
            model_name=f"simple-autoencoder-{ticker.lower()}",
            model_file_path=f"simple_autoencoder_{ticker}.pt",
            metadata={"hidden_layer_size": config.hidden_layer_size},
        )
        experiment_tracker.finish_run()

        self.models[ticker] = autoencoder
        self.training_results[ticker] = TickerTrainingResult(
            training_loss_history=training_loss_history,
            validation_loss=validation_loss,
            normalization_statistics=normalization_statistics,
        )

    def get_model(self, ticker: str) -> SimpleAutoencoder:
        """Get the trained autoencoder registered for a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol whose trained autoencoder should be returned.

        Returns
        -------
        SimpleAutoencoder
            The autoencoder trained for `ticker`.

        Raises
        ------
        KeyError
            If no model has been trained for `ticker` yet.
        """
        if ticker not in self.models:
            raise KeyError(
                f"No model trained for ticker '{ticker}'. "
                "Call train_model first."
            )
        return self.models[ticker]

    def _build_data_loaders(
        self,
        ticker: str,
    ) -> tuple[DataLoader, DataLoader, tuple[float, float]]:
        """Build the training and validation loaders of a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol whose log-returns should be windowed.

        Returns
        -------
        tuple[DataLoader, DataLoader, tuple[float, float]]
            The training loader, the validation loader and the mean and
            standard deviation of the training log-returns.
        """
        config = self.training_config

        price_data = self.fetch_price_data(ticker)
        price_tensor = torch.tensor(
            price_data["close_price"].to_numpy(), dtype=torch.float32
        )
        log_return_tensor = (
            torch.log(price_tensor[1:]) - torch.log(price_tensor[:-1])
        )

        train_split_index = int(
            len(log_return_tensor) * (1 - config.validation_fraction)
        )
        normalized_train_tensor, return_mean, return_std = normalize_series(
            log_return_tensor[:train_split_index]
        )
        normalized_validation_tensor, _, _ = normalize_series(
            log_return_tensor[train_split_index:],
            mean=return_mean,
            std=return_std,
        )

        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(config.random_seed)

        train_loader = DataLoader(
            SlidingWindowReconstructionDataset(
                series_values=normalized_train_tensor,
                window_size=config.window_size,
            ),
            batch_size=config.batch_size,
            shuffle=True,
            generator=dataloader_generator,
        )
        validation_loader = DataLoader(
            SlidingWindowReconstructionDataset(
                series_values=normalized_validation_tensor,
                window_size=config.window_size,
            ),
            batch_size=config.batch_size,
            shuffle=False,
        )

        return train_loader, validation_loader, (return_mean, return_std)

    def _fit(
        self,
        autoencoder: SimpleAutoencoder,
        train_loader: DataLoader,
        experiment_tracker: WandbExperimentTracker,
    ) -> list[float]:
        """Run the training loop and log the loss of every epoch.

        Parameters
        ----------
        autoencoder : SimpleAutoencoder
            Autoencoder whose weights are updated in place.
        train_loader : DataLoader
            Loader that yields mini-batches of training windows.
        experiment_tracker : WandbExperimentTracker
            Tracker with an active run, used to log the epoch losses.

        Returns
        -------
        list[float]
            Average training loss of each epoch.
        """
        config = self.training_config
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(
            autoencoder.parameters(), lr=config.learning_rate
        )

        training_loss_history = []
        epoch_progress_bar = tqdm(
            range(config.number_of_epochs), desc="Training", unit="epoch"
        )
        for epoch_index in epoch_progress_bar:
            autoencoder.train()
            epoch_loss_total = 0.0

            for batch_windows in train_loader:
                batch_windows = batch_windows.to(config.device)

                optimizer.zero_grad()
                reconstruction = autoencoder(batch_windows)
                loss_value = loss_function(reconstruction, batch_windows)
                loss_value.backward()
                optimizer.step()
                epoch_loss_total += loss_value.item() * batch_windows.shape[0]

            epoch_average_loss = epoch_loss_total / len(train_loader.dataset)
            training_loss_history.append(epoch_average_loss)

            experiment_tracker.log_metrics(
                {
                    "epoch": epoch_index + 1,
                    "train/loss_mse": epoch_average_loss,
                }
            )
            epoch_progress_bar.set_postfix(loss=f"{epoch_average_loss:.6f}")

        return training_loss_history

    def _evaluate(
        self,
        autoencoder: SimpleAutoencoder,
        data_loader: DataLoader,
    ) -> float:
        """Compute the average reconstruction MSE over a data loader.

        Parameters
        ----------
        autoencoder : SimpleAutoencoder
            Autoencoder to evaluate.
        data_loader : DataLoader
            Loader that yields mini-batches of windows.

        Returns
        -------
        float
            Average mean squared reconstruction error per window.
        """
        device = self.training_config.device
        loss_function = nn.MSELoss()
        autoencoder.eval()
        loss_total = 0.0

        with torch.no_grad():
            for batch_windows in data_loader:
                batch_windows = batch_windows.to(device)
                reconstruction = autoencoder(batch_windows)
                loss_value = loss_function(reconstruction, batch_windows)
                loss_total += loss_value.item() * batch_windows.shape[0]

        return loss_total / len(data_loader.dataset)
