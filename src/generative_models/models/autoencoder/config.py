"""Typed configuration objects for the autoencoder training pipeline."""

from dataclasses import dataclass, field

POSITIVE_FIELD_NAMES = (
    "window_size",
    "batch_size",
    "hidden_layer_size",
    "latent_layer_size",
    "number_of_epochs",
    "learning_rate",
)


@dataclass
class MarketDataConfig:
    """Settings of the price series and of the sliding windows.

    Parameters
    ----------
    start_date : str
        Start date of the historical range, in `"YYYY-MM-DD"` format.
    end_date : str
        End date of the historical range, in `"YYYY-MM-DD"` format.
    train_split_ratio : float
        Fraction of the log-return series, taken chronologically from
        the beginning, used for training.
    window_size : int
        Number of consecutive log-returns in each window.
    batch_size : int
        Number of windows per mini-batch.
    """

    start_date: str = "2020-01-01"
    end_date: str = "2026-09-15"
    train_split_ratio: float = 0.8
    window_size: int = 10
    batch_size: int = 32


@dataclass
class AutoencoderArchitectureConfig:
    """Sizes of the hidden and latent layers of the autoencoder.

    Parameters
    ----------
    hidden_layer_size : int
        Number of units in the hidden layers of encoder and decoder.
    latent_layer_size : int
        Dimensionality of the latent representation.
    """

    hidden_layer_size: int = 16
    latent_layer_size: int = 4


@dataclass
class OptimizationConfig:
    """Settings of the optimization loop.

    Parameters
    ----------
    number_of_epochs : int
        Number of passes over the training windows.
    learning_rate : float
        Learning rate given to the Adam optimizer.
    random_seed : int
        Seed used to make the training run reproducible.
    """

    number_of_epochs: int = 300
    learning_rate: float = 1e-3
    random_seed: int = 42


@dataclass
class TrackingConfig:
    """Settings of experiment tracking and of the local artifacts.

    Parameters
    ----------
    project_name : str
        Name of the wandb project where the run is logged.
    output_directory : str
        Directory where the trained model weights are written.
    env_file_path : str
        Path to the `.env` file holding the `WANDB_API_KEY` variable.
    """

    project_name: str = "autoencoder-simple-outlier-trading-agent"
    output_directory: str = "artifacts"
    env_file_path: str = ".env"


@dataclass
class AutoencoderTrainingConfig:
    """Full configuration of one autoencoder training run.

    The ticker is not part of the configuration: it is passed straight
    to the trainer, so that the same configuration can be reused for
    several tickers.

    Parameters
    ----------
    market_data : MarketDataConfig
        Settings of the price series and of the sliding windows.
    architecture : AutoencoderArchitectureConfig
        Sizes of the hidden and latent layers.
    optimization : OptimizationConfig
        Settings of the optimization loop.
    tracking : TrackingConfig
        Settings of experiment tracking and of the local artifacts.
    contamination : float
        Expected fraction of outlier windows, which defines the
        percentile used as the anomaly threshold.
    device : str | None
        Device used for training. `None` means that the trainer
        decides automatically.

    Raises
    ------
    ValueError
        If `train_split_ratio` or `contamination` fall outside the
        open interval `(0, 1)`, or if any of the window, batch, layer,
        epoch or learning rate values is not positive.

    Example
    -------
    >>> config = AutoencoderTrainingConfig(
    ...     market_data=MarketDataConfig(window_size=5),
    ...     device="cpu",
    ... )
    >>> config.to_flat_dict()["window_size"]
    5
    """

    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    architecture: AutoencoderArchitectureConfig = field(
        default_factory=AutoencoderArchitectureConfig
    )
    optimization: OptimizationConfig = field(
        default_factory=OptimizationConfig
    )
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    contamination: float = 0.05
    device: str | None = None

    def __post_init__(self) -> None:
        """Validate every numeric field of the configuration.

        Raises
        ------
        ValueError
            If any value is outside the range documented for it.
        """
        if not 0 < self.market_data.train_split_ratio < 1:
            raise ValueError(
                "train_split_ratio must be between 0 and 1."
            )
        if not 0 < self.contamination < 1:
            raise ValueError("contamination must be between 0 and 1.")

        flat_config = self.to_flat_dict()
        for field_name in POSITIVE_FIELD_NAMES:
            if flat_config[field_name] <= 0:
                raise ValueError(f"{field_name} must be positive.")

    def to_flat_dict(self) -> dict[str, object]:
        """Flatten the configuration into a single dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary with the fields of the four sub-configurations
            plus `contamination` and `device`, keyed by field name.
            This is the shape expected by the `config` argument of
            `WandbExperimentTracker`.

        Example
        -------
        >>> sorted(AutoencoderTrainingConfig().to_flat_dict())[:3]
        ['batch_size', 'contamination', 'device']
        """
        return {
            "start_date": self.market_data.start_date,
            "end_date": self.market_data.end_date,
            "train_split_ratio": self.market_data.train_split_ratio,
            "window_size": self.market_data.window_size,
            "batch_size": self.market_data.batch_size,
            "hidden_layer_size": self.architecture.hidden_layer_size,
            "latent_layer_size": self.architecture.latent_layer_size,
            "number_of_epochs": self.optimization.number_of_epochs,
            "learning_rate": self.optimization.learning_rate,
            "random_seed": self.optimization.random_seed,
            "project_name": self.tracking.project_name,
            "output_directory": self.tracking.output_directory,
            "env_file_path": self.tracking.env_file_path,
            "contamination": self.contamination,
            "device": self.device,
        }
