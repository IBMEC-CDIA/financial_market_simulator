"""Typed configuration objects for training the autoencoder."""

from dataclasses import dataclass, field


@dataclass
class MarketDataConfig:
    """Settings that define the historical price data used for training.

    Parameters
    ----------
    start_date : str
        First date of the historical range, in `"YYYY-MM-DD"` format.
    end_date : str
        Last date of the historical range, in `"YYYY-MM-DD"` format.
    train_split_ratio : float
        Fraction of the log-returns, in chronological order, used for
        training. The remaining values are used for validation.
    window_size : int
        Number of consecutive log-returns in each training window.
    batch_size : int
        Number of windows per training batch.
    """

    start_date: str = "2020-01-01"
    end_date: str = "2026-09-15"
    train_split_ratio: float = 0.8
    window_size: int = 10
    batch_size: int = 32


@dataclass
class AutoencoderArchitectureConfig:
    """Sizes of the layers of the autoencoder.

    Parameters
    ----------
    hidden_layer_size : int
        Number of units in the hidden layers.
    latent_layer_size : int
        Size of the latent representation.
    """

    hidden_layer_size: int = 16
    latent_layer_size: int = 4


@dataclass
class OptimizationConfig:
    """Settings that control how the autoencoder is optimized.

    Parameters
    ----------
    number_of_epochs : int
        Number of passes over the training windows.
    learning_rate : float
        Step size used by the Adam optimizer.
    random_seed : int
        Seed used to make the weight initialization and the shuffling
        of the training batches reproducible.
    """

    number_of_epochs: int = 300
    learning_rate: float = 1e-3
    random_seed: int = 42


@dataclass
class TrackingConfig:
    """Settings that control where and how the experiment is tracked.

    Parameters
    ----------
    project_name : str
        Name of the wandb project where the run is logged.
    output_directory : str
        Directory where the trained model weights are saved.
    env_file_path : str
        Path to the `.env` file that contains `WANDB_API_KEY`.
    """

    project_name: str = "autoencoder-simple-outlier-trading-agent"
    output_directory: str = "artifacts"
    env_file_path: str = ".env"


@dataclass
class AutoencoderTrainingConfig:
    """Complete configuration of one autoencoder training run.

    The ticker is not part of this configuration, so the same settings
    can be reused for several tickers.

    Parameters
    ----------
    market_data : MarketDataConfig
        Settings of the historical price data.
    architecture : AutoencoderArchitectureConfig
        Sizes of the autoencoder layers.
    optimization : OptimizationConfig
        Settings of the optimization loop.
    tracking : TrackingConfig
        Settings of the experiment tracking and of the saved model.
    contamination : float
        Expected fraction of anomalous windows. It defines the
        percentile of the reconstruction scores used as the anomaly
        threshold.
    device : str | None
        Device used for training, for example `"cpu"` or `"cuda"`. When
        `None`, the trainer chooses automatically.

    Raises
    ------
    ValueError
        If `train_split_ratio` or `contamination` is not strictly
        between 0 and 1, or if a size, epoch count or learning rate is
        not positive.

    Example
    -------
    >>> training_config = AutoencoderTrainingConfig(
    ...     market_data=MarketDataConfig(window_size=20),
    ...     optimization=OptimizationConfig(number_of_epochs=50),
    ... )
    >>> training_config.to_flat_dict()["window_size"]
    20
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
        """Validate the ranges and sizes of the configuration."""
        _check_open_unit_interval(
            "train_split_ratio",
            self.market_data.train_split_ratio,
        )
        _check_open_unit_interval("contamination", self.contamination)

        positive_values = {
            "window_size": self.market_data.window_size,
            "batch_size": self.market_data.batch_size,
            "hidden_layer_size": self.architecture.hidden_layer_size,
            "latent_layer_size": self.architecture.latent_layer_size,
            "number_of_epochs": self.optimization.number_of_epochs,
            "learning_rate": self.optimization.learning_rate,
        }
        for field_name, field_value in positive_values.items():
            if field_value <= 0:
                raise ValueError(f"{field_name} must be positive.")

    def to_flat_dict(self) -> dict[str, object]:
        """Flatten all settings into a single dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary whose keys are the field names of the nested
            configurations, plus `contamination` and `device`. It is
            used as the `config` of the wandb run.
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


def _check_open_unit_interval(field_name: str, field_value: float) -> None:
    """Raise a ValueError unless the value is strictly between 0 and 1.

    Parameters
    ----------
    field_name : str
        Name of the configuration field, used in the error message.
    field_value : float
        Value to check.

    Raises
    ------
    ValueError
        If `field_value` is not in the open interval `(0, 1)`.
    """
    if not 0 < field_value < 1:
        raise ValueError(f"{field_name} must be between 0 and 1.")
