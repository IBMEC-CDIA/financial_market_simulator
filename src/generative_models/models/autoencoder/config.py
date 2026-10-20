"""Configuration objects for training the simple autoencoder."""

from dataclasses import asdict, dataclass, field


@dataclass
class MarketDataConfig:
    """Settings for collecting and windowing the market data.

    Parameters
    ----------
    start_date : str
        Start date of the historical price range, in `"YYYY-MM-DD"`
        format.
    end_date : str
        End date of the historical price range, in `"YYYY-MM-DD"`
        format.
    train_split_ratio : float
        Fraction of the log-return series, in chronological order,
        used for training. The remainder is used for validation. Must
        be strictly between `0` and `1`.
    window_size : int
        Number of consecutive log-returns in each sliding window.
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
    """Settings for the autoencoder architecture.

    Parameters
    ----------
    hidden_layer_size : int
        Number of units in the hidden layers of the encoder and the
        decoder.
    latent_layer_size : int
        Dimensionality of the latent representation.
    """

    hidden_layer_size: int = 16
    latent_layer_size: int = 4


@dataclass
class OptimizationConfig:
    """Settings for the optimization loop.

    Parameters
    ----------
    number_of_epochs : int
        Number of full passes over the training windows.
    learning_rate : float
        Learning rate used by the Adam optimizer.
    random_seed : int
        Seed used for weight initialization and mini-batch shuffling.
    """

    number_of_epochs: int = 300
    learning_rate: float = 1e-3
    random_seed: int = 42


@dataclass
class TrackingConfig:
    """Settings for experiment tracking and model persistence.

    Parameters
    ----------
    project_name : str
        Name of the wandb project where the run is logged.
    output_directory : str
        Local directory where the trained model weights are saved.
    env_file_path : str
        Path to the `.env` file containing the `WANDB_API_KEY`
        variable.
    """

    project_name: str = "autoencoder-simple-outlier-trading-agent"
    output_directory: str = "artifacts"
    env_file_path: str = ".env"


@dataclass
class AutoencoderTrainingConfig:
    """Complete configuration for training the simple autoencoder.

    Groups every setting required by `AutoencoderTrainer`, except the
    ticker symbol, which is given to the trainer directly so that the
    same configuration can be reused across several tickers.

    Parameters
    ----------
    market_data : MarketDataConfig
        Data collection, splitting and windowing settings.
    architecture : AutoencoderArchitectureConfig
        Autoencoder layer sizes.
    optimization : OptimizationConfig
        Epochs, learning rate and random seed.
    tracking : TrackingConfig
        wandb project and model output settings.
    contamination : float
        Expected fraction of outlier windows, used to define the
        anomaly threshold as the `(1 - contamination)` percentile of
        the reconstruction errors. Must be strictly between `0` and
        `1`.
    device : str | None
        Torch device used for training (for example, `"cpu"` or
        `"cuda"`). When `None`, CUDA is used if available.

    Raises
    ------
    ValueError
        If `train_split_ratio` or `contamination` is outside the open
        interval `(0, 1)`, or if any size, epoch count or learning
        rate is not positive.

    Example
    -------
    >>> config = AutoencoderTrainingConfig(
    ...     market_data=MarketDataConfig(window_size=20),
    ...     optimization=OptimizationConfig(number_of_epochs=50),
    ... )
    >>> config.to_flat_dict()["window_size"]
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
        """Validate the configuration values."""
        if not 0.0 < self.market_data.train_split_ratio < 1.0:
            raise ValueError("train_split_ratio must be between 0 and 1.")
        if not 0.0 < self.contamination < 1.0:
            raise ValueError("contamination must be between 0 and 1.")

        positive_values = {
            "window_size": self.market_data.window_size,
            "batch_size": self.market_data.batch_size,
            "hidden_layer_size": self.architecture.hidden_layer_size,
            "latent_layer_size": self.architecture.latent_layer_size,
            "number_of_epochs": self.optimization.number_of_epochs,
            "learning_rate": self.optimization.learning_rate,
        }
        for value_name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{value_name} must be positive.")

    def to_flat_dict(self) -> dict[str, object]:
        """Flatten every setting into a single dictionary.

        Useful for experiment tracking, where a flat `config`
        dictionary is easier to filter and compare across runs.

        Returns
        -------
        dict[str, object]
            Mapping from each setting name to its value.
        """
        return {
            **asdict(self.market_data),
            **asdict(self.architecture),
            **asdict(self.optimization),
            **asdict(self.tracking),
            "contamination": self.contamination,
            "device": self.device,
        }
