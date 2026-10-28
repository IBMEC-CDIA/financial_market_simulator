"""Typed configuration objects for training the simple autoencoder."""

from dataclasses import asdict, dataclass, field


@dataclass
class MarketDataConfig:
    """Settings for downloading and windowing the market data.

    Parameters
    ----------
    start_date : str
        First date of the price history, in `YYYY-MM-DD` format.
    end_date : str
        Last date of the price history, in `YYYY-MM-DD` format.
    train_split_ratio : float
        Fraction of the log-returns, taken chronologically, used for
        training. The remainder is used for validation.
    window_size : int
        Number of consecutive log-returns in each window.
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
    """Layer sizes of the simple autoencoder.

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
    """Settings of the optimization loop.

    Parameters
    ----------
    number_of_epochs : int
        Number of passes over the training windows.
    learning_rate : float
        Learning rate of the Adam optimizer.
    random_seed : int
        Seed used for weight initialization and batch shuffling.
    """

    number_of_epochs: int = 300
    learning_rate: float = 1e-3
    random_seed: int = 42


@dataclass
class TrackingConfig:
    """Settings of the experiment tracking and model output.

    Parameters
    ----------
    project_name : str
        Name of the wandb project where runs are logged.
    output_directory : str
        Directory where the trained weights are saved.
    env_file_path : str
        Path to the `.env` file containing `WANDB_API_KEY`.
    """

    project_name: str = "autoencoder-simple-outlier-trading-agent"
    output_directory: str = "artifacts"
    env_file_path: str = ".env"


@dataclass
class AutoencoderTrainingConfig:
    """Complete, validated configuration of an autoencoder training run.

    The values are validated when the object is created. The ticker is
    not part of the configuration, so the same configuration can be
    reused for several tickers.

    Parameters
    ----------
    market_data : MarketDataConfig
        Data download and windowing settings.
    architecture : AutoencoderArchitectureConfig
        Layer sizes of the autoencoder.
    optimization : OptimizationConfig
        Optimization loop settings.
    tracking : TrackingConfig
        Experiment tracking and output settings.
    contamination : float
        Expected fraction of anomalous windows, used to compute the
        anomaly threshold. Must be in the open interval `(0, 1)`.
    device : str | None
        Torch device used for training. `None` lets the trainer pick
        `"cuda"` when available and `"cpu"` otherwise.

    Raises
    ------
    ValueError
        If `train_split_ratio` or `contamination` is outside `(0, 1)`,
        or if a size, the number of epochs or the learning rate is not
        positive.

    Example
    -------
    >>> training_config = AutoencoderTrainingConfig(
    ...     market_data=MarketDataConfig(window_size=20),
    ...     optimization=OptimizationConfig(number_of_epochs=100),
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
        """Validate the configuration values."""
        if not 0 < self.market_data.train_split_ratio < 1:
            raise ValueError("train_split_ratio must be between 0 and 1.")
        if not 0 < self.contamination < 1:
            raise ValueError("contamination must be between 0 and 1.")

        positive_fields = {
            "window_size": self.market_data.window_size,
            "batch_size": self.market_data.batch_size,
            "hidden_layer_size": self.architecture.hidden_layer_size,
            "latent_layer_size": self.architecture.latent_layer_size,
            "number_of_epochs": self.optimization.number_of_epochs,
            "learning_rate": self.optimization.learning_rate,
        }
        for field_name, field_value in positive_fields.items():
            if field_value <= 0:
                raise ValueError(f"{field_name} must be positive.")

    def to_flat_dict(self) -> dict[str, object]:
        """Flatten the configuration into a single dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary with every field of the four sub-configurations
            plus `contamination` and `device`, keyed by field name, as
            expected by the wandb run `config`.
        """
        return {
            **asdict(self.market_data),
            **asdict(self.architecture),
            **asdict(self.optimization),
            **asdict(self.tracking),
            "contamination": self.contamination,
            "device": self.device,
        }
