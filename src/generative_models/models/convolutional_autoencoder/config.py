"""Typed configuration objects for training the CNN autoencoder."""

# pylint: disable=duplicate-code

from dataclasses import dataclass, field

from generative_models.models.base_autoencoder.config import (
    validate_experiment_name,
)


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
        training.
    validation_split_ratio : float
        Fraction of the log-returns, right after the training ones, used
        to monitor training. The remaining log-returns, at the end of
        the series, form the held-out test set that is only evaluated
        after training.
    window_size : int
        Number of consecutive log-returns in each training window.
    batch_size : int
        Number of windows per training batch.
    """

    start_date: str = "2020-01-01"
    end_date: str = "2026-09-15"
    train_split_ratio: float = 0.7
    validation_split_ratio: float = 0.15
    window_size: int = 10
    batch_size: int = 32


@dataclass
class ConvolutionalArchitectureConfig:
    """Sizes of the layers of the convolutional autoencoder.

    Parameters
    ----------
    hidden_channel_count : int
        Number of filters of each convolutional layer. It plays the
        role that the hidden layer size plays in a dense autoencoder,
        and it is reported as `hidden_layer_size`.
    latent_layer_size : int
        Size of the latent representation.
    """

    hidden_channel_count: int = 16
    latent_layer_size: int = 4


@dataclass
class OptimizationConfig:
    """Settings that control how the autoencoder is optimized.

    Parameters
    ----------
    number_of_epochs : int
        Largest number of passes over the training windows. Early
        stopping may end the training before this number is reached.
    learning_rate : float
        Step size used by the Adam optimizer.
    random_seed : int
        Seed used to make the weight initialization and the shuffling
        of the training batches reproducible.
    weight_decay : float
        L2 penalty of the Adam optimizer. It keeps the convolution
        filters small, which reduces the gap between the training and
        the validation errors. Zero disables the penalty.
    early_stopping_patience : int | None
        Number of epochs without an improvement of the validation
        error after which the training stops. The weights of the best
        epoch are restored. `None` trains for `number_of_epochs` and
        keeps the weights of the last epoch.
    """

    number_of_epochs: int = 300
    learning_rate: float = 1e-3
    random_seed: int = 42
    weight_decay: float = 1e-3
    early_stopping_patience: int | None = 20


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
    experiment_name : str
        Experiment the model is registered under, used as the wandb alias
        of the model. It must be a valid experiment name.
    """

    project_name: str = "autoencoder-convolutional-outlier-trading-agent"
    output_directory: str = "artifacts"
    env_file_path: str = ".env"
    experiment_name: str = "default"

    def __post_init__(self) -> None:
        """Validate the experiment name."""
        validate_experiment_name(self.experiment_name)


@dataclass
class ConvolutionalAutoencoderTrainingConfig:
    """Complete configuration of one CNN autoencoder training run.

    The ticker is not part of this configuration, so the same settings
    can be reused for several tickers.

    Parameters
    ----------
    market_data : MarketDataConfig
        Settings of the historical price data.
    architecture : ConvolutionalArchitectureConfig
        Sizes of the convolutional autoencoder layers.
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
    >>> training_config = ConvolutionalAutoencoderTrainingConfig(
    ...     market_data=MarketDataConfig(window_size=20),
    ...     optimization=OptimizationConfig(number_of_epochs=50),
    ... )
    >>> training_config.to_flat_dict()["window_size"]
    20
    """

    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    architecture: ConvolutionalArchitectureConfig = field(
        default_factory=ConvolutionalArchitectureConfig
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
            "hidden_channel_count": (
                self.architecture.hidden_channel_count
            ),
            "latent_layer_size": self.architecture.latent_layer_size,
            "number_of_epochs": self.optimization.number_of_epochs,
            "learning_rate": self.optimization.learning_rate,
        }
        for field_name, field_value in positive_values.items():
            if field_value <= 0:
                raise ValueError(f"{field_name} must be positive.")

        if self.optimization.weight_decay < 0:
            raise ValueError("weight_decay must not be negative.")

        patience = self.optimization.early_stopping_patience
        if patience is not None and patience <= 0:
            raise ValueError(
                "early_stopping_patience must be positive or None."
            )

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
            "hidden_channel_count": (
                self.architecture.hidden_channel_count
            ),
            "latent_layer_size": self.architecture.latent_layer_size,
            "number_of_epochs": self.optimization.number_of_epochs,
            "learning_rate": self.optimization.learning_rate,
            "random_seed": self.optimization.random_seed,
            "weight_decay": self.optimization.weight_decay,
            "early_stopping_patience": (
                self.optimization.early_stopping_patience
            ),
            "project_name": self.tracking.project_name,
            "experiment_name": self.tracking.experiment_name,
            "output_directory": self.tracking.output_directory,
            "env_file_path": self.tracking.env_file_path,
            "contamination": self.contamination,
            "device": self.device,
        }


def _check_open_unit_interval(
    field_name: str,
    field_value: float,
) -> None:
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
