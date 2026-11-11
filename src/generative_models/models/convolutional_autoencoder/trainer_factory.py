"""Factory that creates the CNN autoencoder trainer for the manager."""

# pylint: disable=duplicate-code

import pandas as pd

from generative_models.models.convolutional_autoencoder.config import (
    ConvolutionalArchitectureConfig,
    ConvolutionalAutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
    TrackingConfig,
)
from generative_models.models.convolutional_autoencoder.trainer import (
    ConvolutionalAutoencoderTrainer,
)
from generative_models.models.training_options import TrainingOptions


def build_convolutional_config(
    options: TrainingOptions | None,
) -> ConvolutionalAutoencoderTrainingConfig:
    """Convert the shared training options into the model configuration.

    Parameters
    ----------
    options : TrainingOptions | None
        Options shared by the models. When `None`, the default
        configuration of the convolutional autoencoder is returned.

    Returns
    -------
    ConvolutionalAutoencoderTrainingConfig
        Configuration with the values of `options`.

    Example
    -------
    >>> config = build_convolutional_config(TrainingOptions(epochs=10))
    >>> config.optimization.number_of_epochs
    10
    """
    if options is None:
        return ConvolutionalAutoencoderTrainingConfig()

    return ConvolutionalAutoencoderTrainingConfig(
        market_data=MarketDataConfig(
            start_date=options.start_date,
            end_date=options.end_date,
            window_size=options.window_size,
        ),
        architecture=ConvolutionalArchitectureConfig(),
        optimization=OptimizationConfig(
            number_of_epochs=options.epochs,
            learning_rate=options.learning_rate,
        ),
        tracking=TrackingConfig(
            output_directory=options.output_directory,
            experiment_name=options.experiment_name,
        ),
        device=options.device,
    )


def build_trainer(
    ticker: str,
    price_data: pd.DataFrame | None = None,
    options: TrainingOptions | None = None,
) -> ConvolutionalAutoencoderTrainer:
    """Create a convolutional autoencoder trainer for a ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol the model is trained on.
    price_data : pd.DataFrame | None
        Prices already prepared with `build_price_dataset`. When `None`,
        the trainer downloads the prices itself.
    options : TrainingOptions | None
        Training options shared by the models. When `None`, the default
        configuration of the convolutional autoencoder is used.

    Returns
    -------
    ConvolutionalAutoencoderTrainer
        Trainer ready to be run with its `train` method.

    Example
    -------
    >>> trainer = build_trainer("AAPL", price_data)
    >>> result = trainer.train()
    """
    return ConvolutionalAutoencoderTrainer(
        ticker=ticker,
        config=build_convolutional_config(options),
        price_data=price_data,
    )
