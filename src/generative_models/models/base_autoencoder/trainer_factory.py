"""Factory that creates the autoencoder trainer for the ticker manager."""

import pandas as pd

from generative_models.models.base_autoencoder.config import (
    AutoencoderArchitectureConfig,
    AutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
    TrackingConfig,
)
from generative_models.models.base_autoencoder.trainer import (
    AutoencoderTrainer,
)
from generative_models.models.training_options import TrainingOptions


def build_autoencoder_config(
    options: TrainingOptions | None,
) -> AutoencoderTrainingConfig:
    """Convert the shared training options into the autoencoder configuration.

    Parameters
    ----------
    options : TrainingOptions | None
        Options shared by the models. When `None`, the default configuration
        of the autoencoder is returned.

    Returns
    -------
    AutoencoderTrainingConfig
        Configuration with the values of `options`.
    """
    if options is None:
        return AutoencoderTrainingConfig()

    return AutoencoderTrainingConfig(
        market_data=MarketDataConfig(
            start_date=options.start_date,
            end_date=options.end_date,
            window_size=options.window_size,
        ),
        architecture=AutoencoderArchitectureConfig(),
        optimization=OptimizationConfig(
            number_of_epochs=options.epochs,
            learning_rate=options.learning_rate,
        ),
        tracking=TrackingConfig(
            output_directory=options.output_directory,
        ),
        device=options.device,
    )


def build_trainer(
    ticker: str,
    price_data: pd.DataFrame | None = None,
    options: TrainingOptions | None = None,
) -> AutoencoderTrainer:
    """Create an autoencoder trainer for a ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol the model is trained on.
    price_data : pd.DataFrame | None
        Prices already prepared with `build_price_dataset`. When `None`, the
        trainer downloads the prices itself.
    options : TrainingOptions | None
        Training options shared by the models. When `None`, the default
        configuration of the autoencoder is used.

    Returns
    -------
    AutoencoderTrainer
        Trainer ready to be run with its `train` method.

    Example
    -------
    >>> trainer = build_trainer("AAPL", price_data)
    >>> result = trainer.train()
    """
    return AutoencoderTrainer(
        ticker=ticker,
        config=build_autoencoder_config(options),
        price_data=price_data,
    )
