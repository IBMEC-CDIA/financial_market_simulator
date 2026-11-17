"""Factory that creates the GRU autoencoder trainer for the ticker manager."""

# pylint: disable=duplicate-code

import pandas as pd

from generative_models.models.gru_autoencoder.config import (
    GRUArchitectureConfig,
    GRUAutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
    TrackingConfig,
)
from generative_models.models.gru_autoencoder.trainer import (
    GRUAutoencoderTrainer,
)
from generative_models.models.training_options import TrainingOptions


def build_autoencoder_config(
    options: TrainingOptions | None,
) -> GRUAutoencoderTrainingConfig:
    """Convert the shared training options into the autoencoder configuration.

    Parameters
    ----------
    options : TrainingOptions | None
        Options shared by the models. When `None`, the default configuration
        of the autoencoder is returned.

    Returns
    -------
    GRUAutoencoderTrainingConfig
        Configuration with the values of `options`.
    """
    if options is None:
        return GRUAutoencoderTrainingConfig()

    return GRUAutoencoderTrainingConfig(
        market_data=MarketDataConfig(
            start_date=options.start_date,
            end_date=options.end_date,
            window_size=options.window_size,
        ),
        architecture=GRUArchitectureConfig(),
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
) -> GRUAutoencoderTrainer:
    """Create a GRU autoencoder trainer for a ticker.

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
    GRUAutoencoderTrainer
        Trainer ready to be run with its `train` method.

    Example
    -------
    >>> trainer = build_trainer("AAPL", price_data)
    >>> result = trainer.train()
    """
    return GRUAutoencoderTrainer(
        ticker=ticker,
        config=build_autoencoder_config(options),
        price_data=price_data,
    )
