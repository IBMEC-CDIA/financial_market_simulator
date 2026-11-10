"""Factory that creates the LSTM autoencoder trainer for the ticker manager."""

# pylint: disable=duplicate-code

import pandas as pd

from generative_models.models.lstm_autoencoder.config import (
    LSTMAutoencoderArchitectureConfig,
    LSTMAutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
    TrackingConfig,
)
from generative_models.models.lstm_autoencoder.trainer import (
    LSTMAutoencoderTrainer,
)
from generative_models.models.training_options import TrainingOptions


def build_lstm_autoencoder_config(
    options: TrainingOptions | None,
) -> LSTMAutoencoderTrainingConfig:
    """Convert the shared training options into the LSTM configuration.

    Parameters
    ----------
    options : TrainingOptions | None
        Options shared by the models. When `None`, the default
        configuration of the LSTM autoencoder is returned.

    Returns
    -------
    LSTMAutoencoderTrainingConfig
        Configuration with the values of `options`.
    """
    if options is None:
        return LSTMAutoencoderTrainingConfig()

    return LSTMAutoencoderTrainingConfig(
        market_data=MarketDataConfig(
            start_date=options.start_date,
            end_date=options.end_date,
            window_size=options.window_size,
        ),
        architecture=LSTMAutoencoderArchitectureConfig(),
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
) -> LSTMAutoencoderTrainer:
    """Create an LSTM autoencoder trainer for a ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol the model is trained on.
    price_data : pd.DataFrame | None
        Prices already prepared with `build_price_dataset`, with the
        `close_price`, `log_return` and `split` columns. When `None`, the
        trainer downloads the prices itself.
    options : TrainingOptions | None
        Training options shared by the models. When `None`, the default
        configuration of the LSTM autoencoder is used.

    Returns
    -------
    LSTMAutoencoderTrainer
        Trainer ready to be run with its `train` method.

    Example
    -------
    >>> trainer = build_trainer("AAPL", price_data)
    >>> result = trainer.train()
    """
    return LSTMAutoencoderTrainer(
        ticker=ticker,
        config=build_lstm_autoencoder_config(options),
        price_data=price_data,
    )
