"""LSTM autoencoder that flags anomalous windows of daily log-returns."""

from generative_models.models.lstm_autoencoder.autoencoder import (
    LSTMAutoencoder,
)
from generative_models.models.lstm_autoencoder.config import (
    LSTMAutoencoderArchitectureConfig,
    LSTMAutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
    TrackingConfig,
)
from generative_models.models.lstm_autoencoder.trainer import (
    LSTMAutoencoderTrainer,
    LSTMAutoencoderTrainingResult,
)

__all__ = [
    "LSTMAutoencoder",
    "LSTMAutoencoderArchitectureConfig",
    "LSTMAutoencoderTrainer",
    "LSTMAutoencoderTrainingConfig",
    "LSTMAutoencoderTrainingResult",
    "MarketDataConfig",
    "OptimizationConfig",
    "TrackingConfig",
]
