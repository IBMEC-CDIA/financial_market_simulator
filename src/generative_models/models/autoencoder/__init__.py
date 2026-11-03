"""Alias of the base autoencoder package, kept for the old import path."""

# pylint: disable=duplicate-code

from generative_models.models.base_autoencoder import (
    AutoencoderArchitectureConfig,
    AutoencoderTrainer,
    AutoencoderTrainingConfig,
    AutoencoderTrainingResult,
    MarketDataConfig,
    OptimizationConfig,
    SimpleAutoencoder,
    TrackingConfig,
)

__all__ = [
    "AutoencoderArchitectureConfig",
    "AutoencoderTrainer",
    "AutoencoderTrainingConfig",
    "AutoencoderTrainingResult",
    "MarketDataConfig",
    "OptimizationConfig",
    "SimpleAutoencoder",
    "TrackingConfig",
]
