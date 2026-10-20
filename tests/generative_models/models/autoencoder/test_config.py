"""Unit tests for the autoencoder training configuration."""

import pytest

from generative_models.models.autoencoder.config import (
    AutoencoderArchitectureConfig,
    AutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
)


def test_default_config_is_valid() -> None:
    """Check that the default configuration can be instantiated."""
    config = AutoencoderTrainingConfig()

    assert config.market_data.window_size == 10
    assert config.optimization.number_of_epochs == 300
    assert config.device is None


def test_to_flat_dict_merges_every_section() -> None:
    """Check that the flat dictionary exposes every nested setting."""
    config = AutoencoderTrainingConfig(
        market_data=MarketDataConfig(window_size=20),
        optimization=OptimizationConfig(learning_rate=0.01),
    )

    flat_config = config.to_flat_dict()

    assert flat_config["window_size"] == 20
    assert flat_config["learning_rate"] == 0.01
    assert flat_config["hidden_layer_size"] == 16
    assert flat_config["project_name"] == config.tracking.project_name
    assert flat_config["contamination"] == 0.05


@pytest.mark.parametrize("train_split_ratio", [0.0, 1.0, 1.5])
def test_invalid_train_split_ratio_raises(train_split_ratio: float) -> None:
    """Check that a split ratio outside (0, 1) is rejected."""
    with pytest.raises(ValueError, match="train_split_ratio"):
        AutoencoderTrainingConfig(
            market_data=MarketDataConfig(
                train_split_ratio=train_split_ratio,
            ),
        )


@pytest.mark.parametrize("contamination", [0.0, 1.0])
def test_invalid_contamination_raises(contamination: float) -> None:
    """Check that a contamination outside (0, 1) is rejected."""
    with pytest.raises(ValueError, match="contamination"):
        AutoencoderTrainingConfig(contamination=contamination)


def test_non_positive_layer_size_raises() -> None:
    """Check that a non-positive layer size is rejected."""
    with pytest.raises(ValueError, match="latent_layer_size"):
        AutoencoderTrainingConfig(
            architecture=AutoencoderArchitectureConfig(latent_layer_size=0),
        )
