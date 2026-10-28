"""Unit tests for the autoencoder training configuration."""

import pytest

from generative_models.models.autoencoder.config import (
    AutoencoderArchitectureConfig,
    AutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
    TrackingConfig,
)


def test_default_config_uses_default_values() -> None:
    """CA-01: the default config is created with every default value."""
    config = AutoencoderTrainingConfig()

    assert config.market_data == MarketDataConfig(
        start_date="2020-01-01",
        end_date="2026-09-15",
        train_split_ratio=0.8,
        window_size=10,
        batch_size=32,
    )
    assert config.architecture == AutoencoderArchitectureConfig(
        hidden_layer_size=16,
        latent_layer_size=4,
    )
    assert config.optimization == OptimizationConfig(
        number_of_epochs=300,
        learning_rate=1e-3,
        random_seed=42,
    )
    assert config.tracking == TrackingConfig(
        project_name="autoencoder-simple-outlier-trading-agent",
        output_directory="artifacts",
        env_file_path=".env",
    )
    assert config.contamination == 0.05
    assert config.device is None


def test_default_configs_do_not_share_sub_configs() -> None:
    """CA-02: default instances use distinct MarketDataConfig objects."""
    first_config = AutoencoderTrainingConfig()
    second_config = AutoencoderTrainingConfig()

    assert first_config.market_data is not second_config.market_data


def test_to_flat_dict_has_all_keys_and_values() -> None:
    """CA-03: to_flat_dict exposes the 15 fields with their values."""
    config = AutoencoderTrainingConfig(
        market_data=MarketDataConfig(window_size=20),
    )

    flat_config = config.to_flat_dict()

    assert set(flat_config.keys()) == {
        "start_date",
        "end_date",
        "train_split_ratio",
        "window_size",
        "batch_size",
        "hidden_layer_size",
        "latent_layer_size",
        "number_of_epochs",
        "learning_rate",
        "random_seed",
        "project_name",
        "output_directory",
        "env_file_path",
        "contamination",
        "device",
    }
    assert flat_config["window_size"] == 20


@pytest.mark.parametrize("train_split_ratio", [0.0, 1.0, 1.5])
def test_invalid_train_split_ratio_raises(train_split_ratio: float) -> None:
    """CA-04: train_split_ratio outside (0, 1) raises ValueError."""
    with pytest.raises(ValueError, match="train_split_ratio"):
        AutoencoderTrainingConfig(
            market_data=MarketDataConfig(train_split_ratio=train_split_ratio),
        )


@pytest.mark.parametrize("contamination", [0.0, 1.0])
def test_invalid_contamination_raises(contamination: float) -> None:
    """CA-05: contamination outside (0, 1) raises ValueError."""
    with pytest.raises(ValueError, match="contamination"):
        AutoencoderTrainingConfig(contamination=contamination)


def test_non_positive_hidden_layer_size_raises() -> None:
    """CA-06: hidden_layer_size=0 raises ValueError naming the field."""
    with pytest.raises(ValueError, match="hidden_layer_size"):
        AutoencoderTrainingConfig(
            architecture=AutoencoderArchitectureConfig(hidden_layer_size=0),
        )


def test_negative_learning_rate_raises() -> None:
    """CA-07: learning_rate=-0.1 raises ValueError naming the field."""
    with pytest.raises(ValueError, match="learning_rate"):
        AutoencoderTrainingConfig(
            optimization=OptimizationConfig(learning_rate=-0.1),
        )
