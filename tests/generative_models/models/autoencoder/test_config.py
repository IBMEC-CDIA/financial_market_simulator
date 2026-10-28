"""Unit tests for the autoencoder training configuration classes."""

import pytest

from generative_models.models.autoencoder.config import (
    AutoencoderArchitectureConfig,
    AutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
    TrackingConfig,
)

EXPECTED_FLAT_KEYS = {
    "batch_size",
    "contamination",
    "device",
    "end_date",
    "env_file_path",
    "hidden_layer_size",
    "latent_layer_size",
    "learning_rate",
    "number_of_epochs",
    "output_directory",
    "project_name",
    "random_seed",
    "start_date",
    "train_split_ratio",
    "window_size",
}


def test_default_configuration_uses_documented_values() -> None:
    """CA-01: the default configuration is built with the documented values."""
    config = AutoencoderTrainingConfig()

    assert config.market_data.start_date == "2020-01-01"
    assert config.market_data.end_date == "2026-09-15"
    assert config.market_data.train_split_ratio == 0.8
    assert config.market_data.window_size == 10
    assert config.market_data.batch_size == 32
    assert config.architecture.hidden_layer_size == 16
    assert config.architecture.latent_layer_size == 4
    assert config.optimization.number_of_epochs == 300
    assert config.optimization.learning_rate == 1e-3
    assert config.optimization.random_seed == 42
    assert config.tracking.project_name == (
        "autoencoder-simple-outlier-trading-agent"
    )
    assert config.tracking.output_directory == "artifacts"
    assert config.tracking.env_file_path == ".env"
    assert config.contamination == 0.05
    assert config.device is None


def test_default_sub_configurations_are_not_shared() -> None:
    """CA-02: two default instances do not share the MarketDataConfig."""
    first_config = AutoencoderTrainingConfig()
    second_config = AutoencoderTrainingConfig()

    assert first_config.market_data is not second_config.market_data


def test_to_flat_dict_contains_every_field() -> None:
    """CA-03: to_flat_dict holds exactly the fifteen documented keys."""
    config = AutoencoderTrainingConfig(
        market_data=MarketDataConfig(window_size=20),
        architecture=AutoencoderArchitectureConfig(hidden_layer_size=8),
        optimization=OptimizationConfig(number_of_epochs=5),
        tracking=TrackingConfig(output_directory="models"),
        device="cpu",
    )

    flat_config = config.to_flat_dict()

    assert set(flat_config) == EXPECTED_FLAT_KEYS
    assert flat_config["window_size"] == 20
    assert flat_config["hidden_layer_size"] == 8
    assert flat_config["number_of_epochs"] == 5
    assert flat_config["output_directory"] == "models"
    assert flat_config["device"] == "cpu"


@pytest.mark.parametrize("train_split_ratio", [0.0, 1.0, 1.5])
def test_invalid_train_split_ratio_raises_value_error(
    train_split_ratio: float,
) -> None:
    """CA-04: a train_split_ratio outside (0, 1) raises ValueError."""
    with pytest.raises(ValueError, match="train_split_ratio"):
        AutoencoderTrainingConfig(
            market_data=MarketDataConfig(
                train_split_ratio=train_split_ratio,
            ),
        )


@pytest.mark.parametrize("contamination", [0.0, 1.0])
def test_invalid_contamination_raises_value_error(
    contamination: float,
) -> None:
    """CA-05: a contamination outside (0, 1) raises ValueError."""
    with pytest.raises(ValueError, match="contamination"):
        AutoencoderTrainingConfig(contamination=contamination)


def test_non_positive_hidden_layer_size_raises_value_error() -> None:
    """CA-06: hidden_layer_size equal to 0 raises ValueError."""
    with pytest.raises(ValueError, match="hidden_layer_size"):
        AutoencoderTrainingConfig(
            architecture=AutoencoderArchitectureConfig(hidden_layer_size=0),
        )


def test_negative_learning_rate_raises_value_error() -> None:
    """CA-07: a negative learning_rate raises ValueError."""
    with pytest.raises(ValueError, match="learning_rate"):
        AutoencoderTrainingConfig(
            optimization=OptimizationConfig(learning_rate=-0.1),
        )
