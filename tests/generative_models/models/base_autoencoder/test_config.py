"""Unit tests for the autoencoder training configuration."""

import pytest

from generative_models.models.base_autoencoder.config import (
    AutoencoderArchitectureConfig,
    AutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
    TrackingConfig,
    validate_experiment_name,
)

EXPECTED_FLAT_KEYS = {
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
    "experiment_name",
    "contamination",
    "device",
}


def test_default_configuration_is_created() -> None:
    """Check that the default configuration is valid and has defaults."""
    training_config = AutoencoderTrainingConfig()

    assert training_config.market_data.window_size == 10
    assert training_config.architecture.hidden_layer_size == 16
    assert training_config.optimization.number_of_epochs == 300
    assert training_config.contamination == 0.05
    assert training_config.device is None


def test_default_sub_configurations_are_not_shared() -> None:
    """Check that each instance gets its own MarketDataConfig object."""
    first_config = AutoencoderTrainingConfig()
    second_config = AutoencoderTrainingConfig()

    assert first_config.market_data is not second_config.market_data


def test_to_flat_dict_contains_exactly_the_expected_keys() -> None:
    """Check that the flat dictionary has the 16 documented keys."""
    flat_dictionary = AutoencoderTrainingConfig().to_flat_dict()

    assert set(flat_dictionary.keys()) == EXPECTED_FLAT_KEYS


def test_to_flat_dict_uses_nested_values() -> None:
    """Check that the flat dictionary reflects the nested settings."""
    training_config = AutoencoderTrainingConfig(
        market_data=MarketDataConfig(window_size=20),
        optimization=OptimizationConfig(learning_rate=0.01),
        tracking=TrackingConfig(project_name="custom-project"),
        device="cpu",
    )

    flat_dictionary = training_config.to_flat_dict()

    assert flat_dictionary["window_size"] == 20
    assert flat_dictionary["learning_rate"] == 0.01
    assert flat_dictionary["project_name"] == "custom-project"
    assert flat_dictionary["device"] == "cpu"


@pytest.mark.parametrize("train_split_ratio", [0.0, 1.0, 1.5])
def test_invalid_train_split_ratio_raises_value_error(
    train_split_ratio: float,
) -> None:
    """Check that train_split_ratio must be strictly between 0 and 1."""
    with pytest.raises(ValueError, match="train_split_ratio"):
        AutoencoderTrainingConfig(
            market_data=MarketDataConfig(train_split_ratio=train_split_ratio),
        )


@pytest.mark.parametrize("contamination", [0.0, 1.0])
def test_invalid_contamination_raises_value_error(
    contamination: float,
) -> None:
    """Check that contamination must be strictly between 0 and 1."""
    with pytest.raises(ValueError, match="contamination"):
        AutoencoderTrainingConfig(contamination=contamination)


def test_zero_hidden_layer_size_raises_value_error() -> None:
    """Check that a zero hidden layer size is rejected by name."""
    with pytest.raises(ValueError, match="hidden_layer_size"):
        AutoencoderTrainingConfig(
            architecture=AutoencoderArchitectureConfig(hidden_layer_size=0),
        )


def test_negative_learning_rate_raises_value_error() -> None:
    """Check that a negative learning rate is rejected by name."""
    with pytest.raises(ValueError, match="learning_rate"):
        AutoencoderTrainingConfig(
            optimization=OptimizationConfig(learning_rate=-0.1),
        )


def test_default_experiment_name_is_default() -> None:
    """Check that a tracking configuration starts in the default experiment."""
    assert TrackingConfig().experiment_name == "default"


@pytest.mark.parametrize(
    "experiment_name",
    ["aapl-baseline", "v2", "0-smoke", "a" * 40],
)
def test_valid_experiment_names_are_accepted(experiment_name: str) -> None:
    """Check that lowercase names with digits and hyphens are accepted."""
    assert validate_experiment_name(experiment_name) == experiment_name


@pytest.mark.parametrize(
    "experiment_name",
    ["", "Baseline", "-start", "has space", "under_score", "a" * 41],
)
def test_invalid_experiment_names_are_rejected(experiment_name: str) -> None:
    """Check that names outside the allowed format are rejected."""
    with pytest.raises(ValueError, match="experiment_name must start"):
        validate_experiment_name(experiment_name)


def test_latest_is_reserved_for_the_latest_model() -> None:
    """Check that the experiment name cannot be the alias latest."""
    with pytest.raises(ValueError, match="cannot be 'latest'"):
        TrackingConfig(experiment_name="latest")
