"""Unit tests for the CNN autoencoder training configuration."""

# pylint: disable=duplicate-code

import pytest

from generative_models.models.convolutional_autoencoder.config import (
    ConvolutionalArchitectureConfig,
    ConvolutionalAutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
    TrackingConfig,
)

EXPECTED_FLAT_KEYS = {
    "start_date",
    "end_date",
    "train_split_ratio",
    "window_size",
    "batch_size",
    "hidden_channel_count",
    "latent_layer_size",
    "number_of_epochs",
    "learning_rate",
    "random_seed",
    "weight_decay",
    "early_stopping_patience",
    "project_name",
    "output_directory",
    "env_file_path",
    "contamination",
    "device",
}


def test_default_configuration_is_created() -> None:
    """Check that the default configuration is valid and has defaults."""
    training_config = ConvolutionalAutoencoderTrainingConfig()

    assert training_config.market_data.window_size == 10
    assert training_config.architecture.hidden_channel_count == 16
    assert training_config.architecture.latent_layer_size == 4
    assert training_config.optimization.number_of_epochs == 300
    assert training_config.contamination == 0.05
    assert training_config.device is None


def test_default_project_name_is_specific_to_this_model() -> None:
    """Check that the wandb project is not the one of base_autoencoder."""
    training_config = ConvolutionalAutoencoderTrainingConfig()

    assert training_config.tracking.project_name == (
        "autoencoder-convolutional-outlier-trading-agent"
    )


def test_default_sub_configurations_are_not_shared() -> None:
    """Check that each instance gets its own MarketDataConfig object."""
    first_config = ConvolutionalAutoencoderTrainingConfig()
    second_config = ConvolutionalAutoencoderTrainingConfig()

    assert first_config.market_data is not second_config.market_data


def test_default_regularization_is_enabled() -> None:
    """Check the weight decay and the patience that limit overfitting."""
    optimization = ConvolutionalAutoencoderTrainingConfig().optimization

    assert optimization.weight_decay == pytest.approx(1e-3)
    assert optimization.early_stopping_patience == 20


def test_to_flat_dict_contains_exactly_the_expected_keys() -> None:
    """Check that the flat dictionary has the 17 documented keys."""
    flat_dictionary = (
        ConvolutionalAutoencoderTrainingConfig().to_flat_dict()
    )

    assert set(flat_dictionary.keys()) == EXPECTED_FLAT_KEYS


def test_to_flat_dict_uses_nested_values() -> None:
    """Check that the flat dictionary reflects the nested settings."""
    training_config = ConvolutionalAutoencoderTrainingConfig(
        market_data=MarketDataConfig(window_size=20),
        architecture=ConvolutionalArchitectureConfig(
            hidden_channel_count=32,
        ),
        optimization=OptimizationConfig(learning_rate=0.01),
        tracking=TrackingConfig(project_name="custom-project"),
        device="cpu",
    )

    flat_dictionary = training_config.to_flat_dict()

    assert flat_dictionary["window_size"] == 20
    assert flat_dictionary["hidden_channel_count"] == 32
    assert flat_dictionary["learning_rate"] == 0.01
    assert flat_dictionary["project_name"] == "custom-project"
    assert flat_dictionary["device"] == "cpu"


@pytest.mark.parametrize("train_split_ratio", [0.0, 1.0, 1.5])
def test_invalid_train_split_ratio_raises_value_error(
    train_split_ratio: float,
) -> None:
    """Check that train_split_ratio must be strictly between 0 and 1."""
    with pytest.raises(ValueError, match="train_split_ratio"):
        ConvolutionalAutoencoderTrainingConfig(
            market_data=MarketDataConfig(
                train_split_ratio=train_split_ratio,
            ),
        )


@pytest.mark.parametrize("contamination", [0.0, 1.0])
def test_invalid_contamination_raises_value_error(
    contamination: float,
) -> None:
    """Check that contamination must be strictly between 0 and 1."""
    with pytest.raises(ValueError, match="contamination"):
        ConvolutionalAutoencoderTrainingConfig(
            contamination=contamination,
        )


def test_zero_hidden_channel_count_raises_value_error() -> None:
    """Check that a zero channel count is rejected by name."""
    with pytest.raises(ValueError, match="hidden_channel_count"):
        ConvolutionalAutoencoderTrainingConfig(
            architecture=ConvolutionalArchitectureConfig(
                hidden_channel_count=0,
            ),
        )


def test_zero_latent_layer_size_raises_value_error() -> None:
    """Check that a zero latent size is rejected by name."""
    with pytest.raises(ValueError, match="latent_layer_size"):
        ConvolutionalAutoencoderTrainingConfig(
            architecture=ConvolutionalArchitectureConfig(
                latent_layer_size=0,
            ),
        )


def test_negative_weight_decay_raises_value_error() -> None:
    """Check that a negative L2 penalty is rejected by name."""
    with pytest.raises(ValueError, match="weight_decay"):
        ConvolutionalAutoencoderTrainingConfig(
            optimization=OptimizationConfig(weight_decay=-0.1),
        )


def test_zero_weight_decay_is_accepted() -> None:
    """Check that the L2 penalty can be switched off."""
    training_config = ConvolutionalAutoencoderTrainingConfig(
        optimization=OptimizationConfig(weight_decay=0.0),
    )

    assert training_config.optimization.weight_decay == 0.0


@pytest.mark.parametrize("patience", [0, -5])
def test_non_positive_patience_raises_value_error(patience: int) -> None:
    """Check that the patience must be positive when it is given."""
    with pytest.raises(ValueError, match="early_stopping_patience"):
        ConvolutionalAutoencoderTrainingConfig(
            optimization=OptimizationConfig(
                early_stopping_patience=patience,
            ),
        )


def test_patience_of_none_disables_early_stopping() -> None:
    """Check that None is accepted to train every epoch."""
    training_config = ConvolutionalAutoencoderTrainingConfig(
        optimization=OptimizationConfig(early_stopping_patience=None),
    )

    assert training_config.optimization.early_stopping_patience is None


def test_negative_learning_rate_raises_value_error() -> None:
    """Check that a negative learning rate is rejected by name."""
    with pytest.raises(ValueError, match="learning_rate"):
        ConvolutionalAutoencoderTrainingConfig(
            optimization=OptimizationConfig(learning_rate=-0.1),
        )
