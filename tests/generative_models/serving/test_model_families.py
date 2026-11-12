"""Unit tests for the choice of the model family in the serving layer."""

# pylint: disable=duplicate-code

import types

import numpy as np
import pytest
import torch
from torch import nn

from generative_models.models.base_autoencoder.autoencoder import (
    SimpleAutoencoder,
)
from generative_models.serving import model_families
from generative_models.serving.model_families import (
    build_predictor,
    find_missing_serving_keys,
    load_serving_module,
    resolve_architecture,
    resolve_model_family,
)

LEGACY_METADATA = {
    "ticker": "AAPL",
    "window_size": 10,
    "hidden_layer_size": 8,
    "latent_layer_size": 4,
    "return_train_mean": 0.001,
    "return_train_std": 0.02,
    "anomaly_threshold": 1.3,
}
CONVOLUTIONAL_METADATA = {
    "ticker": "NVDA",
    "window_size": 10,
    "return_train_mean": 0.001,
    "return_train_std": 0.02,
    "anomaly_threshold": 1.3,
    "model_family": "convolutional_autoencoder",
    "architecture": {
        "input_dim": 10,
        "hidden_channels": 16,
        "latent_dim": 4,
    },
}


def test_resolve_model_family_defaults_to_the_base_autoencoder() -> None:
    """Check that an artifact without a family is the base autoencoder."""
    assert resolve_model_family(LEGACY_METADATA) == "base_autoencoder"


def test_resolve_model_family_reads_the_stored_family() -> None:
    """Check that the family stored with the model is used."""
    assert (
        resolve_model_family(CONVOLUTIONAL_METADATA)
        == "convolutional_autoencoder"
    )


def test_resolve_architecture_uses_the_legacy_keys_of_the_base_model() -> None:
    """Check the architecture of a base autoencoder without a stored one."""
    assert resolve_architecture(LEGACY_METADATA) == {
        "input_dim": 10,
        "hidden_dim": 8,
        "latent_dim": 4,
    }


def test_resolve_architecture_uses_the_stored_arguments() -> None:
    """Check that the stored constructor arguments are returned as ints."""
    assert resolve_architecture(CONVOLUTIONAL_METADATA) == {
        "input_dim": 10,
        "hidden_channels": 16,
        "latent_dim": 4,
    }


def test_resolve_architecture_refuses_other_family_without_architecture() -> (
    None
):
    """Check that a non-base model cannot be rebuilt without its arguments."""
    metadata = {**CONVOLUTIONAL_METADATA}
    del metadata["architecture"]

    with pytest.raises(ValueError, match="convolutional_autoencoder"):
        resolve_architecture(metadata)


def test_find_missing_serving_keys_is_empty_for_a_complete_base_model() -> (
    None
):
    """Check that a complete base artifact can be served."""
    assert find_missing_serving_keys(LEGACY_METADATA) == []


def test_find_missing_serving_keys_lists_the_legacy_architecture_keys() -> (
    None
):
    """Check that a base artifact without its layer sizes is refused."""
    metadata = {**LEGACY_METADATA}
    del metadata["hidden_layer_size"]

    assert find_missing_serving_keys(metadata) == ["hidden_layer_size"]


def test_find_missing_serving_keys_accepts_a_stored_architecture() -> None:
    """Check that a complete model of another family can be served."""
    assert find_missing_serving_keys(CONVOLUTIONAL_METADATA) == []


def test_find_missing_serving_keys_asks_for_the_architecture() -> None:
    """Check that a model of another family must store its architecture."""
    metadata = {**CONVOLUTIONAL_METADATA}
    del metadata["architecture"]

    assert find_missing_serving_keys(metadata) == ["architecture"]


def test_find_missing_serving_keys_lists_missing_common_keys() -> None:
    """Check that the ticker and the training statistics are required."""
    missing_keys = find_missing_serving_keys(
        {"model_family": "gru_autoencoder"}
    )

    assert missing_keys == [
        "ticker",
        "window_size",
        "return_train_mean",
        "return_train_std",
        "anomaly_threshold",
        "architecture",
    ]


def test_load_serving_module_returns_the_base_hooks() -> None:
    """Check that the base autoencoder exposes the serving functions."""
    module = load_serving_module("base_autoencoder")

    assert callable(module.build_model)
    assert callable(module.score_windows)


def test_load_serving_module_refuses_an_invalid_name() -> None:
    """Check that a family name with path characters is refused."""
    with pytest.raises(ValueError, match="Invalid model name"):
        load_serving_module("../base_autoencoder")


def test_load_serving_module_refuses_a_family_without_serving() -> None:
    """Check that a family with no serving module cannot be served."""
    with pytest.raises(ModuleNotFoundError):
        load_serving_module("autoencoder")


def _base_state_dict() -> dict[str, torch.Tensor]:
    """Build the weights of an untrained base autoencoder."""
    return SimpleAutoencoder(
        input_dim=10,
        hidden_dim=8,
        latent_dim=4,
    ).state_dict()


def test_build_predictor_rebuilds_a_base_model_from_its_weights() -> None:
    """Check that a stored base model is scored after the round trip."""
    predictor = build_predictor(
        metadata=LEGACY_METADATA,
        state_dict=_base_state_dict(),
        device="cpu",
        artifact_name="simple-autoencoder-aapl:v1",
    )

    prediction = predictor.predict(np.linspace(-0.01, 0.01, 30))

    assert prediction.reconstruction_scores.shape == (21,)
    assert predictor.metadata.ticker == "AAPL"
    assert predictor.metadata.artifact_name == "simple-autoencoder-aapl:v1"


def test_build_predictor_uses_the_scorer_of_the_model_family(
    monkeypatch,
) -> None:
    """Check that the predictor scores with the family's own function."""
    scored_windows = []

    def fake_score_windows(_model: nn.Module, input_tensor: torch.Tensor):
        """Record the windows and return a constant score."""
        scored_windows.append(input_tensor.shape)

        return np.zeros(input_tensor.shape[0])

    fake_module = types.SimpleNamespace(
        build_model=lambda architecture: SimpleAutoencoder(
            input_dim=architecture["input_dim"],
            hidden_dim=architecture["hidden_dim"],
            latent_dim=architecture["latent_dim"],
        ),
        score_windows=fake_score_windows,
    )
    monkeypatch.setattr(
        model_families,
        "load_serving_module",
        lambda model_family: fake_module,
    )

    predictor = build_predictor(
        metadata=LEGACY_METADATA,
        state_dict=_base_state_dict(),
        device="cpu",
        artifact_name="simple-autoencoder-aapl:v1",
    )
    prediction = predictor.predict(np.linspace(-0.01, 0.01, 30))

    assert scored_windows
    assert np.all(prediction.reconstruction_scores == 0)
