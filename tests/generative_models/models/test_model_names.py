"""Unit tests for the names derived from a model package."""

import types

import pytest

from generative_models.models import model_names
from generative_models.models.model_names import (
    derive_experiment_name,
    load_artifact_prefix,
    load_project_name,
    validate_model_name,
)


def test_validate_model_name_accepts_a_package_name() -> None:
    """Check that a snake_case package name is accepted."""
    validate_model_name("convolutional_autoencoder")


@pytest.mark.parametrize("model_name", ["Base-Auto", "../models", "1model"])
def test_validate_model_name_refuses_other_names(model_name: str) -> None:
    """Check that names with paths, hyphens or capitals are refused."""
    with pytest.raises(ValueError, match="Invalid model name"):
        validate_model_name(model_name)


def test_derive_experiment_name_replaces_underscores_with_hyphens() -> None:
    """Check the experiment name of the convolutional model package."""
    assert (
        derive_experiment_name("convolutional_autoencoder")
        == "convolutional-autoencoder"
    )


def test_derive_experiment_name_refuses_a_name_wandb_would_not_accept() -> (
    None
):
    """Check that a model name too long for an alias is refused."""
    with pytest.raises(ValueError, match="at most 40 characters"):
        derive_experiment_name("a" * 41)


def test_load_artifact_prefix_reads_the_trainer_constant() -> None:
    """Check that the prefix comes from the trainer of the package."""
    assert load_artifact_prefix("base_autoencoder") == "simple-autoencoder-"


def test_load_artifact_prefix_refuses_a_package_without_trainer() -> None:
    """Check that a package with no trainer module cannot be served."""
    with pytest.raises(ModuleNotFoundError):
        load_artifact_prefix("autoencoder")


def test_load_artifact_prefix_refuses_a_trainer_without_prefix(
    monkeypatch,
) -> None:
    """Check that a trainer that does not define the prefix is refused."""
    fake_trainer = types.ModuleType("fake_trainer")
    monkeypatch.setattr(
        model_names.importlib,
        "import_module",
        lambda module_path: fake_trainer,
    )

    with pytest.raises(ValueError, match="MODEL_NAME_PREFIX"):
        load_artifact_prefix("fake_model")


def test_load_project_name_reads_the_trainer_constant() -> None:
    """Check that the base model keeps its own wandb project."""
    assert (
        load_project_name("base_autoencoder")
        == "autoencoder-simple-outlier-trading-agent"
    )


def test_load_project_name_refuses_a_trainer_without_project(
    monkeypatch,
) -> None:
    """Check that a trainer without a wandb project is refused."""
    fake_trainer = types.ModuleType("fake_trainer")
    monkeypatch.setattr(
        model_names.importlib,
        "import_module",
        lambda module_path: fake_trainer,
    )

    with pytest.raises(ValueError, match="WANDB_PROJECT_NAME"):
        load_project_name("fake_model")
