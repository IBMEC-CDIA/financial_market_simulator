"""Unit tests for ModelRegistry."""

import pytest

from generative_models.serving.autoencoder_model_registry import (
    AutoencoderModelRegistry,
)
from generative_models.serving.model_registry import ModelRegistry


def test_constructor_starts_with_no_models() -> None:
    """Check that no model is kept in memory right after construction."""
    assert not ModelRegistry().models


@pytest.mark.parametrize(
    "method_name, args",
    [
        ("load_model", ("AAPL",)),
        ("load_all_models", ()),
        ("get_model", ("AAPL",)),
        ("list_loaded_tickers", ()),
    ],
)
def test_unimplemented_methods_raise_not_implemented_error(
    method_name: str,
    args: tuple,
) -> None:
    """Check that every skeleton method raises NotImplementedError."""
    registry = ModelRegistry()

    with pytest.raises(NotImplementedError):
        getattr(registry, method_name)(*args)


def test_autoencoder_registry_inherits_from_model_registry() -> None:
    """Check that the autoencoder registry reuses the base interface."""
    assert issubclass(AutoencoderModelRegistry, ModelRegistry)


def test_autoencoder_registry_overrides_every_base_method() -> None:
    """Check that no skeleton method is left unimplemented."""
    for method_name in (
        "load_model",
        "load_all_models",
        "get_model",
        "list_loaded_tickers",
    ):
        assert getattr(AutoencoderModelRegistry, method_name) is not getattr(
            ModelRegistry, method_name
        )
