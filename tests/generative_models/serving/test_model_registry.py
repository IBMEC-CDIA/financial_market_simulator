"""Unit tests for ModelRegistry."""

import pytest

from generative_models.serving.model_registry import ModelRegistry


def test_constructor_starts_with_no_models() -> None:
    """Check that no model is kept in memory right after construction."""
    registry = ModelRegistry()

    assert not registry.models


def test_each_instance_has_its_own_models_dictionary() -> None:
    """Check that the models dictionary is not shared between instances."""
    first_registry = ModelRegistry()
    second_registry = ModelRegistry()

    first_registry.models["AAPL"] = object()

    assert not second_registry.models


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


def test_subclass_can_implement_the_interface() -> None:
    """Check that a subclass reuses the base models dictionary."""

    class DictModelRegistry(ModelRegistry):
        """Registry that stores plain objects in the models dictionary."""

        def load_model(self, ticker: str) -> str:
            self.models[ticker] = f"model-{ticker}"
            return self.models[ticker]

        def load_all_models(self) -> list[str]:
            self.load_model("AAPL")
            return self.list_loaded_tickers()

        def get_model(self, ticker: str) -> str:
            return self.models[ticker]

        def list_loaded_tickers(self) -> list[str]:
            return list(self.models)

    registry = DictModelRegistry()

    assert registry.load_all_models() == ["AAPL"]
    assert registry.get_model("AAPL") == "model-AAPL"
