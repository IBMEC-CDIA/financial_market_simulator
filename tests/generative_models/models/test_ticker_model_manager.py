"""Unit tests for TickerModelManager."""

import pytest

from generative_models.models.ticker_model_manager import TickerModelManager


def _build_manager() -> TickerModelManager:
    return TickerModelManager(
        tickers={
            "AAPL": {"start_date": "2020-01-01", "end_date": "2026-01-01"},
            "MSFT": {"start_date": "2021-01-01", "end_date": "2026-01-01"},
        }
    )


def test_constructor_stores_tickers_configuration() -> None:
    """Check that the tickers dictionary is stored unchanged."""
    tickers = {
        "AAPL": {"start_date": "2020-01-01", "end_date": "2026-01-01"},
    }

    manager = TickerModelManager(tickers=tickers)

    assert manager.tickers == tickers


def test_constructor_starts_with_no_trained_models() -> None:
    """Check that no model is registered right after construction."""
    manager = _build_manager()

    assert manager.models == {}


@pytest.mark.parametrize(
    "method_name, args",
    [
        ("list_tickers", ()),
        ("get_date_range", ("AAPL",)),
        ("fetch_price_data", ("AAPL",)),
        ("train_model", ("AAPL",)),
        ("get_model", ("AAPL",)),
    ],
)
def test_unimplemented_methods_raise_not_implemented_error(
    method_name: str,
    args: tuple,
) -> None:
    """Check that every skeleton method still raises NotImplementedError."""
    manager = _build_manager()

    with pytest.raises(NotImplementedError):
        getattr(manager, method_name)(*args)
