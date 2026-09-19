"""Unit tests for TickerModelManager."""

from unittest.mock import patch

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

    assert not manager.models


@pytest.mark.parametrize(
    "method_name, args",
    [
        ("list_tickers", ()),
        ("get_date_range", ("AAPL",)),
        ("fetch_price_data", ("AAPL",)),
        ("train_model", ("AAPL",)),
        ("train_all_models", ()),
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


def test_train_all_models_trains_every_ticker_in_order() -> None:
    """Check that train_model runs once per ticker, in listed order."""
    manager = _build_manager()

    with patch.object(
        TickerModelManager, "list_tickers", return_value=["MSFT", "AAPL"]
    ), patch.object(TickerModelManager, "train_model") as mock_train_model:
        manager.train_all_models()

    assert [call.args[0] for call in mock_train_model.call_args_list] == [
        "MSFT",
        "AAPL",
    ]


def test_train_all_models_does_nothing_without_tickers() -> None:
    """Check that no training happens when there are no tickers."""
    manager = TickerModelManager(tickers={})

    with patch.object(
        TickerModelManager, "list_tickers", return_value=[]
    ), patch.object(TickerModelManager, "train_model") as mock_train_model:
        manager.train_all_models()

    mock_train_model.assert_not_called()


def test_train_all_models_trains_sequentially_not_in_parallel() -> None:
    """Check that a ticker only starts training after the previous ends."""
    manager = _build_manager()
    events = []

    def fake_train_model(ticker: str) -> None:
        events.append(f"start {ticker}")
        events.append(f"end {ticker}")

    with patch.object(
        TickerModelManager, "list_tickers", return_value=["AAPL", "MSFT"]
    ), patch.object(
        TickerModelManager, "train_model", side_effect=fake_train_model
    ):
        manager.train_all_models()

    assert events == ["start AAPL", "end AAPL", "start MSFT", "end MSFT"]


def test_train_all_models_stops_at_first_failure() -> None:
    """Check that an error propagates and later tickers are not trained."""
    manager = _build_manager()

    with patch.object(
        TickerModelManager, "list_tickers", return_value=["AAPL", "MSFT"]
    ), patch.object(
        TickerModelManager,
        "train_model",
        side_effect=RuntimeError("training failed"),
    ) as mock_train_model:
        with pytest.raises(RuntimeError, match="training failed"):
            manager.train_all_models()

    mock_train_model.assert_called_once_with("AAPL")


def test_train_all_models_returns_none() -> None:
    """Check that train_all_models has no return value."""
    manager = _build_manager()

    with patch.object(
        TickerModelManager, "list_tickers", return_value=["AAPL"]
    ), patch.object(TickerModelManager, "train_model"):
        assert manager.train_all_models() is None
