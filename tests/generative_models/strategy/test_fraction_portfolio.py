"""Unit tests for the fraction-based portfolio strategy."""

# pylint: disable=duplicate-code

import numpy as np
import pandas as pd
import pytest

from generative_models.strategy.cooldown_portfolio import BUY_AND_HOLD_NAME
from generative_models.strategy.fraction_portfolio import (
    evaluate_fraction_strategies,
    simulate_fraction_portfolio,
)


def _dates(number_of_days: int) -> np.ndarray:
    """Build consecutive business dates."""
    return pd.bdate_range("2024-01-01", periods=number_of_days).to_numpy()


def test_sell_moves_the_sold_shares_to_cash() -> None:
    """Check that a sale of 20% of 10 shares at 100 adds 200 to cash."""
    simulated = simulate_fraction_portfolio(
        dates=_dates(2),
        price_values=np.array([100.0, 100.0]),
        signal_values=np.array(["hold", "sell"], dtype=object),
        initial_stock_value=1000.0,
        initial_cash=5000.0,
        sell_fraction=0.2,
        buy_fraction=0.3,
    )

    assert simulated["action"].tolist() == ["hold", "sell"]
    assert simulated["shares"].iloc[-1] == pytest.approx(8.0)
    assert simulated["cash"].iloc[-1] == pytest.approx(5200.0)
    assert simulated["total_value"].iloc[-1] == pytest.approx(6000.0)


def test_buy_spends_part_of_the_cash() -> None:
    """Check the total value after one buy at a lower price."""
    simulated = simulate_fraction_portfolio(
        dates=_dates(2),
        price_values=np.array([100.0, 90.0]),
        signal_values=np.array(["hold", "buy"], dtype=object),
        initial_stock_value=1000.0,
        initial_cash=500.0,
        sell_fraction=0.2,
        buy_fraction=0.3,
    )

    assert simulated["total_value"].iloc[-1] == pytest.approx(1400.0)


def test_evaluation_includes_buy_and_hold_and_sorts_by_value() -> None:
    """Check that buy and hold is compared and the table is sorted."""
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    signals = np.array(["buy", "hold", "hold", "hold", "hold"], dtype=object)

    histories, table = evaluate_fraction_strategies(
        dates=_dates(5),
        price_values=prices,
        model_signals={"modelo": signals},
        initial_stock_value=10000.0,
        initial_cash=5000.0,
        sell_fraction=0.2,
        buy_fraction=0.3,
    )

    assert set(histories) == {"modelo", BUY_AND_HOLD_NAME}
    assert table["final_total_value"].is_monotonic_decreasing
