"""Unit tests for the cooldown portfolio strategy evaluator."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from generative_models.strategy.cooldown_portfolio import (
    BUY_AND_HOLD_NAME,
    compute_portfolio_metrics,
    evaluate_cooldown_strategies,
    run_strategy_evaluation,
    simulate_cooldown_portfolio,
)


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _simulate(
    prices: list[float],
    signals: list[str],
    stock_value: float = 1000.0,
    cash: float = 5000.0,
    sell_fraction: float = 0.5,
    buy_fraction: float = 0.25,
    cooldown_days: int = 0,
    cost_rate: float = 0.0,
) -> pd.DataFrame:
    """Run one simulation with simple, easy to check values."""
    dates = pd.bdate_range("2024-01-01", periods=len(prices)).to_numpy()

    return simulate_cooldown_portfolio(
        dates=dates,
        price_values=np.array(prices, dtype=float),
        signal_values=np.array(signals, dtype=object),
        initial_stock_value=stock_value,
        initial_cash=cash,
        sell_fraction=sell_fraction,
        buy_fraction=buy_fraction,
        cooldown_days=cooldown_days,
        transaction_cost_rate=cost_rate,
        daily_cash_rate=0.0,
    )


def test_buy_signal_spends_part_of_the_cash() -> None:
    """Check the total value after one buy at a lower price."""
    simulated = _simulate(
        prices=[100.0, 90.0],
        signals=["hold", "buy"],
        cash=500.0,
        buy_fraction=0.25,
    )

    assert simulated["action"].tolist() == ["hold", "buy"]
    assert simulated["total_value"].iloc[-1] == pytest.approx(1400.0)


def test_cooldown_ignores_signals_during_the_waiting_days() -> None:
    """Check that a sell during the cooldown is not executed."""
    simulated = _simulate(
        prices=[100.0, 100.0, 100.0],
        signals=["sell", "sell", "sell"],
        cooldown_days=2,
    )

    assert simulated["action"].tolist() == ["sell", "hold", "sell"]
    assert simulated["shares"].iloc[-1] == pytest.approx(2.5)


def test_transaction_cost_reduces_the_bought_shares() -> None:
    """Check that the cost is paid out of the amount spent on shares."""
    simulated = _simulate(
        prices=[100.0, 100.0],
        signals=["hold", "buy"],
        stock_value=1000.0,
        cash=5000.0,
        buy_fraction=0.25,
        cost_rate=0.01,
    )

    assert simulated["shares"].iloc[-1] == pytest.approx(10.0 + 12.375)
    assert simulated["cash"].iloc[-1] == pytest.approx(3750.0)


def test_sell_without_shares_is_a_hold() -> None:
    """Check that no sale happens when the position is empty."""
    simulated = _simulate(
        prices=[100.0, 100.0],
        signals=["sell", "sell"],
        stock_value=0.0,
    )

    assert simulated["action"].tolist() == ["hold", "hold"]


def test_metrics_report_return_drawdown_and_trade_counts() -> None:
    """Check the return, the drawdown and the counts of the metrics."""
    frame = pd.DataFrame(
        {
            "action": ["buy", "hold", "sell", "hold"],
            "total_value": [100.0, 120.0, 90.0, 110.0],
        }
    )

    metrics = compute_portfolio_metrics(frame, initial_total_value=100.0)

    assert metrics["final_total_value"] == pytest.approx(110.0)
    assert metrics["total_return_percent"] == pytest.approx(10.0)
    assert metrics["max_drawdown_percent"] == pytest.approx(-25.0)
    assert metrics["number_of_buys"] == 1
    assert metrics["number_of_sells"] == 1


def test_evaluation_includes_buy_and_hold_and_sorts_by_value() -> None:
    """Check that buy and hold is compared and the table is sorted."""
    dates = pd.bdate_range("2024-01-01", periods=5).to_numpy()
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    signals = np.array(["buy", "hold", "hold", "hold", "hold"], dtype=object)

    histories, table = evaluate_cooldown_strategies(
        dates=dates,
        price_values=prices,
        model_signals={"modelo": signals},
    )

    assert set(histories) == {"modelo", BUY_AND_HOLD_NAME}
    assert table["final_total_value"].is_monotonic_decreasing


@patch("generative_models.strategy.cooldown_portfolio.plt.show")
def test_run_evaluation_returns_the_metrics_table(mock_show) -> None:
    """Check that the evaluation returns the table and shows the plot."""
    dates = pd.bdate_range("2024-01-01", periods=3).to_numpy()
    prices = np.array([100.0, 100.0, 100.0])
    signals = np.array(["hold", "hold", "hold"], dtype=object)

    table = run_strategy_evaluation(dates, prices, {"modelo": signals})

    assert len(table) == 2
    mock_show.assert_called_once()
