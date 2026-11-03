"""Evaluate buy and sell strategies that trade fractions of the portfolio."""

# pylint: disable=duplicate-code

import numpy as np
import pandas as pd

from generative_models.strategy.cooldown_portfolio import (
    BUY_AND_HOLD_NAME,
    compute_portfolio_metrics,
)


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def simulate_fraction_portfolio(
    dates: np.ndarray,
    price_values: np.ndarray,
    signal_values: np.ndarray,
    initial_stock_value: float,
    initial_cash: float,
    sell_fraction: float,
    buy_fraction: float,
) -> pd.DataFrame:
    """Simulate a portfolio that reacts to buy and sell signals.

    On a sell signal, the investor sells `sell_fraction` of the current
    stock position and moves the proceeds to cash. On a buy signal, the
    investor spends `buy_fraction` of the current cash to buy more stock.
    Hold signals do not change the portfolio. Transaction costs and taxes
    are not considered.

    Parameters
    ----------
    dates : np.ndarray
        Dates aligned with the prices and signals.
    price_values : np.ndarray
        Price of the asset on each date.
    signal_values : np.ndarray
        `"buy"`, `"sell"` or `"hold"` for each date.
    initial_stock_value : float
        Money invested in the asset at the first date.
    initial_cash : float
        Money available in cash at the first date.
    sell_fraction : float
        Fraction of the current stock position sold on each sell signal,
        between 0 and 1.
    buy_fraction : float
        Fraction of the current cash spent on each buy signal, between 0
        and 1.

    Returns
    -------
    pd.DataFrame
        Data frame indexed by date, with the columns `action`, `price`,
        `shares`, `stock_value`, `cash` and `total_value`.

    Example
    -------
    >>> simulated = simulate_fraction_portfolio(
    ...     dates=np.array(["2024-01-01", "2024-01-02"]),
    ...     price_values=np.array([100.0, 90.0]),
    ...     signal_values=np.array(["hold", "buy"]),
    ...     initial_stock_value=1000.0,
    ...     initial_cash=500.0,
    ...     sell_fraction=0.2,
    ...     buy_fraction=0.3,
    ... )
    >>> float(round(simulated["total_value"].iloc[-1], 2))
    1400.0
    """
    shares_held = initial_stock_value / price_values[0]
    cash_balance = initial_cash
    records = []

    for index, current_price in enumerate(price_values):
        current_signal = signal_values[index]
        action = "hold"

        if current_signal == "sell" and shares_held > 0:
            sold_shares = shares_held * sell_fraction
            shares_held -= sold_shares
            cash_balance += sold_shares * current_price
            action = "sell"
        elif current_signal == "buy" and cash_balance > 0:
            spent_cash = cash_balance * buy_fraction
            shares_held += spent_cash / current_price
            cash_balance -= spent_cash
            action = "buy"

        stock_value = shares_held * current_price
        records.append(
            {
                "action": action,
                "price": current_price,
                "shares": shares_held,
                "stock_value": stock_value,
                "cash": cash_balance,
                "total_value": stock_value + cash_balance,
            }
        )

    return pd.DataFrame(records, index=pd.DatetimeIndex(dates))


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def evaluate_fraction_strategies(
    dates: np.ndarray,
    price_values: np.ndarray,
    model_signals: dict[str, np.ndarray],
    initial_stock_value: float,
    initial_cash: float,
    sell_fraction: float,
    buy_fraction: float,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Run the fraction strategy for every model and for buy and hold.

    Parameters
    ----------
    dates : np.ndarray
        Dates aligned with the prices.
    price_values : np.ndarray
        Price of the asset on each date.
    model_signals : dict[str, np.ndarray]
        Mapping from each strategy name to its array of `"buy"`, `"sell"`
        or `"hold"` signals.
    initial_stock_value : float
        Money invested in the asset at the first date.
    initial_cash : float
        Money available in cash at the first date.
    sell_fraction : float
        Fraction of the stock position sold on each sell signal.
    buy_fraction : float
        Fraction of the cash spent on each buy signal.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], pd.DataFrame]
        The portfolio data frame of each strategy, keyed by name, and a
        table with one row of metrics per strategy, sorted by the final
        total value in descending order.

    Example
    -------
    >>> histories, table = evaluate_fraction_strategies(
    ...     dates, price_values, {"modelo": signals}, 10000.0, 5000.0,
    ...     0.2, 0.3,
    ... )
    """
    initial_total_value = initial_stock_value + initial_cash
    hold_signal_values = np.full(len(price_values), "hold", dtype=object)

    all_signals = dict(model_signals)
    all_signals[BUY_AND_HOLD_NAME] = hold_signal_values

    portfolio_histories = {}
    metrics_rows = []
    for strategy_name, signal_values in all_signals.items():
        portfolio_frame = simulate_fraction_portfolio(
            dates=dates,
            price_values=price_values,
            signal_values=signal_values,
            initial_stock_value=initial_stock_value,
            initial_cash=initial_cash,
            sell_fraction=sell_fraction,
            buy_fraction=buy_fraction,
        )
        portfolio_histories[strategy_name] = portfolio_frame
        metrics_rows.append(
            {
                "strategy": strategy_name,
                **compute_portfolio_metrics(
                    portfolio_frame,
                    initial_total_value,
                ),
            }
        )

    metrics_table = pd.DataFrame(metrics_rows).sort_values(
        "final_total_value",
        ascending=False,
    )

    return portfolio_histories, metrics_table
