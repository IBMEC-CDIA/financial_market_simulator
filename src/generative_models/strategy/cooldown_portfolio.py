"""Evaluate trading strategies with cooldown, costs and cash yield."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INITIAL_STOCK_VALUE = 10000.0
INITIAL_CASH = 5000.0
SELL_FRACTION = 0.10
BUY_FRACTION = 0.25
COOLDOWN_DAYS = 10
TRANSACTION_COST_RATE = 0.001
CASH_ANNUAL_RATE = 0.10
TRADING_DAYS_PER_YEAR = 252
TICKER = "AAPL"
BUY_AND_HOLD_NAME = "Comprar e manter"


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def simulate_cooldown_portfolio(
    dates: np.ndarray,
    price_values: np.ndarray,
    signal_values: np.ndarray,
    initial_stock_value: float,
    initial_cash: float,
    sell_fraction: float,
    buy_fraction: float,
    cooldown_days: int,
    transaction_cost_rate: float,
    daily_cash_rate: float,
) -> pd.DataFrame:
    """Simulate a portfolio with cooldown, transaction costs and cash yield.

    Each day, the cash balance grows by `daily_cash_rate`. On a sell
    signal, the investor sells `sell_fraction` of the stock position. On
    a buy signal, the investor spends `buy_fraction` of the cash. Every
    trade pays `transaction_cost_rate` on the traded value. After a
    trade, no new trade is made for `cooldown_days` days, even if new
    signals appear. Signals received during the cooldown are ignored.

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
        Fraction of the stock position sold on each sell signal, between
        0 and 1.
    buy_fraction : float
        Fraction of the cash spent on each buy signal, between 0 and 1.
    cooldown_days : int
        Minimum number of days between two trades.
    transaction_cost_rate : float
        Cost paid on each trade, as a fraction of the traded value.
    daily_cash_rate : float
        Interest earned by the cash balance each day, as a fraction.

    Returns
    -------
    pd.DataFrame
        Data frame indexed by date, with one row per date and the columns
        `action`, `price`, `shares`, `stock_value`, `cash` and
        `total_value`.

    Example
    -------
    >>> simulated = simulate_cooldown_portfolio(
    ...     dates=np.array(["2024-01-01", "2024-01-02"]),
    ...     price_values=np.array([100.0, 90.0]),
    ...     signal_values=np.array(["hold", "buy"]),
    ...     initial_stock_value=1000.0,
    ...     initial_cash=500.0,
    ...     sell_fraction=0.1,
    ...     buy_fraction=0.25,
    ...     cooldown_days=0,
    ...     transaction_cost_rate=0.0,
    ...     daily_cash_rate=0.0,
    ... )
    >>> float(round(simulated["total_value"].iloc[-1], 2))
    1400.0
    """
    shares_held = initial_stock_value / price_values[0]
    cash_balance = initial_cash
    last_trade_index = -cooldown_days
    records = []

    for index, current_price in enumerate(price_values):
        if index > 0:
            cash_balance *= 1.0 + daily_cash_rate

        current_signal = signal_values[index]
        action = "hold"
        is_in_cooldown = index - last_trade_index < cooldown_days

        if not is_in_cooldown:
            if current_signal == "sell" and shares_held > 0:
                sold_value = shares_held * current_price * sell_fraction
                trade_cost = sold_value * transaction_cost_rate
                shares_held -= sold_value / current_price
                cash_balance += sold_value - trade_cost
                action = "sell"
                last_trade_index = index
            elif current_signal == "buy" and cash_balance > 0:
                spent_cash = cash_balance * buy_fraction
                trade_cost = spent_cash * transaction_cost_rate
                shares_held += (spent_cash - trade_cost) / current_price
                cash_balance -= spent_cash
                action = "buy"
                last_trade_index = index

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


def compute_portfolio_metrics(
    portfolio_frame: pd.DataFrame,
    initial_total_value: float,
) -> dict[str, float]:
    """Summarize the performance of a simulated portfolio.

    Parameters
    ----------
    portfolio_frame : pd.DataFrame
        Data frame returned by `simulate_cooldown_portfolio`, with the
        columns `action` and `total_value`.
    initial_total_value : float
        Total value of the portfolio (stock plus cash) at the start of the
        simulation.

    Returns
    -------
    dict[str, float]
        Keys `final_total_value`, `total_return_percent`,
        `max_drawdown_percent`, `number_of_buys` and `number_of_sells`. The
        drawdown is the largest percentage drop from a previous peak of the
        total value.

    Example
    -------
    >>> metrics = compute_portfolio_metrics(portfolio_frame, 15000.0)
    >>> metrics["number_of_buys"]
    1
    """
    total_values = portfolio_frame["total_value"].to_numpy()
    running_peak = np.maximum.accumulate(total_values)
    drawdowns = total_values / running_peak - 1.0
    final_total_value = float(total_values[-1])
    total_return_percent = (
        final_total_value / initial_total_value - 1.0
    ) * 100.0

    return {
        "final_total_value": final_total_value,
        "total_return_percent": total_return_percent,
        "max_drawdown_percent": float(drawdowns.min()) * 100.0,
        "number_of_buys": int((portfolio_frame["action"] == "buy").sum()),
        "number_of_sells": int((portfolio_frame["action"] == "sell").sum()),
    }


def evaluate_cooldown_strategies(
    dates: np.ndarray,
    price_values: np.ndarray,
    model_signals: dict[str, np.ndarray],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Run the cooldown strategy for every model and for buy and hold.

    The parameters of the strategy are read from the module constants.

    Parameters
    ----------
    dates : np.ndarray
        Dates aligned with the prices.
    price_values : np.ndarray
        Price of the asset on each date.
    model_signals : dict[str, np.ndarray]
        Mapping from each strategy name to its array of `"buy"`, `"sell"`
        or `"hold"` signals.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], pd.DataFrame]
        The portfolio data frame of each strategy, keyed by name, and a
        table with one row of metrics per strategy, sorted by the final
        total value in descending order.

    Example
    -------
    >>> histories, table = evaluate_cooldown_strategies(
    ...     dates, price_values, {"modelo": signals}
    ... )
    >>> table["strategy"].iloc[0]
    'modelo'
    """
    initial_total_value = INITIAL_STOCK_VALUE + INITIAL_CASH
    daily_cash_rate = (1.0 + CASH_ANNUAL_RATE) ** (
        1.0 / TRADING_DAYS_PER_YEAR
    ) - 1.0
    hold_signal_values = np.full(len(price_values), "hold", dtype=object)

    all_signals = dict(model_signals)
    all_signals[BUY_AND_HOLD_NAME] = hold_signal_values

    portfolio_histories = {}
    metrics_rows = []

    for strategy_name, signal_values in all_signals.items():
        portfolio_frame = simulate_cooldown_portfolio(
            dates=dates,
            price_values=price_values,
            signal_values=signal_values,
            initial_stock_value=INITIAL_STOCK_VALUE,
            initial_cash=INITIAL_CASH,
            sell_fraction=SELL_FRACTION,
            buy_fraction=BUY_FRACTION,
            cooldown_days=COOLDOWN_DAYS,
            transaction_cost_rate=TRANSACTION_COST_RATE,
            daily_cash_rate=daily_cash_rate,
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


def plot_portfolio_evolution(
    portfolio_histories: dict[str, pd.DataFrame],
    initial_total_value: float,
    ticker: str,
) -> None:
    """Plot the total value of each strategy over time.

    The buy and hold strategy is drawn with a dashed line, and the initial
    total value is marked with a dotted gray line.

    Parameters
    ----------
    portfolio_histories : dict[str, pd.DataFrame]
        Portfolio data frame of each strategy, as returned by
        `evaluate_cooldown_strategies`.
    initial_total_value : float
        Total value of the portfolio at the start of the simulation, used
        as the reference line.
    ticker : str
        Ticker symbol shown in the chart title.

    Returns
    -------
    None
        The figure is displayed.
    """
    figure_portfolio, axis_portfolio = plt.subplots(figsize=(14, 6))
    for strategy_name, portfolio_frame in portfolio_histories.items():
        line_style = "--" if strategy_name == BUY_AND_HOLD_NAME else "-"
        axis_portfolio.plot(
            portfolio_frame.index,
            portfolio_frame["total_value"],
            linestyle=line_style,
            label=strategy_name,
        )
    axis_portfolio.axhline(
        initial_total_value,
        color="gray",
        linestyle=":",
        linewidth=0.8,
    )
    axis_portfolio.set_title(
        f"{ticker} - estrategia com cooldown, custos e rendimento do caixa"
    )
    axis_portfolio.set_xlabel("Data")
    axis_portfolio.set_ylabel("Patrimonio total")
    axis_portfolio.legend()
    plt.show()
    plt.close(figure_portfolio)


def run_strategy_evaluation(
    dates: np.ndarray,
    price_values: np.ndarray,
    model_signals: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Evaluate the strategies, print the metrics and plot the results.

    Parameters
    ----------
    dates : np.ndarray
        Dates aligned with the prices.
    price_values : np.ndarray
        Price of the asset on each date.
    model_signals : dict[str, np.ndarray]
        Mapping from each strategy name to its array of `"buy"`, `"sell"`
        or `"hold"` signals.

    Returns
    -------
    pd.DataFrame
        Table with the metrics of each strategy, sorted by the final total
        value in descending order.

    Example
    -------
    >>> metrics = run_strategy_evaluation(dates, prices, {"modelo": signals})
    """
    initial_total_value = INITIAL_STOCK_VALUE + INITIAL_CASH
    portfolio_histories, metrics_table = evaluate_cooldown_strategies(
        dates=dates,
        price_values=price_values,
        model_signals=model_signals,
    )
    print(metrics_table.round(2).to_string(index=False))
    plot_portfolio_evolution(
        portfolio_histories=portfolio_histories,
        initial_total_value=initial_total_value,
        ticker=TICKER,
    )

    return metrics_table
