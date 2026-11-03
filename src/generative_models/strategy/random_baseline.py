"""Random trading baseline with the same signal frequencies as a model."""

# pylint: disable=duplicate-code

import numpy as np
import pandas as pd

from generative_models.strategy.cooldown_portfolio import (
    compute_portfolio_metrics,
)
from generative_models.strategy.fraction_portfolio import (
    simulate_fraction_portfolio,
)

BASELINE_NAME_PREFIX = "Aleatoria - "


def signal_frequencies(signal_values: np.ndarray) -> tuple[float, float]:
    """Compute how often a model gives buy and sell signals.

    Parameters
    ----------
    signal_values : np.ndarray
        `"buy"`, `"sell"` or `"hold"` for each date.

    Returns
    -------
    tuple[float, float]
        The fraction of the dates with a buy signal and the fraction of
        the dates with a sell signal.

    Example
    -------
    >>> signal_frequencies(np.array(["buy", "hold", "sell", "hold"]))
    (0.25, 0.25)
    """
    number_of_days = len(signal_values)
    buy_probability = float(np.sum(signal_values == "buy")) / number_of_days
    sell_probability = float(np.sum(signal_values == "sell")) / number_of_days

    return buy_probability, sell_probability


def generate_random_signals(
    number_of_days: int,
    buy_probability: float,
    sell_probability: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Draw a random sequence of buy, sell and hold signals.

    Parameters
    ----------
    number_of_days : int
        Number of signals to draw, one for each date.
    buy_probability : float
        Probability of a buy signal on each date.
    sell_probability : float
        Probability of a sell signal on each date.
    random_generator : np.random.Generator
        Generator used to draw the signals, so that runs can be repeated.

    Returns
    -------
    np.ndarray
        Array of `"buy"`, `"sell"` or `"hold"` with `number_of_days`
        elements.

    Raises
    ------
    ValueError
        If a probability is negative, or if the two probabilities add up
        to more than 1.

    Example
    -------
    >>> signals = generate_random_signals(
    ...     5, 0.2, 0.1, np.random.default_rng(0)
    ... )
    """
    if buy_probability < 0 or sell_probability < 0:
        raise ValueError("The signal probabilities cannot be negative.")

    hold_probability = 1.0 - buy_probability - sell_probability
    if hold_probability < -1e-12:
        raise ValueError(
            "The buy and sell probabilities add up to more than 1."
        )

    return random_generator.choice(
        np.array(["buy", "sell", "hold"], dtype=object),
        size=number_of_days,
        p=[buy_probability, sell_probability, max(hold_probability, 0.0)],
    )


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def evaluate_random_baseline(
    dates: np.ndarray,
    price_values: np.ndarray,
    model_signal_values: np.ndarray,
    number_of_runs: int,
    random_seed: int,
    initial_stock_value: float,
    initial_cash: float,
    sell_fraction: float,
    buy_fraction: float,
) -> dict[str, float]:
    """Average the metrics of random strategies that trade like a model.

    Each run draws random signals with the same buy and sell frequencies as
    the model, simulates the portfolio and computes its metrics. The
    result is the average of the metrics over all the runs.

    Parameters
    ----------
    dates : np.ndarray
        Dates aligned with the prices.
    price_values : np.ndarray
        Price of the asset on each date.
    model_signal_values : np.ndarray
        Signals of the model, used to take its buy and sell frequencies.
    number_of_runs : int
        Number of random strategies simulated.
    random_seed : int
        Seed of the random generator, so the result can be reproduced.
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
    dict[str, float]
        The average of each metric returned by `compute_portfolio_metrics`,
        plus `final_total_value_std`, the standard deviation of the final
        total value over the runs.

    Raises
    ------
    ValueError
        If `number_of_runs` is smaller than 1.
    """
    if number_of_runs < 1:
        raise ValueError("number_of_runs must be at least 1.")

    buy_probability, sell_probability = signal_frequencies(
        model_signal_values
    )
    initial_total_value = initial_stock_value + initial_cash
    random_generator = np.random.default_rng(random_seed)
    run_metrics = []

    for _ in range(number_of_runs):
        signal_values = generate_random_signals(
            number_of_days=len(price_values),
            buy_probability=buy_probability,
            sell_probability=sell_probability,
            random_generator=random_generator,
        )
        portfolio_frame = simulate_fraction_portfolio(
            dates=dates,
            price_values=price_values,
            signal_values=signal_values,
            initial_stock_value=initial_stock_value,
            initial_cash=initial_cash,
            sell_fraction=sell_fraction,
            buy_fraction=buy_fraction,
        )
        run_metrics.append(
            compute_portfolio_metrics(portfolio_frame, initial_total_value)
        )

    metrics_frame = pd.DataFrame(run_metrics)
    average_metrics = {
        name: float(value) for name, value in metrics_frame.mean().items()
    }
    average_metrics["final_total_value_std"] = float(
        metrics_frame["final_total_value"].std(ddof=0)
    )

    return average_metrics


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def evaluate_random_baselines(
    dates: np.ndarray,
    price_values: np.ndarray,
    model_signals: dict[str, np.ndarray],
    number_of_runs: int,
    random_seed: int,
    initial_stock_value: float,
    initial_cash: float,
    sell_fraction: float,
    buy_fraction: float,
) -> pd.DataFrame:
    """Build one random baseline per model, for comparison in a table.

    Parameters
    ----------
    dates : np.ndarray
        Dates aligned with the prices.
    price_values : np.ndarray
        Price of the asset on each date.
    model_signals : dict[str, np.ndarray]
        Mapping from each model name to its array of signals.
    number_of_runs : int
        Number of random strategies simulated for each model.
    random_seed : int
        Seed of the random generator, shared by all the models.
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
    pd.DataFrame
        One row per model, named `"Aleatoria - <model>"`, with the same
        metric columns as the strategy table.

    Example
    -------
    >>> baselines = evaluate_random_baselines(
    ...     dates, prices, {"modelo": signals}, 100, 42,
    ...     10000.0, 5000.0, 0.2, 0.3,
    ... )
    """
    rows = []
    for model_name, signal_values in model_signals.items():
        metrics = evaluate_random_baseline(
            dates=dates,
            price_values=price_values,
            model_signal_values=signal_values,
            number_of_runs=number_of_runs,
            random_seed=random_seed,
            initial_stock_value=initial_stock_value,
            initial_cash=initial_cash,
            sell_fraction=sell_fraction,
            buy_fraction=buy_fraction,
        )
        rows.append(
            {"strategy": f"{BASELINE_NAME_PREFIX}{model_name}", **metrics}
        )

    return pd.DataFrame(rows)
