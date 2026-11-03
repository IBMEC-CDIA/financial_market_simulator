"""Unit tests for the random trading baseline."""

# pylint: disable=duplicate-code

import numpy as np
import pandas as pd
import pytest

from generative_models.strategy.random_baseline import (
    BASELINE_NAME_PREFIX,
    evaluate_random_baseline,
    evaluate_random_baselines,
    generate_random_signals,
    signal_frequencies,
)


def _dates(number_of_days: int) -> np.ndarray:
    """Build consecutive business dates."""
    return pd.bdate_range("2024-01-01", periods=number_of_days).to_numpy()


def test_signal_frequencies_count_buy_and_sell_days() -> None:
    """Check the fraction of buy and sell signals of a model."""
    signals = np.array(["buy", "hold", "sell", "hold"], dtype=object)

    assert signal_frequencies(signals) == (0.25, 0.25)


def test_zero_probabilities_give_only_hold_signals() -> None:
    """Check that no buy or sell is drawn when both probabilities are 0."""
    signals = generate_random_signals(
        number_of_days=20,
        buy_probability=0.0,
        sell_probability=0.0,
        random_generator=np.random.default_rng(0),
    )

    assert set(signals) == {"hold"}


def test_generated_signals_only_use_the_three_actions() -> None:
    """Check that the draw only returns buy, sell or hold."""
    signals = generate_random_signals(
        number_of_days=50,
        buy_probability=0.3,
        sell_probability=0.2,
        random_generator=np.random.default_rng(1),
    )

    assert set(signals) <= {"buy", "sell", "hold"}
    assert len(signals) == 50


def test_negative_probability_is_rejected() -> None:
    """Check that a negative probability raises ValueError."""
    with pytest.raises(ValueError, match="negative"):
        generate_random_signals(
            number_of_days=5,
            buy_probability=-0.1,
            sell_probability=0.1,
            random_generator=np.random.default_rng(0),
        )


def test_probabilities_above_one_are_rejected() -> None:
    """Check that buy and sell probabilities cannot add up to more than 1."""
    with pytest.raises(ValueError, match="more than 1"):
        generate_random_signals(
            number_of_days=5,
            buy_probability=0.7,
            sell_probability=0.6,
            random_generator=np.random.default_rng(0),
        )


def test_baseline_without_trades_matches_buy_and_hold() -> None:
    """Check that a model without signals gives the buy and hold result."""
    prices = np.array([100.0, 110.0, 120.0])
    hold_signals = np.array(["hold", "hold", "hold"], dtype=object)

    metrics = evaluate_random_baseline(
        dates=_dates(3),
        price_values=prices,
        model_signal_values=hold_signals,
        number_of_runs=5,
        random_seed=0,
        initial_stock_value=1000.0,
        initial_cash=5000.0,
        sell_fraction=0.2,
        buy_fraction=0.3,
    )

    assert metrics["final_total_value"] == pytest.approx(6200.0)
    assert metrics["final_total_value_std"] == pytest.approx(0.0)


def test_same_seed_gives_the_same_baseline() -> None:
    """Check that the baseline can be reproduced with the same seed."""
    prices = np.linspace(100.0, 130.0, 40)
    model_signals = np.array(["buy", "sell"] * 20, dtype=object)
    arguments = {
        "dates": _dates(40),
        "price_values": prices,
        "model_signal_values": model_signals,
        "number_of_runs": 10,
        "initial_stock_value": 10000.0,
        "initial_cash": 5000.0,
        "sell_fraction": 0.2,
        "buy_fraction": 0.3,
    }

    first = evaluate_random_baseline(random_seed=7, **arguments)
    second = evaluate_random_baseline(random_seed=7, **arguments)

    assert first == second


def test_zero_runs_are_rejected() -> None:
    """Check that the baseline needs at least one run."""
    with pytest.raises(ValueError, match="number_of_runs"):
        evaluate_random_baseline(
            dates=_dates(3),
            price_values=np.array([100.0, 100.0, 100.0]),
            model_signal_values=np.array(["hold"] * 3, dtype=object),
            number_of_runs=0,
            random_seed=0,
            initial_stock_value=1000.0,
            initial_cash=5000.0,
            sell_fraction=0.2,
            buy_fraction=0.3,
        )


def test_one_baseline_row_per_model_is_built() -> None:
    """Check the row name and the columns of the baseline table."""
    table = evaluate_random_baselines(
        dates=_dates(3),
        price_values=np.array([100.0, 101.0, 102.0]),
        model_signals={
            "modelo": np.array(["buy", "hold", "sell"], dtype=object),
        },
        number_of_runs=3,
        random_seed=0,
        initial_stock_value=1000.0,
        initial_cash=5000.0,
        sell_fraction=0.2,
        buy_fraction=0.3,
    )

    assert table["strategy"].tolist() == [f"{BASELINE_NAME_PREFIX}modelo"]
    assert "final_total_value" in table.columns
