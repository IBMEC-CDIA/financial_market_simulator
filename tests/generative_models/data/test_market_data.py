"""Unit tests for market data collection utilities."""

import numpy as np
import pandas as pd
import pytest

from generative_models.data.market_data import (
    add_log_return_column,
    add_split_column,
    compute_log_return_values,
    load_price_series,
    save_price_series,
)


def test_compute_log_return_values_matches_manual_calculation() -> None:
    """Check that the log-return formula matches a manual calculation."""
    price_values = np.array([100.0, 110.0, 99.0], dtype=np.float32)

    log_return_values = compute_log_return_values(price_values)

    expected_log_return_values = np.array(
        [
            np.log(110.0) - np.log(100.0),
            np.log(99.0) - np.log(110.0),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        log_return_values, expected_log_return_values, rtol=1e-6
    )


def test_compute_log_return_values_has_one_fewer_element() -> None:
    """Check that the output series has one fewer element than the input."""
    price_values = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float32)

    log_return_values = compute_log_return_values(price_values)

    assert len(log_return_values) == len(price_values) - 1


def test_add_log_return_column_matches_compute_log_return_values() -> None:
    """Check that the new column equals the log-returns, with NaN first."""
    price_data = pd.DataFrame({"close_price": [100.0, 110.0, 99.0]})

    dataset = add_log_return_column(price_data)

    assert list(dataset.columns) == ["close_price", "log_return"]
    assert len(dataset) == len(price_data)
    assert np.isnan(dataset["log_return"].iloc[0])
    np.testing.assert_allclose(
        dataset["log_return"].iloc[1:].to_numpy(),
        compute_log_return_values(price_data["close_price"].to_numpy()),
        rtol=1e-6,
    )


def test_add_split_column_marks_train_validation_and_test() -> None:
    """Check the chronological split and that the first row stays empty."""
    price_data = pd.DataFrame(
        {"close_price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}
    )
    dataset = add_log_return_column(price_data)

    split_dataset = add_split_column(
        dataset,
        train_split_ratio=0.4,
        validation_split_ratio=0.2,
    )

    assert list(split_dataset["split"].iloc[1:]) == [
        "train",
        "train",
        "validation",
        "test",
        "test",
    ]
    assert pd.isna(split_dataset["split"].iloc[0])


def test_add_split_column_rejects_non_positive_ratio() -> None:
    """Check that a ratio of zero or less is rejected."""
    dataset = add_log_return_column(
        pd.DataFrame({"close_price": [100.0, 101.0]})
    )

    with pytest.raises(ValueError, match="greater than 0"):
        add_split_column(
            dataset,
            train_split_ratio=0.0,
            validation_split_ratio=0.2,
        )


def test_add_split_column_rejects_ratios_that_leave_no_test_set() -> None:
    """Check that train and validation ratios must leave a test set."""
    dataset = add_log_return_column(
        pd.DataFrame({"close_price": [100.0, 101.0]})
    )

    with pytest.raises(ValueError, match="less than 1"):
        add_split_column(
            dataset,
            train_split_ratio=0.7,
            validation_split_ratio=0.3,
        )


def test_save_and_load_price_series_round_trip(tmp_path) -> None:
    """Check that a saved price series is loaded with the same values."""
    price_data = pd.DataFrame(
        {"close_price": [100.0, 101.5, 99.25]},
        index=pd.to_datetime(
            ["2026-01-02", "2026-01-05", "2026-01-06"],
        ).rename("date"),
    )
    output_path = tmp_path / "market-data" / "AAPL.csv"

    returned_path = save_price_series(price_data, output_path)
    loaded_data = load_price_series(output_path)

    assert returned_path == output_path
    assert output_path.exists()
    pd.testing.assert_frame_equal(loaded_data, price_data, check_freq=False)


def test_load_price_series_raises_without_close_price_column(tmp_path) -> None:
    """Check that a file without the close_price column is rejected."""
    input_path = tmp_path / "prices.csv"
    input_path.write_text("date,open_price\n2026-01-02,100.0\n")

    with pytest.raises(ValueError, match="close_price"):
        load_price_series(input_path)
