"""End-to-end test of the collect-data, train and evaluate jobs.

The prices are simulated, so the test does not depend on yfinance. The
wandb client is disabled, so the test does not need an API key.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from generative_models.data.collect_market_data import main as collect_main
from generative_models.models.autoencoder.overfitting import (
    build_report_path,
    load_overfitting_report,
)
from generative_models.models.autoencoder.overfitting import (
    main as evaluate_main,
)
from generative_models.models.autoencoder.train_autoencoder import (
    main as train_main,
)

FETCH_PATH = "generative_models.data.collect_market_data.fetch_price_series"
NUMBER_OF_PRICES = 400
TRAIN_SPLIT_RATIO = 0.7
VALIDATION_SPLIT_RATIO = 0.15


def _make_fetched_prices() -> pd.DataFrame:
    """Build a simulated price series in the format returned by yfinance."""
    rng = np.random.default_rng(42)
    price_values = 100 * np.exp(
        np.cumsum(rng.normal(0.0, 0.01, NUMBER_OF_PRICES))
    )
    index = pd.bdate_range(
        "2020-01-01",
        periods=NUMBER_OF_PRICES,
        name="Date",
    )

    return pd.DataFrame({"close_price": price_values}, index=index)


@pytest.fixture
def offline_wandb(monkeypatch) -> None:
    """Disable wandb for the test, so no run or artifact is sent."""
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.mark.usefixtures("offline_wandb")
@patch(FETCH_PATH)
def test_collect_train_and_evaluate_run_end_to_end(
    mock_fetch_price_series,
    tmp_path,
) -> None:
    """Check that the three jobs chain together and produce their files."""
    mock_fetch_price_series.return_value = _make_fetched_prices()
    prices_path = tmp_path / "market-data" / "AAPL.csv"
    artifacts_directory = tmp_path / "artifacts"

    collect_main(["--ticker", "AAPL", "--output-path", str(prices_path)])

    collected_data = pd.read_csv(prices_path, index_col=0)
    expected_columns = {"close_price", "log_return", "split"}
    assert expected_columns <= set(collected_data.columns)
    split_counts = collected_data["split"].value_counts()
    return_count = NUMBER_OF_PRICES - 1
    expected_train_count = int(return_count * TRAIN_SPLIT_RATIO)
    expected_validation_count = int(return_count * VALIDATION_SPLIT_RATIO)
    assert split_counts["train"] == expected_train_count
    assert split_counts["validation"] == expected_validation_count
    assert split_counts["test"] == (
        return_count - expected_train_count - expected_validation_count
    )

    train_main(
        [
            "--ticker",
            "AAPL",
            "--price-data-path",
            str(prices_path),
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--output-directory",
            str(artifacts_directory),
        ]
    )

    model_path = artifacts_directory / "simple-autoencoder-aapl.pt"
    report_path = build_report_path(model_path)
    assert model_path.exists()
    assert report_path.exists()
    report = load_overfitting_report(report_path)
    assert report.ticker == "AAPL"
    assert report.overfitting_ratio > 0

    assert report.test_loss is not None
    assert report.baseline_test_loss is not None
    assert evaluate_main(
        ["--report-path", str(report_path), "--max-overfitting-ratio", "0"]
    ) is False
    assert evaluate_main(
        ["--report-path", str(report_path), "--max-overfitting-ratio", "1e9"]
    ) is (report.test_loss < report.baseline_test_loss)
