"""Unit tests for the autoencoder prediction command line script."""

# pylint: disable=duplicate-code

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from generative_models.models.autoencoder.autoencoder import (
    SimpleAutoencoder,
)
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPredictor,
)
from generative_models.serving.predict_autoencoder import (
    build_argument_parser,
    build_prediction_table,
    load_predictor,
    main,
    predict_period,
)

SCRIPT_MODULE = "generative_models.serving.predict_autoencoder"
WINDOW_SIZE = 5


def build_predictor(ticker: str = "PETR4.SA") -> AutoencoderPredictor:
    """Build a predictor wrapping an untrained autoencoder.

    Parameters
    ----------
    ticker : str
        Ticker symbol stored in the predictor metadata.

    Returns
    -------
    AutoencoderPredictor
        Predictor on CPU, built from a fixed random seed.
    """
    torch.manual_seed(42)

    return AutoencoderPredictor(
        model=SimpleAutoencoder(input_dim=WINDOW_SIZE),
        metadata=AutoencoderModelMetadata(
            ticker=ticker,
            window_size=WINDOW_SIZE,
            return_train_mean=0.001,
            return_train_std=0.02,
            anomaly_threshold=0.5,
            artifact_name=f"simple-autoencoder-{ticker.lower()}:v1",
        ),
        device="cpu",
    )


def build_price_dataframe(number_of_prices: int) -> pd.DataFrame:
    """Build a synthetic price series indexed by business day.

    Parameters
    ----------
    number_of_prices : int
        Number of daily prices to generate.

    Returns
    -------
    pd.DataFrame
        Single-column data frame holding the prices in close_price.
    """
    random_generator = np.random.default_rng(5)
    log_increments = random_generator.normal(
        loc=0.0,
        scale=0.02,
        size=number_of_prices,
    )
    price_values = 100.0 * np.exp(np.cumsum(log_increments))
    price_dates = pd.bdate_range("2026-09-01", periods=number_of_prices)

    return pd.DataFrame({"close_price": price_values}, index=price_dates)


def test_default_arguments() -> None:
    """CA-01: without arguments the ticker is empty and dates default."""
    arguments = build_argument_parser().parse_args([])

    assert arguments.ticker is None
    assert arguments.top_outliers == 5
    assert arguments.end_date == date.today().isoformat()


def test_prediction_table_has_one_row_per_window() -> None:
    """CA-02: 10 prices and a window of 5 produce 5 dated rows."""
    price_data = build_price_dataframe(10)
    predictor = build_predictor()
    prediction = predictor.predict_from_prices(
        price_data["close_price"].to_numpy()
    )

    table = build_prediction_table(price_data, prediction)

    assert len(table) == 5
    assert list(table.columns) == [
        "close_price",
        "reconstruction_score",
        "is_outlier",
    ]
    assert list(table.index) == list(price_data.index[5:10])


def test_prediction_table_prices_follow_the_window_end() -> None:
    """CA-03: each row carries the price of window_end_index plus one."""
    price_data = build_price_dataframe(10)
    predictor = build_predictor()
    prediction = predictor.predict_from_prices(
        price_data["close_price"].to_numpy()
    )

    table = build_prediction_table(price_data, prediction)

    expected_prices = price_data["close_price"].to_numpy()[
        prediction.window_end_indices + 1
    ]

    assert np.allclose(table["close_price"].to_numpy(), expected_prices)


def test_load_predictor_chooses_between_latest_and_ticker() -> None:
    """CA-04: the ticker decides which registry method is called."""
    registry = MagicMock()

    load_predictor(registry, None)

    registry.load_latest_model.assert_called_once_with()
    registry.load_model.assert_not_called()

    registry.reset_mock()
    load_predictor(registry, "AAPL")

    registry.load_model.assert_called_once_with("AAPL")
    registry.load_latest_model.assert_not_called()


def test_predict_period_uses_the_metadata_ticker() -> None:
    """CA-05: prices are fetched for the ticker of the predictor."""
    predictor = build_predictor()
    price_data = build_price_dataframe(10)

    with patch(
        f"{SCRIPT_MODULE}.fetch_price_series",
        return_value=price_data,
    ) as fetch_mock:
        table = predict_period(predictor, "2026-09-01", "2026-09-30")

    fetch_mock.assert_called_once_with(
        "PETR4.SA",
        "2026-09-01",
        "2026-09-30",
    )
    assert len(table) == 5


def test_main_without_ticker_loads_the_latest_model() -> None:
    """CA-06: main with no arguments serves the most recent model."""
    predictor = build_predictor()
    price_data = build_price_dataframe(10)

    with patch(f"{SCRIPT_MODULE}.WandbAutoencoderRegistry") as registry_class:
        registry_class.return_value.load_latest_model.return_value = (
            predictor
        )
        with patch(
            f"{SCRIPT_MODULE}.fetch_price_series",
            return_value=price_data,
        ):
            table = main([])

    registry_class.return_value.load_latest_model.assert_called_once()
    assert len(table) == 5


def test_main_with_ticker_loads_that_model() -> None:
    """CA-07: main with a ticker serves the model of that ticker."""
    predictor = build_predictor()
    price_data = build_price_dataframe(10)

    with patch(f"{SCRIPT_MODULE}.WandbAutoencoderRegistry") as registry_class:
        registry_class.return_value.load_model.return_value = predictor
        with patch(
            f"{SCRIPT_MODULE}.fetch_price_series",
            return_value=price_data,
        ):
            main(["--ticker", "PETR4.SA"])

    registry_class.return_value.load_model.assert_called_once_with(
        "PETR4.SA"
    )
