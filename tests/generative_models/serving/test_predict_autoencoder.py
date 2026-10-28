"""Unit tests for the autoencoder prediction script."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder
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

MODULE_PATH = "generative_models.serving.predict_autoencoder"


def _make_predictor(ticker: str = "PETR4.SA") -> AutoencoderPredictor:
    """Build a predictor around an untrained autoencoder."""
    torch.manual_seed(0)
    return AutoencoderPredictor(
        model=SimpleAutoencoder(input_dim=5),
        metadata=AutoencoderModelMetadata(
            ticker=ticker,
            window_size=5,
            return_train_mean=0.0,
            return_train_std=0.02,
            anomaly_threshold=1.0,
            artifact_name="simple-autoencoder-petr4.sa:v0",
        ),
    )


def _make_price_data(number_of_prices: int) -> pd.DataFrame:
    """Build a synthetic price series indexed by business days."""
    random_generator = np.random.default_rng(0)
    price_values = 100 * np.exp(
        np.cumsum(random_generator.normal(0, 0.01, number_of_prices))
    )
    return pd.DataFrame(
        {"close_price": price_values},
        index=pd.bdate_range("2026-09-01", periods=number_of_prices),
    )


def test_parser_defaults() -> None:
    """CA-01: no ticker, five top outliers and end date today."""
    arguments = build_argument_parser().parse_args([])

    assert arguments.ticker is None
    assert arguments.top_outliers == 5
    assert arguments.end_date == date.today().isoformat()


def test_prediction_table_shape_and_index() -> None:
    """CA-02: 10 prices give 5 rows indexed by the dates of prices 5-9."""
    price_data = _make_price_data(10)
    prediction = _make_predictor().predict_from_prices(
        price_data["close_price"].to_numpy()
    )

    prediction_table = build_prediction_table(price_data, prediction)

    assert len(prediction_table) == 5
    assert list(prediction_table.columns) == [
        "close_price",
        "reconstruction_score",
        "is_outlier",
    ]
    assert prediction_table.index.equals(price_data.index[5:10])


def test_prediction_table_aligns_prices() -> None:
    """CA-03: each row holds the price at window_end_index + 1."""
    price_data = _make_price_data(10)
    prediction = _make_predictor().predict_from_prices(
        price_data["close_price"].to_numpy()
    )

    prediction_table = build_prediction_table(price_data, prediction)

    expected_prices = price_data["close_price"].to_numpy()[
        prediction.window_end_indices + 1
    ]
    np.testing.assert_allclose(
        prediction_table["close_price"].to_numpy(), expected_prices
    )


def test_load_predictor_dispatch() -> None:
    """CA-04: the latest model without ticker, the ticker model with it."""
    latest_registry = MagicMock()
    ticker_registry = MagicMock()

    load_predictor(latest_registry, None)
    load_predictor(ticker_registry, "AAPL")

    latest_registry.load_latest_model.assert_called_once_with()
    latest_registry.load_model.assert_not_called()
    ticker_registry.load_model.assert_called_once_with("AAPL")
    ticker_registry.load_latest_model.assert_not_called()


@patch(f"{MODULE_PATH}.fetch_price_series")
def test_predict_period_fetches_ticker_prices(mock_fetch) -> None:
    """CA-05: prices are fetched for the predictor ticker and dates."""
    mock_fetch.return_value = _make_price_data(10)

    predict_period(_make_predictor("AAPL"), "2026-09-01", "2026-09-30")

    mock_fetch.assert_called_once_with("AAPL", "2026-09-01", "2026-09-30")


@patch(f"{MODULE_PATH}.fetch_price_series")
@patch(f"{MODULE_PATH}.WandbAutoencoderRegistry")
def test_main_loads_latest_model(mock_registry_class, mock_fetch) -> None:
    """CA-06: without ticker, main loads the latest model."""
    mock_registry = mock_registry_class.return_value
    mock_registry.load_latest_model.return_value = _make_predictor()
    mock_fetch.return_value = _make_price_data(10)

    prediction_table = main([])

    mock_registry.load_latest_model.assert_called_once_with()
    assert isinstance(prediction_table, pd.DataFrame)
    assert len(prediction_table) == 5


@patch(f"{MODULE_PATH}.fetch_price_series")
@patch(f"{MODULE_PATH}.WandbAutoencoderRegistry")
def test_main_loads_ticker_model(mock_registry_class, mock_fetch) -> None:
    """CA-07: with --ticker, main loads that ticker's model."""
    mock_registry = mock_registry_class.return_value
    mock_registry.load_model.return_value = _make_predictor()
    mock_fetch.return_value = _make_price_data(10)

    main(["--ticker", "PETR4.SA"])

    mock_registry.load_model.assert_called_once_with("PETR4.SA")
