"""Unit tests for the autoencoder prediction command line script."""

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


def build_predictor(ticker: str = "PETR4.SA") -> AutoencoderPredictor:
    """Create a predictor around an untrained autoencoder.

    Parameters
    ----------
    ticker : str
        Ticker stored in the metadata.

    Returns
    -------
    AutoencoderPredictor
        Predictor with a seeded `SimpleAutoencoder(input_dim=5)`.
    """
    torch.manual_seed(0)
    return AutoencoderPredictor(
        model=SimpleAutoencoder(input_dim=5),
        metadata=AutoencoderModelMetadata(
            ticker=ticker,
            window_size=5,
            return_train_mean=0.0,
            return_train_std=0.02,
            anomaly_threshold=1.0,
            artifact_name="simple-autoencoder-petr4.sa:v1",
        ),
    )


def build_price_data(number_of_prices: int) -> pd.DataFrame:
    """Create a synthetic price series indexed by business days.

    Parameters
    ----------
    number_of_prices : int
        Number of prices in the series.

    Returns
    -------
    pd.DataFrame
        Data frame with a `close_price` column.
    """
    random_generator = np.random.default_rng(11)
    log_returns = random_generator.normal(0.0, 0.02, number_of_prices)
    return pd.DataFrame(
        {"close_price": 100.0 * np.exp(np.cumsum(log_returns))},
        index=pd.bdate_range("2026-09-01", periods=number_of_prices),
    )


def test_parser_defaults() -> None:
    """CA-01: check the default ticker, top outliers and end date."""
    arguments = build_argument_parser().parse_args([])

    assert arguments.ticker is None
    assert arguments.top_outliers == 5
    assert arguments.end_date == date.today().isoformat()


def test_prediction_table_has_one_row_per_window() -> None:
    """CA-02: check the rows, columns and dates of the table."""
    price_data = build_price_data(10)
    prediction = build_predictor().predict_from_prices(
        price_data["close_price"].to_numpy()
    )

    prediction_table = build_prediction_table(price_data, prediction)

    assert len(prediction_table) == 5
    assert list(prediction_table.columns) == [
        "close_price",
        "reconstruction_score",
        "is_outlier",
    ]
    pd.testing.assert_index_equal(
        prediction_table.index,
        price_data.index[5:10],
    )


def test_prediction_table_aligns_close_price() -> None:
    """CA-03: check that each row holds the price after the window end."""
    price_data = build_price_data(10)
    prediction = build_predictor().predict_from_prices(
        price_data["close_price"].to_numpy()
    )

    prediction_table = build_prediction_table(price_data, prediction)

    np.testing.assert_array_equal(
        prediction_table["close_price"].to_numpy(),
        price_data["close_price"].to_numpy()[
            prediction.window_end_indices + 1
        ],
    )


def test_load_predictor_chooses_registry_method() -> None:
    """CA-04: check which registry method each call uses."""
    latest_registry = MagicMock()
    ticker_registry = MagicMock()

    load_predictor(latest_registry, None)
    load_predictor(ticker_registry, "AAPL")

    latest_registry.load_latest_model.assert_called_once_with()
    latest_registry.load_model.assert_not_called()
    ticker_registry.load_model.assert_called_once_with("AAPL")
    ticker_registry.load_latest_model.assert_not_called()


def test_predict_period_fetches_prices_of_model_ticker() -> None:
    """CA-05: check the ticker and dates used to fetch the prices."""
    with patch(
        f"{MODULE_PATH}.fetch_price_series",
        return_value=build_price_data(10),
    ) as mock_fetch_price_series:
        prediction_table = predict_period(
            build_predictor("AAPL"),
            "2026-09-01",
            "2026-09-30",
        )

    mock_fetch_price_series.assert_called_once_with(
        "AAPL",
        "2026-09-01",
        "2026-09-30",
    )
    assert len(prediction_table) == 5


def test_main_without_ticker_loads_latest_model() -> None:
    """CA-06: check that main loads the latest model and returns a table."""
    with (
        patch(f"{MODULE_PATH}.WandbAutoencoderRegistry") as mock_registry,
        patch(
            f"{MODULE_PATH}.fetch_price_series",
            return_value=build_price_data(12),
        ),
    ):
        registry = mock_registry.return_value
        registry.load_latest_model.return_value = build_predictor()

        prediction_table = main([])

    registry.load_latest_model.assert_called_once_with()
    registry.load_model.assert_not_called()
    assert len(prediction_table) == 7


def test_main_with_ticker_loads_ticker_model() -> None:
    """CA-07: check that --ticker loads the model of that ticker."""
    with (
        patch(f"{MODULE_PATH}.WandbAutoencoderRegistry") as mock_registry,
        patch(
            f"{MODULE_PATH}.fetch_price_series",
            return_value=build_price_data(12),
        ),
    ):
        registry = mock_registry.return_value
        registry.load_model.return_value = build_predictor()

        main(["--ticker", "PETR4.SA"])

    registry.load_model.assert_called_once_with("PETR4.SA")
    registry.load_latest_model.assert_not_called()
