"""Unit tests for the prediction command-line script."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from generative_models.serving.predict_autoencoder import (
    build_argument_parser,
    build_prediction_table,
    load_predictor,
    main,
    predict_period,
)

FETCH_PATH = "generative_models.serving.predict_autoencoder.fetch_price_series"
REGISTRY_PATH = (
    "generative_models.serving.predict_autoencoder.WandbAutoencoderRegistry"
)


def test_parser_defaults_to_the_latest_model_and_today() -> None:
    """Check the defaults when no argument is given."""
    arguments = build_argument_parser().parse_args([])

    assert arguments.ticker is None
    assert arguments.top_outliers == 5
    assert arguments.end_date == date.today().isoformat()


def test_build_prediction_table_aligns_windows_with_prices(
    make_predictor,
    make_price_frame,
) -> None:
    """Check the columns and the dates of the table for 10 prices."""
    predictor = make_predictor()
    price_data = make_price_frame(10)
    prediction = predictor.predict_from_prices(
        price_data["close_price"].to_numpy()
    )

    prediction_table = build_prediction_table(price_data, prediction)

    assert len(prediction_table) == 5
    assert list(prediction_table.columns) == [
        "close_price",
        "reconstruction_score",
        "is_outlier",
    ]
    assert list(prediction_table.index) == list(price_data.index[5:10])


def test_build_prediction_table_uses_the_price_after_each_window_end(
    make_predictor,
    make_price_frame,
) -> None:
    """Check that each row shows the price at window_end_index + 1."""
    predictor = make_predictor()
    price_data = make_price_frame(10)
    prediction = predictor.predict_from_prices(
        price_data["close_price"].to_numpy()
    )

    prediction_table = build_prediction_table(price_data, prediction)

    for row_position, window_end_index in enumerate(
        prediction.window_end_indices
    ):
        expected_price = price_data["close_price"].iloc[window_end_index + 1]
        assert prediction_table["close_price"].iloc[row_position] == (
            expected_price
        )


def test_load_predictor_uses_latest_or_named_model() -> None:
    """Check which registry method is called for each ticker input."""
    registry = MagicMock()

    load_predictor(registry, None)
    load_predictor(registry, "AAPL")

    registry.load_latest_model.assert_called_once_with()
    registry.load_model.assert_called_once_with("AAPL")


@patch(FETCH_PATH)
def test_predict_period_fetches_prices_of_the_metadata_ticker(
    mock_fetch_price_series,
    make_predictor,
    make_price_frame,
) -> None:
    """Check that prices are downloaded for the predictor's ticker."""
    mock_fetch_price_series.return_value = make_price_frame(10)
    predictor = make_predictor(ticker="PETR4.SA")

    prediction_table = predict_period(
        predictor,
        "2026-09-01",
        "2026-09-30",
    )

    mock_fetch_price_series.assert_called_once_with(
        "PETR4.SA",
        "2026-09-01",
        "2026-09-30",
    )
    assert len(prediction_table) == 5


@patch(FETCH_PATH)
@patch(REGISTRY_PATH)
def test_main_without_arguments_loads_the_latest_model(
    mock_registry_class,
    mock_fetch_price_series,
    capsys,
    make_predictor,
    make_price_frame,
) -> None:
    """Check that main with no arguments uses the latest model."""
    registry = mock_registry_class.return_value
    registry.load_latest_model.return_value = make_predictor()
    mock_fetch_price_series.return_value = make_price_frame(20)

    prediction_table = main([])

    registry.load_latest_model.assert_called_once_with()
    assert isinstance(prediction_table, pd.DataFrame)
    assert len(prediction_table) == 15
    assert "Model: simple-autoencoder-petr4.sa:v1" in capsys.readouterr().out


@patch(FETCH_PATH)
@patch(REGISTRY_PATH)
def test_main_with_ticker_loads_that_ticker(
    mock_registry_class,
    mock_fetch_price_series,
    make_predictor,
    make_price_frame,
) -> None:
    """Check that --ticker selects the model of that ticker."""
    registry = mock_registry_class.return_value
    registry.load_model.return_value = make_predictor()
    mock_fetch_price_series.return_value = make_price_frame(20)

    main(["--ticker", "PETR4.SA"])

    registry.load_model.assert_called_once_with("PETR4.SA")
