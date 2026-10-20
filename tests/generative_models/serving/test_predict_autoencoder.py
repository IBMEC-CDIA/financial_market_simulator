"""Unit tests for the autoencoder prediction script."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPrediction,
    AutoencoderPredictor,
)
from generative_models.serving.predict_autoencoder import (
    build_argument_parser,
    build_prediction_table,
    main,
)

MODULE_PATH = "generative_models.serving.predict_autoencoder"


def _make_price_data(number_of_prices: int) -> pd.DataFrame:
    """Build a synthetic price series indexed by business day."""
    log_returns = np.random.default_rng(0).normal(0, 0.01, number_of_prices)
    return pd.DataFrame(
        {"close_price": 100.0 * np.exp(np.cumsum(log_returns))},
        index=pd.bdate_range("2026-01-01", periods=number_of_prices),
    )


def _make_predictor() -> AutoencoderPredictor:
    """Build a predictor around an untrained autoencoder."""
    torch.manual_seed(0)
    return AutoencoderPredictor(
        model=SimpleAutoencoder(input_dim=5),
        metadata=AutoencoderModelMetadata(
            ticker="PETR4.SA",
            window_size=5,
            return_train_mean=0.0,
            return_train_std=0.01,
            anomaly_threshold=1.0,
            artifact_name="simple-autoencoder-petr4.sa:v0",
        ),
    )


def test_parser_defaults_to_latest_model() -> None:
    """Check that no ticker is required and the period is one year."""
    arguments = build_argument_parser().parse_args([])

    assert arguments.ticker is None
    assert arguments.start_date < arguments.end_date


def test_build_prediction_table_aligns_windows_with_dates() -> None:
    """Check that each window is indexed by the date of its last return."""
    price_data = _make_price_data(8)
    prediction = AutoencoderPrediction(
        reconstruction_scores=np.array([0.1, 0.2, 3.0]),
        is_outlier=np.array([False, False, True]),
        window_end_indices=np.array([4, 5, 6]),
    )

    prediction_table = build_prediction_table(price_data, prediction)

    assert list(prediction_table.index) == list(price_data.index[5:8])
    np.testing.assert_allclose(
        prediction_table["close_price"],
        price_data["close_price"].to_numpy()[5:8],
    )
    assert prediction_table["is_outlier"].tolist() == [False, False, True]


@patch(f"{MODULE_PATH}.fetch_price_series")
@patch(f"{MODULE_PATH}.WandbAutoencoderRegistry")
def test_main_loads_latest_model_by_default(
    mock_registry_class, mock_fetch_price_series
) -> None:
    """Check that main loads the newest model and scores its ticker."""
    mock_registry = mock_registry_class.return_value
    mock_registry.load_latest_model.return_value = _make_predictor()
    mock_fetch_price_series.return_value = _make_price_data(30)

    prediction_table = main(["--start-date", "2026-01-01"])

    mock_registry.load_latest_model.assert_called_once_with()
    mock_registry.load_model.assert_not_called()
    assert mock_fetch_price_series.call_args.args[0] == "PETR4.SA"
    assert len(prediction_table) == 29 - 5 + 1


@patch(f"{MODULE_PATH}.fetch_price_series")
@patch(f"{MODULE_PATH}.WandbAutoencoderRegistry")
def test_main_loads_model_of_given_ticker(
    mock_registry_class, mock_fetch_price_series
) -> None:
    """Check that --ticker loads the model of that ticker."""
    mock_registry = mock_registry_class.return_value
    mock_registry.load_model.return_value = _make_predictor()
    mock_fetch_price_series.return_value = _make_price_data(30)

    main(["--ticker", "PETR4.SA"])

    mock_registry.load_model.assert_called_once_with("PETR4.SA")
    mock_registry.load_latest_model.assert_not_called()
