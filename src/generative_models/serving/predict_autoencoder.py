"""Command-line script that scores recent prices with a wandb model."""

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from generative_models.data.market_data import fetch_price_series
from generative_models.models.autoencoder.config import TrackingConfig
from generative_models.serving.autoencoder_predictor import (
    AutoencoderPrediction,
    AutoencoderPredictor,
)
from generative_models.serving.wandb_autoencoder_registry import (
    WandbAutoencoderRegistry,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the parser of the prediction command-line arguments.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the model selection, period and display options.
    """
    today = date.today()
    parser = argparse.ArgumentParser(
        description="Score recent prices with a trained autoencoder.",
    )
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument(
        "--project-name", type=str, default=TrackingConfig.project_name
    )
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument(
        "--start-date",
        type=str,
        default=(today - timedelta(days=365)).isoformat(),
    )
    parser.add_argument("--end-date", type=str, default=today.isoformat())
    parser.add_argument("--top-outliers", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    return parser


def build_prediction_table(
    price_data: pd.DataFrame,
    prediction: AutoencoderPrediction,
) -> pd.DataFrame:
    """Align each window with the date and price where it ends.

    Parameters
    ----------
    price_data : pd.DataFrame
        Prices indexed by date, with a `close_price` column.
    prediction : AutoencoderPrediction
        Prediction made from these prices.

    Returns
    -------
    pd.DataFrame
        One row per window, indexed by the date of price row
        `window_end_index + 1`, with the columns `close_price`,
        `reconstruction_score` and `is_outlier`.
    """
    price_row_indices = prediction.window_end_indices + 1
    window_price_data = price_data.iloc[price_row_indices]
    return pd.DataFrame(
        {
            "close_price": np.asarray(
                window_price_data["close_price"]
            ).reshape(-1),
            "reconstruction_score": prediction.reconstruction_scores,
            "is_outlier": prediction.is_outlier,
        },
        index=window_price_data.index,
    )


def load_predictor(
    registry: WandbAutoencoderRegistry,
    ticker: str | None = None,
) -> AutoencoderPredictor:
    """Load the predictor of a ticker, or the latest one.

    Parameters
    ----------
    registry : WandbAutoencoderRegistry
        Registry used to load the model.
    ticker : str | None
        Ticker symbol. When empty, the latest model is loaded.

    Returns
    -------
    AutoencoderPredictor
        Loaded predictor.
    """
    if ticker:
        return registry.load_model(ticker)
    return registry.load_latest_model()


def predict_period(
    predictor: AutoencoderPredictor,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch the prices of a period and score every window.

    Parameters
    ----------
    predictor : AutoencoderPredictor
        Predictor whose ticker is fetched.
    start_date : str
        Start date, in `"YYYY-MM-DD"` format.
    end_date : str
        End date, in `"YYYY-MM-DD"` format.

    Returns
    -------
    pd.DataFrame
        Table built by `build_prediction_table`.
    """
    price_data = fetch_price_series(
        predictor.metadata.ticker,
        start_date,
        end_date,
    )
    prediction = predictor.predict_from_prices(
        np.asarray(price_data["close_price"]).reshape(-1)
    )
    return build_prediction_table(price_data, prediction)


def main(
    command_line_arguments: list[str] | None = None,
) -> pd.DataFrame:
    """Load a model, score the requested period and print a summary.

    Parameters
    ----------
    command_line_arguments : list[str] | None
        Arguments to parse. When `None`, `sys.argv` is used.

    Returns
    -------
    pd.DataFrame
        Prediction table of the period.

    Example
    -------
    >>> prediction_table = main(["--ticker", "PETR4.SA"])
    """
    arguments = build_argument_parser().parse_args(command_line_arguments)
    registry = WandbAutoencoderRegistry(
        project_name=arguments.project_name,
        entity=arguments.entity,
        device=arguments.device,
    )
    predictor = load_predictor(registry, arguments.ticker)
    prediction_table = predict_period(
        predictor,
        arguments.start_date,
        arguments.end_date,
    )

    metadata = predictor.metadata
    number_of_windows = len(prediction_table)
    number_of_outliers = int(prediction_table["is_outlier"].sum())
    outlier_percentage = (
        100 * number_of_outliers / number_of_windows
        if number_of_windows
        else 0.0
    )
    print(f"Artifact: {metadata.artifact_name}")
    print(
        f"Ticker: {metadata.ticker} | window_size: {metadata.window_size}"
        f" | anomaly_threshold: {metadata.anomaly_threshold:.6f}"
    )
    print(
        f"Period: {arguments.start_date} to {arguments.end_date} | "
        f"windows: {number_of_windows} | outliers: {number_of_outliers} "
        f"({outlier_percentage:.2f}%)"
    )
    print(
        "Mean reconstruction score: "
        f"{prediction_table['reconstruction_score'].mean():.6f}"
    )
    print(f"Top {arguments.top_outliers} windows by reconstruction score:")
    print(
        prediction_table.nlargest(
            arguments.top_outliers, "reconstruction_score"
        ).to_string()
    )
    if number_of_windows:
        last_window_status = (
            "OUTLIER" if prediction_table["is_outlier"].iloc[-1] else "normal"
        )
        print(f"Last window: {last_window_status}")
    return prediction_table


if __name__ == "__main__":
    main()
