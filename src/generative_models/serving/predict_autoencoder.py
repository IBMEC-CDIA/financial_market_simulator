"""Command-line script that scores recent prices with a registered model."""

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

DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_TOP_OUTLIERS = 5


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the parser of the prediction script.

    The default period is the last 365 days, ending today. It is
    computed when the parser is built.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the options `--ticker`, `--project-name`, `--entity`,
        `--start-date`, `--end-date`, `--top-outliers` and `--device`.

    Example
    -------
    >>> arguments = build_argument_parser().parse_args(["--ticker", "AAPL"])
    >>> arguments.top_outliers
    5
    """
    today = date.today()
    default_start_date = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    parser = argparse.ArgumentParser(
        description="Score recent prices with a registered autoencoder.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker of the model to use. Defaults to the latest model.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=TrackingConfig.project_name,
        help="Name of the wandb project that holds the models.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="wandb entity (user or team). Defaults to the account entity.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=default_start_date.isoformat(),
        help="First date of the price history (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=today.isoformat(),
        help="Last date of the price history (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--top-outliers",
        type=int,
        default=DEFAULT_TOP_OUTLIERS,
        help="Number of windows with the highest scores to print.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device where the model runs, for example cpu or cuda.",
    )

    return parser


def build_prediction_table(
    price_data: pd.DataFrame,
    prediction: AutoencoderPrediction,
) -> pd.DataFrame:
    """Align the predicted windows with the dates of their prices.

    The window whose last log-return has index `i` ends at the price
    `i + 1`, so each row uses the price of that position.

    Parameters
    ----------
    price_data : pd.DataFrame
        Prices with a `close_price` column, indexed by date.
    prediction : AutoencoderPrediction
        Prediction made from the `close_price` values of `price_data`.

    Returns
    -------
    pd.DataFrame
        One row per window, indexed by the date of the window's price,
        with the columns `close_price`, `reconstruction_score` and
        `is_outlier`, in this order.

    Example
    -------
    >>> table = build_prediction_table(price_data, prediction)
    >>> list(table.columns)
    ['close_price', 'reconstruction_score', 'is_outlier']
    """
    price_row_positions = prediction.window_end_indices + 1
    close_prices = price_data["close_price"].to_numpy()

    return pd.DataFrame(
        {
            "close_price": close_prices[price_row_positions],
            "reconstruction_score": prediction.reconstruction_scores,
            "is_outlier": prediction.is_outlier,
        },
        index=price_data.index[price_row_positions],
    )


def load_predictor(
    registry: WandbAutoencoderRegistry,
    ticker: str | None = None,
) -> AutoencoderPredictor:
    """Load the predictor of a ticker, or the latest one when no ticker.

    Parameters
    ----------
    registry : WandbAutoencoderRegistry
        Registry that downloads the models from wandb.
    ticker : str | None
        Ticker symbol. When empty or `None`, the latest model of any
        ticker is loaded.

    Returns
    -------
    AutoencoderPredictor
        Predictor ready for inference.
    """
    if ticker:
        return registry.load_model(ticker)

    return registry.load_latest_model()


def predict_period(
    predictor: AutoencoderPredictor,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download the prices of a period and score every window of it.

    Parameters
    ----------
    predictor : AutoencoderPredictor
        Predictor whose ticker is used to download the prices.
    start_date : str
        First date of the period, in `"YYYY-MM-DD"` format.
    end_date : str
        Last date of the period, in `"YYYY-MM-DD"` format.

    Returns
    -------
    pd.DataFrame
        Prediction table, as built by `build_prediction_table`.

    Example
    -------
    >>> table = predict_period(predictor, "2026-01-01", "2026-06-30")
    """
    price_data = fetch_price_series(
        predictor.metadata.ticker,
        start_date,
        end_date,
    )
    prediction = predictor.predict_from_prices(
        price_data["close_price"].to_numpy()
    )

    return build_prediction_table(price_data, prediction)


def main(
    command_line_arguments: list[str] | None = None,
) -> pd.DataFrame:
    """Run the prediction for a ticker and print a summary.

    Parameters
    ----------
    command_line_arguments : list[str] | None
        Arguments to parse. When `None`, the arguments of `sys.argv` are
        used.

    Returns
    -------
    pd.DataFrame
        Prediction table of the period, as built by
        `build_prediction_table`.

    Raises
    ------
    SystemExit
        If the arguments are malformed (raised by `argparse`).

    Example
    -------
    >>> table = main(["--ticker", "PETR4.SA", "--top-outliers", "10"])
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
    _print_summary(predictor, prediction_table, arguments.top_outliers)

    return prediction_table


def _print_summary(
    predictor: AutoencoderPredictor,
    prediction_table: pd.DataFrame,
    top_outliers: int,
) -> None:
    """Print the model, the period and the most anomalous windows.

    Parameters
    ----------
    predictor : AutoencoderPredictor
        Predictor that produced the table.
    prediction_table : pd.DataFrame
        Prediction table with at least one window.
    top_outliers : int
        Number of windows with the highest scores to list.
    """
    metadata = predictor.metadata
    number_of_windows = len(prediction_table)
    number_of_outliers = int(prediction_table["is_outlier"].sum())
    outlier_percentage = 100 * number_of_outliers / number_of_windows
    last_status = (
        "OUTLIER" if prediction_table["is_outlier"].iloc[-1] else "normal"
    )

    print(f"Model: {metadata.artifact_name}")
    print(f"Ticker: {metadata.ticker}")
    print(f"Window size: {metadata.window_size}")
    print(f"Anomaly threshold: {metadata.anomaly_threshold:.6f}")
    print(
        f"Period: {prediction_table.index[0]} to "
        f"{prediction_table.index[-1]}"
    )
    print(
        f"Windows: {number_of_windows} | Outliers: {number_of_outliers} "
        f"({outlier_percentage:.2f}%)"
    )
    print(
        "Mean reconstruction score: "
        f"{np.mean(prediction_table['reconstruction_score']):.6f}"
    )
    print(f"Top {top_outliers} windows by reconstruction score:")
    top_windows = prediction_table.nlargest(
        top_outliers,
        "reconstruction_score",
    )
    print(top_windows.to_string())
    print(f"Last window: {last_status}")


if __name__ == "__main__":
    main()
