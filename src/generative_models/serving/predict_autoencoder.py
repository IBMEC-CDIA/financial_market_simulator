"""Command line script that scores a period with a trained model."""

import argparse
from datetime import date, timedelta

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

DEFAULT_HISTORY_DAYS = 365
DEFAULT_TOP_OUTLIERS = 5


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the parser of the prediction script.

    Returns
    -------
    argparse.ArgumentParser
        Parser whose arguments are all optional: without a ticker the
        most recent model of the project is served, and the default
        period covers the last year.

    Example
    -------
    >>> arguments = build_argument_parser().parse_args([])
    >>> arguments.ticker is None
    True
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=DEFAULT_HISTORY_DAYS)

    parser = argparse.ArgumentParser(
        description=(
            "Score the log-return windows of a period with an "
            "autoencoder registered in wandb."
        )
    )
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument(
        "--project-name",
        type=str,
        default=TrackingConfig.project_name,
    )
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument(
        "--start-date",
        type=str,
        default=start_date.isoformat(),
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=end_date.isoformat(),
    )
    parser.add_argument(
        "--top-outliers",
        type=int,
        default=DEFAULT_TOP_OUTLIERS,
    )
    parser.add_argument("--device", type=str, default=None)

    return parser


def build_prediction_table(
    price_data: pd.DataFrame,
    prediction: AutoencoderPrediction,
) -> pd.DataFrame:
    """Align a prediction with the dates and prices it came from.

    Parameters
    ----------
    price_data : pd.DataFrame
        Prices indexed by date, with the closing price in the
        `close_price` column.
    prediction : AutoencoderPrediction
        Prediction computed from the prices of `price_data`.

    Returns
    -------
    pd.DataFrame
        One row per window, indexed by the date of the price row that
        closes the window, with the columns `close_price`,
        `reconstruction_score` and `is_outlier`. A window ending at
        the log-return index `i` refers to the return between the
        prices `i` and `i + 1`, so the price row of the window is
        `i + 1`.

    Example
    -------
    >>> table = build_prediction_table(price_data, prediction)
    >>> list(table.columns)
    ['close_price', 'reconstruction_score', 'is_outlier']
    """
    price_row_indices = prediction.window_end_indices + 1

    return pd.DataFrame(
        {
            "close_price": price_data["close_price"].to_numpy()[
                price_row_indices
            ],
            "reconstruction_score": prediction.reconstruction_scores,
            "is_outlier": prediction.is_outlier,
        },
        index=price_data.index[price_row_indices],
    )


def load_predictor(
    registry: WandbAutoencoderRegistry,
    ticker: str | None = None,
) -> AutoencoderPredictor:
    """Load either the model of a ticker or the most recent one.

    Parameters
    ----------
    registry : WandbAutoencoderRegistry
        Registry used to read the models from wandb.
    ticker : str | None
        Ticker symbol to serve. When `None` or empty, the most
        recently created model of the project is served.

    Returns
    -------
    AutoencoderPredictor
        Predictor ready to score a series.
    """
    if ticker:
        return registry.load_model(ticker)

    return registry.load_latest_model()


def predict_period(
    predictor: AutoencoderPredictor,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Score every window of a period of market data.

    Parameters
    ----------
    predictor : AutoencoderPredictor
        Predictor whose metadata names the ticker to download.
    start_date : str
        Start date of the period, in `"YYYY-MM-DD"` format.
    end_date : str
        End date of the period, in `"YYYY-MM-DD"` format.

    Returns
    -------
    pd.DataFrame
        Table returned by `build_prediction_table` for the downloaded
        prices.

    Example
    -------
    >>> table = predict_period(predictor, "2026-01-01", "2026-06-01")
    >>> int(table["is_outlier"].sum())
    4
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


def _print_prediction_summary(
    predictor: AutoencoderPredictor,
    prediction_table: pd.DataFrame,
    top_outliers: int,
) -> None:
    """Print a human-readable summary of a prediction.

    Parameters
    ----------
    predictor : AutoencoderPredictor
        Predictor that produced the table, used for the header.
    prediction_table : pd.DataFrame
        Table returned by `predict_period`.
    top_outliers : int
        Number of highest scoring windows to list.
    """
    metadata = predictor.metadata
    number_of_windows = len(prediction_table)
    number_of_outliers = int(prediction_table["is_outlier"].sum())
    outlier_rate = (
        100 * number_of_outliers / number_of_windows
        if number_of_windows
        else 0.0
    )

    print(f"Artifact: {metadata.artifact_name}")
    print(f"Ticker: {metadata.ticker}")
    print(f"Window size: {metadata.window_size}")
    print(f"Anomaly threshold: {metadata.anomaly_threshold:.6f}")

    if number_of_windows == 0:
        print("No window could be scored for the requested period.")
        return

    print(
        f"Period: {prediction_table.index[0].date()} to "
        f"{prediction_table.index[-1].date()}"
    )
    print(f"Windows scored: {number_of_windows}")
    print(f"Outlier windows: {number_of_outliers} ({outlier_rate:.2f}%)")
    print(
        "Mean reconstruction score: "
        f"{prediction_table['reconstruction_score'].mean():.6f}"
    )

    print(f"Top {top_outliers} windows by reconstruction score:")
    highest_scores = prediction_table.nlargest(
        top_outliers,
        "reconstruction_score",
    )
    for window_date, window_row in highest_scores.iterrows():
        print(
            f"  {window_date.date()} | "
            f"close: {window_row['close_price']:.2f} | "
            f"score: {window_row['reconstruction_score']:.6f} | "
            f"outlier: {bool(window_row['is_outlier'])}"
        )

    last_window = prediction_table.iloc[-1]
    last_status = (
        "OUTLIER" if bool(last_window["is_outlier"]) else "normal"
    )
    print(
        f"Last window ({prediction_table.index[-1].date()}): "
        f"{last_status}"
    )


def main(
    command_line_arguments: list[str] | None = None,
) -> pd.DataFrame:
    """Score a period with a model registered in wandb.

    Parameters
    ----------
    command_line_arguments : list[str] | None
        Arguments to parse, without the program name. When `None`,
        the arguments of `sys.argv` are used.

    Returns
    -------
    pd.DataFrame
        Prediction table of the requested period, already summarized
        on the standard output.

    Example
    -------
    >>> table = main(["--ticker", "PETR4.SA"])
    >>> list(table.columns)
    ['close_price', 'reconstruction_score', 'is_outlier']
    """
    parser = build_argument_parser()
    arguments = parser.parse_args(command_line_arguments)

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

    _print_prediction_summary(
        predictor,
        prediction_table,
        arguments.top_outliers,
    )

    return prediction_table


if __name__ == "__main__":
    main()
