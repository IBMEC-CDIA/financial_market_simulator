"""Command line entry point for scoring recent prices with a model.

Example
-------
uv run python -m generative_models.serving.predict_autoencoder \\
    --ticker PETR4.SA --start-date 2026-01-01 --top-outliers 10
"""

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


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the parser of the prediction command line arguments.

    Returns
    -------
    argparse.ArgumentParser
        Parser with optional ticker, wandb project, period, number of
        top outliers and device arguments.
    """
    today = date.today()
    argument_parser = argparse.ArgumentParser(
        description=(
            "Load a SimpleAutoencoder registered in wandb and score the "
            "recent log-return windows of its ticker."
        ),
    )
    argument_parser.add_argument("--ticker", type=str, default=None)
    argument_parser.add_argument(
        "--project-name",
        type=str,
        default=TrackingConfig.project_name,
    )
    argument_parser.add_argument("--entity", type=str, default=None)
    argument_parser.add_argument(
        "--start-date",
        type=str,
        default=(today - timedelta(days=365)).isoformat(),
    )
    argument_parser.add_argument(
        "--end-date",
        type=str,
        default=today.isoformat(),
    )
    argument_parser.add_argument("--top-outliers", type=int, default=5)
    argument_parser.add_argument("--device", type=str, default=None)
    return argument_parser


def build_prediction_table(
    price_data: pd.DataFrame,
    prediction: AutoencoderPrediction,
) -> pd.DataFrame:
    """Align each scored window with the date of its last price.

    Parameters
    ----------
    price_data : pd.DataFrame
        Prices indexed by date, with a `close_price` column.
    prediction : AutoencoderPrediction
        Prediction made from the prices of `price_data`.

    Returns
    -------
    pd.DataFrame
        One row per window, indexed by the date of price row
        `window_end_index + 1`, with the columns `close_price`,
        `reconstruction_score` and `is_outlier`.

    Example
    -------
    >>> prediction = predictor.predict_from_prices(
    ...     price_data["close_price"].to_numpy(),
    ... )
    >>> build_prediction_table(price_data, prediction).columns.tolist()
    ['close_price', 'reconstruction_score', 'is_outlier']
    """
    window_price_data = price_data.iloc[prediction.window_end_indices + 1]
    return pd.DataFrame(
        {
            "close_price": window_price_data["close_price"].to_numpy(),
            "reconstruction_score": prediction.reconstruction_scores,
            "is_outlier": prediction.is_outlier,
        },
        index=window_price_data.index,
    )


def load_predictor(
    registry: WandbAutoencoderRegistry,
    ticker: str | None = None,
) -> AutoencoderPredictor:
    """Load the model of a ticker or the most recent model.

    Parameters
    ----------
    registry : WandbAutoencoderRegistry
        Registry used to load the model.
    ticker : str | None
        Ticker symbol of the model. `None` or an empty string loads
        the most recently registered model.

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
    """Download the prices of a period and score every window.

    Parameters
    ----------
    predictor : AutoencoderPredictor
        Predictor whose metadata ticker is downloaded.
    start_date : str
        First date of the period, in `YYYY-MM-DD` format.
    end_date : str
        Last date of the period, in `YYYY-MM-DD` format.

    Returns
    -------
    pd.DataFrame
        Prediction table built by `build_prediction_table`.
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
    """Load a model, score a recent period and print a summary.

    Parameters
    ----------
    command_line_arguments : list[str] | None
        Arguments to parse. `None` reads them from `sys.argv`.

    Returns
    -------
    pd.DataFrame
        Prediction table of the period.
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
    outlier_percentage = 100 * number_of_outliers / number_of_windows
    print(f"Artifact: {metadata.artifact_name}")
    print(f"Ticker: {metadata.ticker}")
    print(f"Window size: {metadata.window_size}")
    print(f"Anomaly threshold: {metadata.anomaly_threshold:.6f}")
    print(f"Period: {arguments.start_date} to {arguments.end_date}")
    print(f"Windows: {number_of_windows}")
    print(
        f"Outliers: {number_of_outliers} ({outlier_percentage:.2f}%)"
    )
    print(
        "Mean reconstruction score: "
        f"{prediction_table['reconstruction_score'].mean():.6f}"
    )
    print(f"Top {arguments.top_outliers} windows by score:")
    print(
        prediction_table.nlargest(
            arguments.top_outliers,
            "reconstruction_score",
        ).to_string()
    )
    last_window_status = (
        "OUTLIER" if prediction_table["is_outlier"].iloc[-1] else "normal"
    )
    print(
        f"Last window ({prediction_table.index[-1]}): {last_window_status}"
    )
    return prediction_table


if __name__ == "__main__":
    main()
