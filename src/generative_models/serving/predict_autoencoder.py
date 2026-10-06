"""Command-line script that loads an autoencoder from wandb and predicts.

The most recently trained autoencoder is loaded by default; pass
`--ticker` to load the latest model of a specific ticker instead. The
script then fetches recent prices for the model ticker, scores every
window and prints a summary, so the loaded model can be validated.

Example
-------
Load the most recent model and predict over the last year::

    uv run python -m generative_models.serving.predict_autoencoder

Load the model of a given ticker over a custom period::

    uv run python -m generative_models.serving.predict_autoencoder \
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
    """Build the command-line parser for the prediction script.

    Returns
    -------
    argparse.ArgumentParser
        Parser with optional ticker, wandb, period and report
        arguments.
    """
    today = date.today()
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained autoencoder from wandb and detect outliers "
            "in recent log-returns."
        ),
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Ticker to load. Defaults to the most recent model.",
    )
    parser.add_argument(
        "--project-name",
        default=TrackingConfig.project_name,
    )
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--start-date",
        default=(today - timedelta(days=365)).isoformat(),
    )
    parser.add_argument("--end-date", default=today.isoformat())
    parser.add_argument("--top-outliers", type=int, default=5)
    parser.add_argument("--device", default=None)
    return parser


def build_prediction_table(
    price_data: pd.DataFrame,
    prediction: AutoencoderPrediction,
) -> pd.DataFrame:
    """Align each window prediction with its end date and price.

    Parameters
    ----------
    price_data : pd.DataFrame
        Price data frame indexed by date with a `close_price` column,
        as returned by `fetch_price_series`.
    prediction : AutoencoderPrediction
        Prediction computed from the prices in `price_data`.

    Returns
    -------
    pd.DataFrame
        One row per window, indexed by the window end date, with the
        `close_price`, `reconstruction_score` and `is_outlier`
        columns.
    """
    price_row_indices = prediction.window_end_indices + 1
    return pd.DataFrame(
        {
            "close_price": price_data["close_price"]
            .to_numpy()
            .flatten()[price_row_indices],
            "reconstruction_score": prediction.reconstruction_scores,
            "is_outlier": prediction.is_outlier,
        },
        index=price_data.index[price_row_indices],
    )


def load_predictor(
    registry: WandbAutoencoderRegistry,
    ticker: str | None = None,
) -> AutoencoderPredictor:
    """Load a predictor from wandb, by ticker or the most recent one.

    Parameters
    ----------
    registry : WandbAutoencoderRegistry
        Registry used to download the model artifact.
    ticker : str | None
        Ticker whose latest model is loaded. When `None`, the most
        recently trained model of any ticker is loaded.

    Returns
    -------
    AutoencoderPredictor
        Predictor built from the selected artifact.
    """
    if ticker:
        return registry.load_model(ticker)
    return registry.load_latest_model()


def predict_period(
    predictor: AutoencoderPredictor,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch the model ticker prices for a period and score them.

    Parameters
    ----------
    predictor : AutoencoderPredictor
        Loaded predictor; its metadata defines the ticker.
    start_date : str
        Start date of the period, in `"YYYY-MM-DD"` format.
    end_date : str
        End date of the period, in `"YYYY-MM-DD"` format.

    Returns
    -------
    pd.DataFrame
        Prediction table built by `build_prediction_table`.

    Example
    -------
    >>> prediction_table = predict_period(
    ...     predictor,
    ...     start_date="2026-01-01",
    ...     end_date="2026-06-30",
    ... )
    >>> list(prediction_table.columns)
    ['close_price', 'reconstruction_score', 'is_outlier']
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
    """Load the model, run predictions and print a validation report.

    Parameters
    ----------
    command_line_arguments : list[str] | None
        Arguments to parse. When `None`, `sys.argv` is used.

    Returns
    -------
    pd.DataFrame
        Prediction table built by `build_prediction_table`.

    Example
    -------
    >>> prediction_table = main(["--ticker", "PETR4.SA"])
    >>> prediction_table["is_outlier"].sum()
    11
    """
    arguments = build_argument_parser().parse_args(command_line_arguments)

    registry = WandbAutoencoderRegistry(
        project_name=arguments.project_name,
        entity=arguments.entity,
        device=arguments.device,
    )
    predictor = load_predictor(registry, arguments.ticker)
    metadata = predictor.metadata

    prediction_table = predict_period(
        predictor,
        arguments.start_date,
        arguments.end_date,
    )

    outlier_table = prediction_table[prediction_table["is_outlier"]]
    latest_window = prediction_table.iloc[-1]
    print(
        f"\nModel: {metadata.artifact_name} | Ticker: {metadata.ticker} | "
        f"Window size: {metadata.window_size} | "
        f"Anomaly threshold: {metadata.anomaly_threshold:.6f}"
    )
    print(
        f"Period: {arguments.start_date} to {arguments.end_date} | "
        f"Windows scored: {len(prediction_table)} | "
        f"Outliers: {len(outlier_table)} "
        f"({len(outlier_table) / len(prediction_table):.1%})"
    )
    print(
        f"Mean reconstruction score: "
        f"{prediction_table['reconstruction_score'].mean():.6f}"
    )
    print(f"\nTop {arguments.top_outliers} windows by reconstruction score:")
    print(
        prediction_table.nlargest(
            arguments.top_outliers,
            "reconstruction_score",
        ).to_string()
    )
    print(
        f"\nLatest window ({prediction_table.index[-1].date()}): "
        f"score {latest_window['reconstruction_score']:.6f} -> "
        f"{'OUTLIER' if latest_window['is_outlier'] else 'normal'}"
    )
    return prediction_table


if __name__ == "__main__":
    main()
