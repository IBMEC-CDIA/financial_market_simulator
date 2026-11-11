"""Command-line entry point that trains the model of several tickers.

The model is chosen with `--model`, and it must be a package in
`generative_models.models` with a `trainer_factory` and an `overfitting`
module. The same command serves every model, so a model needs no copy of it.
"""

import argparse
import importlib
from functools import partial
from pathlib import Path
from types import ModuleType

from generative_models.models.model_names import (
    derive_experiment_name,
    validate_model_name,
)
from generative_models.models.ticker_model_manager import TickerModelManager
from generative_models.models.training_options import TrainingOptions

DEFAULT_MODEL = "base_autoencoder"


def parse_ticker_list(raw_tickers: str) -> list[str]:
    """Split a comma-separated list of tickers.

    Parameters
    ----------
    raw_tickers : str
        Ticker symbols separated by commas, for example `"AAPL, MSFT"`.

    Returns
    -------
    list[str]
        Ticker symbols without surrounding spaces, in the given order.
        Empty items are removed.

    Example
    -------
    >>> parse_ticker_list("AAPL, MSFT,")
    ['AAPL', 'MSFT']
    """
    tickers = [ticker.strip() for ticker in raw_tickers.split(",")]

    return [ticker for ticker in tickers if ticker]


def load_model_module(model_name: str, module_name: str) -> ModuleType:
    """Import a module of a model package.

    Parameters
    ----------
    model_name : str
        Name of the model package.
    module_name : str
        Module inside the package, for example `"trainer_factory"`.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ValueError
        If the model name is not valid.
    ModuleNotFoundError
        If the package or the module does not exist.
    """
    validate_model_name(model_name)

    return importlib.import_module(
        f"generative_models.models.{model_name}.{module_name}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the parser with the model, the tickers and the options.

    Returns
    -------
    argparse.ArgumentParser
        Parser that accepts `--tickers` (required), `--model` and the
        options shared by every model, with the defaults of
        `TrainingOptions`.
    """
    defaults = TrainingOptions()

    parser = argparse.ArgumentParser(
        description="Train the model of several tickers.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model package in generative_models.models to train.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated ticker symbols, for example AAPL,MSFT.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=defaults.start_date,
        help="First date of the historical range (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=defaults.end_date,
        help="Last date of the historical range (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=defaults.window_size,
        help="Number of log-returns in each window.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults.epochs,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.learning_rate,
        help="Learning rate of the optimizer.",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default=defaults.output_directory,
        help="Directory where the weights and reports are saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.device,
        help="Device for training, for example cpu or cuda.",
    )
    return parser


def build_training_options(arguments: argparse.Namespace) -> TrainingOptions:
    """Convert the parsed arguments into the shared training options.

    Parameters
    ----------
    arguments : argparse.Namespace
        Namespace returned by the parser of `build_argument_parser`.

    Returns
    -------
    TrainingOptions
        Options shared by every ticker of the run.
    """
    return TrainingOptions(
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        window_size=arguments.window_size,
        output_directory=arguments.output_directory,
        device=arguments.device,
        experiment_name=derive_experiment_name(arguments.model),
    )


def build_tickers_configuration(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, dict[str, str]]:
    """Give every ticker the same date range, as the manager expects.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols of the run.
    start_date : str
        First date of the historical range, in `"YYYY-MM-DD"` format.
    end_date : str
        Last date of the historical range, in `"YYYY-MM-DD"` format.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping from each ticker to its `start_date` and `end_date`.
    """
    return {
        ticker: {"start_date": start_date, "end_date": end_date}
        for ticker in tickers
    }


def save_ticker_report(
    model_name: str,
    ticker: str,
    result: object,
) -> Path:
    """Write the overfitting report of one ticker with the model's module.

    Parameters
    ----------
    model_name : str
        Name of the model package that produced the result.
    ticker : str
        Ticker symbol the model was trained on.
    result : object
        Result returned by the trainer of the ticker.

    Returns
    -------
    Path
        Path of the overfitting report that was written.
    """
    overfitting = load_model_module(model_name, "overfitting")

    return overfitting.save_overfitting_report(
        overfitting.build_overfitting_report(ticker, result),
        overfitting.build_report_path(result.model_file_path),
    )


def main(
    command_line_arguments: list[str] | None = None,
) -> dict[str, object]:
    """Train the model of every ticker and save the reports.

    Parameters
    ----------
    command_line_arguments : list[str] | None
        Arguments to parse. When `None`, the arguments of `sys.argv`
        are used.

    Returns
    -------
    dict[str, object]
        Training result of each ticker, in the order they were given.

    Example
    -------
    >>> main(["--model", "base_autoencoder", "--tickers", "AAPL,MSFT"])
    """
    arguments = build_argument_parser().parse_args(command_line_arguments)
    validate_model_name(arguments.model)
    tickers = parse_ticker_list(arguments.tickers)
    options = build_training_options(arguments)
    trainer_factory = load_model_module(
        arguments.model,
        "trainer_factory",
    ).build_trainer

    manager = TickerModelManager(
        tickers=build_tickers_configuration(
            tickers,
            options.start_date,
            options.end_date,
        ),
        trainer_factory=partial(trainer_factory, options=options),
    )
    manager.train_all_models()

    results = {}
    for ticker in tickers:
        result = manager.get_model(ticker)
        save_ticker_report(arguments.model, ticker, result)
        results[ticker] = result
        print(
            f"model={arguments.model} "
            f"ticker={ticker} "
            f"validation_loss={result.validation_loss:.6f} "
            f"anomaly_threshold={result.anomaly_threshold:.6f} "
            f"model_file_path={result.model_file_path}"
        )

    return results


if __name__ == "__main__":
    main()
