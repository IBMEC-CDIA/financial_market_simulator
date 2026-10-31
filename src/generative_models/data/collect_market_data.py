"""Command-line entry point that stores the prices of a ticker."""

import argparse
from pathlib import Path

from generative_models.data.market_data import (
    add_log_return_column,
    add_split_column,
    fetch_price_series,
    save_price_series,
)
from generative_models.data.validation import validate_price_dataset
from generative_models.models.autoencoder.config import MarketDataConfig


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the parser of the price collection script.

    The date range is not an option: it comes from `MarketDataConfig`,
    so it matches the range used by the training script.

    Returns
    -------
    argparse.ArgumentParser
        Parser that accepts `--ticker` and `--output-path`, both
        required.

    Example
    -------
    >>> parser = build_argument_parser()
    >>> arguments = parser.parse_args(
    ...     ["--ticker", "AAPL", "--output-path", "prices.csv"]
    ... )
    >>> arguments.ticker
    'AAPL'
    """
    parser = argparse.ArgumentParser(
        description="Download the daily prices of a ticker and save them.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol to download, for example AAPL.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="CSV file where the prices are saved.",
    )
    return parser


def main(
    command_line_arguments: list[str] | None = None,
) -> Path:
    """Download the prices of a ticker and save them to a CSV file.

    Parameters
    ----------
    command_line_arguments : list[str] | None
        Arguments to parse. When `None`, the arguments of `sys.argv`
        are used.

    Returns
    -------
    Path
        Path of the CSV file that was written. A summary line is also
        printed.

    Raises
    ------
    SystemExit
        If the arguments are missing or malformed (raised by `argparse`).

    Example
    -------
    >>> main(["--ticker", "AAPL", "--output-path", "prices.csv"])
    PosixPath('prices.csv')
    """
    parser = build_argument_parser()
    arguments = parser.parse_args(command_line_arguments)

    market_data_config = MarketDataConfig()
    raw_prices = fetch_price_series(
        arguments.ticker,
        market_data_config.start_date,
        market_data_config.end_date,
    )
    validate_price_dataset(raw_prices)
    price_data = add_split_column(
        add_log_return_column(raw_prices),
        market_data_config.train_split_ratio,
        market_data_config.validation_split_ratio,
    )
    output_path = save_price_series(price_data, arguments.output_path)

    print(
        f"ticker={arguments.ticker} "
        f"rows={len(price_data)} "
        f"output_path={output_path}"
    )

    return output_path


if __name__ == "__main__":
    main()
