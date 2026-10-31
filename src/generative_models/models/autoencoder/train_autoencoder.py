"""Command-line entry point for training the autoencoder of a ticker."""

import argparse

from generative_models.data.market_data import load_price_series
from generative_models.models.autoencoder.config import (
    AutoencoderArchitectureConfig,
    AutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
    TrackingConfig,
)
from generative_models.models.autoencoder.overfitting import (
    build_overfitting_report,
    build_report_path,
    save_overfitting_report,
)
from generative_models.models.autoencoder.trainer import (
    AutoencoderTrainer,
    AutoencoderTrainingResult,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the parser with every training option.

    The default values are read from the configuration dataclasses, so
    they are defined in a single place.

    Returns
    -------
    argparse.ArgumentParser
        Parser that accepts the `--ticker` option (required) and the
        optional hyperparameters.

    Example
    -------
    >>> parser = build_argument_parser()
    >>> parser.parse_args(["--ticker", "AAPL", "--epochs", "50"]).epochs
    50
    """
    market_data_defaults = MarketDataConfig()
    architecture_defaults = AutoencoderArchitectureConfig()
    optimization_defaults = OptimizationConfig()
    tracking_defaults = TrackingConfig()
    training_defaults = AutoencoderTrainingConfig()

    parser = argparse.ArgumentParser(
        description="Train the autoencoder anomaly detector for a ticker.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol to train on, for example AAPL.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=market_data_defaults.start_date,
        help="First date of the historical range (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=market_data_defaults.end_date,
        help="Last date of the historical range (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--train-split-ratio",
        type=float,
        default=market_data_defaults.train_split_ratio,
        help="Fraction of the log-returns used for training.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=market_data_defaults.window_size,
        help="Number of log-returns in each window.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=market_data_defaults.batch_size,
        help="Number of windows per training batch.",
    )
    parser.add_argument(
        "--hidden-layer-size",
        type=int,
        default=architecture_defaults.hidden_layer_size,
        help="Number of units in the hidden layers.",
    )
    parser.add_argument(
        "--latent-layer-size",
        type=int,
        default=architecture_defaults.latent_layer_size,
        help="Size of the latent representation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=optimization_defaults.number_of_epochs,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=optimization_defaults.learning_rate,
        help="Learning rate of the Adam optimizer.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=optimization_defaults.random_seed,
        help="Seed for weight initialization and batch shuffling.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=training_defaults.contamination,
        help="Expected fraction of anomalous windows.",
    )
    parser.add_argument(
        "--price-data-path",
        type=str,
        default=None,
        help=(
            "CSV file with prices saved by collect_market_data. When given, "
            "the prices are read from it instead of being downloaded, and "
            "--start-date and --end-date are not used."
        ),
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=tracking_defaults.project_name,
        help="Name of the wandb project.",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default=tracking_defaults.output_directory,
        help="Directory where the trained weights are saved.",
    )
    parser.add_argument(
        "--env-file-path",
        type=str,
        default=tracking_defaults.env_file_path,
        help="Path to the .env file that contains WANDB_API_KEY.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=training_defaults.device,
        help="Device for training, for example cpu or cuda.",
    )

    return parser


def build_training_config(
    arguments: argparse.Namespace,
) -> AutoencoderTrainingConfig:
    """Convert parsed command-line arguments into a training configuration.

    Parameters
    ----------
    arguments : argparse.Namespace
        Namespace returned by the parser of `build_argument_parser`.

    Returns
    -------
    AutoencoderTrainingConfig
        Configuration built from the arguments. Invalid values raise the
        `ValueError` raised by the configuration validation.

    Raises
    ------
    ValueError
        If a value is outside its valid range, for example
        `--epochs 0`.

    Example
    -------
    >>> arguments = build_argument_parser().parse_args(
    ...     ["--ticker", "AAPL", "--window-size", "20"]
    ... )
    >>> build_training_config(arguments).market_data.window_size
    20
    """
    return AutoencoderTrainingConfig(
        market_data=MarketDataConfig(
            start_date=arguments.start_date,
            end_date=arguments.end_date,
            train_split_ratio=arguments.train_split_ratio,
            window_size=arguments.window_size,
            batch_size=arguments.batch_size,
        ),
        architecture=AutoencoderArchitectureConfig(
            hidden_layer_size=arguments.hidden_layer_size,
            latent_layer_size=arguments.latent_layer_size,
        ),
        optimization=OptimizationConfig(
            number_of_epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            random_seed=arguments.random_seed,
        ),
        tracking=TrackingConfig(
            project_name=arguments.project_name,
            output_directory=arguments.output_directory,
            env_file_path=arguments.env_file_path,
        ),
        contamination=arguments.contamination,
        device=arguments.device,
    )


def main(
    command_line_arguments: list[str] | None = None,
) -> AutoencoderTrainingResult:
    """Train the autoencoder of a ticker from the command line.

    Parameters
    ----------
    command_line_arguments : list[str] | None
        Arguments to parse. When `None`, the arguments of `sys.argv`
        are used.

    Returns
    -------
    AutoencoderTrainingResult
        Result of the training run. A summary line is also printed.

    Raises
    ------
    SystemExit
        If the arguments are missing or malformed (raised by `argparse`).
    ValueError
        If a value is outside its valid range. This is raised before the
        trainer is created.

    Example
    -------
    >>> result = main(["--ticker", "AAPL", "--epochs", "2"])
    """
    parser = build_argument_parser()
    arguments = parser.parse_args(command_line_arguments)
    training_config = build_training_config(arguments)

    price_data = None
    if arguments.price_data_path is not None:
        price_data = load_price_series(arguments.price_data_path)

    trainer = AutoencoderTrainer(
        ticker=arguments.ticker,
        config=training_config,
        price_data=price_data,
    )
    result = trainer.train()
    report = build_overfitting_report(arguments.ticker, result)
    report_path = save_overfitting_report(
        report,
        build_report_path(result.model_file_path),
    )

    print(
        f"ticker={arguments.ticker} "
        f"final_train_loss={result.training_loss_history[-1]:.6f} "
        f"validation_loss={result.validation_loss:.6f} "
        f"anomaly_threshold={result.anomaly_threshold:.6f} "
        f"model_file_path={result.model_file_path} "
        f"report_path={report_path}"
    )

    return result


if __name__ == "__main__":
    main()
