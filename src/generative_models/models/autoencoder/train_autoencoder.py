"""Command line entry point of the autoencoder training pipeline."""

import argparse

from generative_models.models.autoencoder.config import (
    AutoencoderArchitectureConfig,
    AutoencoderTrainingConfig,
    MarketDataConfig,
    OptimizationConfig,
    TrackingConfig,
)
from generative_models.models.autoencoder.trainer import (
    AutoencoderTrainer,
    AutoencoderTrainingResult,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the parser of the training script.

    Every default value is read from the configuration dataclasses, so
    that the script and the configuration can never disagree.

    Returns
    -------
    argparse.ArgumentParser
        Parser accepting the required `--ticker` argument plus one
        optional argument per configuration field.

    Example
    -------
    >>> parser = build_argument_parser()
    >>> arguments = parser.parse_args(["--ticker", "AAPL"])
    >>> arguments.ticker
    'AAPL'
    """
    market_data_defaults = MarketDataConfig()
    architecture_defaults = AutoencoderArchitectureConfig()
    optimization_defaults = OptimizationConfig()
    tracking_defaults = TrackingConfig()
    training_defaults = AutoencoderTrainingConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Train the dense autoencoder that detects outlier windows "
            "in the log-return series of a ticker."
        )
    )
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument(
        "--start-date",
        type=str,
        default=market_data_defaults.start_date,
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=market_data_defaults.end_date,
    )
    parser.add_argument(
        "--train-split-ratio",
        type=float,
        default=market_data_defaults.train_split_ratio,
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=market_data_defaults.window_size,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=market_data_defaults.batch_size,
    )
    parser.add_argument(
        "--hidden-layer-size",
        type=int,
        default=architecture_defaults.hidden_layer_size,
    )
    parser.add_argument(
        "--latent-layer-size",
        type=int,
        default=architecture_defaults.latent_layer_size,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=optimization_defaults.number_of_epochs,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=optimization_defaults.learning_rate,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=optimization_defaults.random_seed,
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=training_defaults.contamination,
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=tracking_defaults.project_name,
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default=tracking_defaults.output_directory,
    )
    parser.add_argument(
        "--env-file-path",
        type=str,
        default=tracking_defaults.env_file_path,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=training_defaults.device,
    )

    return parser


def build_training_config(
    arguments: argparse.Namespace,
) -> AutoencoderTrainingConfig:
    """Assemble the training configuration from parsed arguments.

    Parameters
    ----------
    arguments : argparse.Namespace
        Namespace produced by the parser of
        `build_argument_parser`.

    Returns
    -------
    AutoencoderTrainingConfig
        Configuration holding every value given on the command line.

    Raises
    ------
    ValueError
        If any argument falls outside the range accepted by
        `AutoencoderTrainingConfig`.
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
    """Train the autoencoder of one ticker from the command line.

    Parameters
    ----------
    command_line_arguments : list[str] | None
        Arguments to parse, without the program name. When `None`,
        the arguments of `sys.argv` are used.

    Returns
    -------
    AutoencoderTrainingResult
        Result returned by the trainer, already summarized on the
        standard output.

    Example
    -------
    >>> result = main(["--ticker", "AAPL", "--epochs", "10"])
    >>> result.model_file_path.name
    'simple-autoencoder-aapl.pt'
    """
    parser = build_argument_parser()
    arguments = parser.parse_args(command_line_arguments)
    config = build_training_config(arguments)

    trainer = AutoencoderTrainer(ticker=arguments.ticker, config=config)
    result = trainer.train()

    print(
        f"{arguments.ticker} | "
        f"train MSE: {result.training_loss_history[-1]:.6f} | "
        f"validation MSE: {result.validation_loss:.6f} | "
        f"anomaly threshold: {result.anomaly_threshold:.6f} | "
        f"model: {result.model_file_path}"
    )

    return result


if __name__ == "__main__":
    main()
