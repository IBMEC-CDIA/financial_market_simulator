"""Command-line entry point for training the dense autoencoder."""

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
    """Build the parser of the training command-line arguments.

    Default values are read from the configuration dataclasses.

    Returns
    -------
    argparse.ArgumentParser
        Parser with `--ticker` (required) and one optional argument
        per configuration field.
    """
    market_data_defaults = MarketDataConfig()
    architecture_defaults = AutoencoderArchitectureConfig()
    optimization_defaults = OptimizationConfig()
    tracking_defaults = TrackingConfig()
    training_defaults = AutoencoderTrainingConfig()

    parser = argparse.ArgumentParser(
        description="Train a dense autoencoder outlier detector.",
    )
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument(
        "--start-date", type=str, default=market_data_defaults.start_date
    )
    parser.add_argument(
        "--end-date", type=str, default=market_data_defaults.end_date
    )
    parser.add_argument(
        "--train-split-ratio",
        type=float,
        default=market_data_defaults.train_split_ratio,
    )
    parser.add_argument(
        "--window-size", type=int, default=market_data_defaults.window_size
    )
    parser.add_argument(
        "--batch-size", type=int, default=market_data_defaults.batch_size
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
        "--random-seed", type=int, default=optimization_defaults.random_seed
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=training_defaults.contamination,
    )
    parser.add_argument(
        "--project-name", type=str, default=tracking_defaults.project_name
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default=tracking_defaults.output_directory,
    )
    parser.add_argument(
        "--env-file-path", type=str, default=tracking_defaults.env_file_path
    )
    parser.add_argument(
        "--device", type=str, default=training_defaults.device
    )
    return parser


def build_training_config(
    arguments: argparse.Namespace,
) -> AutoencoderTrainingConfig:
    """Build the training configuration from parsed arguments.

    Parameters
    ----------
    arguments : argparse.Namespace
        Arguments parsed by `build_argument_parser`.

    Returns
    -------
    AutoencoderTrainingConfig
        Validated configuration.

    Raises
    ------
    ValueError
        If any value fails the configuration validation.
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
    """Parse the arguments, train the autoencoder and print a summary.

    Parameters
    ----------
    command_line_arguments : list[str] | None
        Arguments to parse. When `None`, `sys.argv` is used.

    Returns
    -------
    AutoencoderTrainingResult
        Result of the training run.

    Example
    -------
    >>> result = main(["--ticker", "AAPL", "--epochs", "100"])
    """
    arguments = build_argument_parser().parse_args(command_line_arguments)
    config = build_training_config(arguments)
    trainer = AutoencoderTrainer(ticker=arguments.ticker, config=config)
    result = trainer.train()

    final_train_loss = (
        result.training_loss_history[-1]
        if result.training_loss_history
        else float("nan")
    )
    print(
        f"ticker={arguments.ticker} "
        f"final_train_loss={final_train_loss:.6f} "
        f"validation_loss={result.validation_loss:.6f} "
        f"anomaly_threshold={result.anomaly_threshold:.6f} "
        f"model_file_path={result.model_file_path}"
    )
    return result


if __name__ == "__main__":
    main()
