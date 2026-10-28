"""Command line entry point for training the simple autoencoder.

Example
-------
uv run python -m generative_models.models.autoencoder.train_autoencoder \\
    --ticker AAPL --epochs 100
"""

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
    """Create the parser of the training command line arguments.

    The default value of every optional argument is read from the
    configuration dataclasses, so the defaults live in a single place.

    Returns
    -------
    argparse.ArgumentParser
        Parser with a required `--ticker` argument and one optional
        argument per training configuration field.
    """
    default_market_data = MarketDataConfig()
    default_architecture = AutoencoderArchitectureConfig()
    default_optimization = OptimizationConfig()
    default_tracking = TrackingConfig()
    default_training_config = AutoencoderTrainingConfig()

    argument_parser = argparse.ArgumentParser(
        description=(
            "Train a SimpleAutoencoder on the log-returns of a ticker "
            "and register it in wandb."
        ),
    )
    argument_parser.add_argument("--ticker", type=str, required=True)
    argument_parser.add_argument(
        "--start-date",
        type=str,
        default=default_market_data.start_date,
    )
    argument_parser.add_argument(
        "--end-date",
        type=str,
        default=default_market_data.end_date,
    )
    argument_parser.add_argument(
        "--train-split-ratio",
        type=float,
        default=default_market_data.train_split_ratio,
    )
    argument_parser.add_argument(
        "--window-size",
        type=int,
        default=default_market_data.window_size,
    )
    argument_parser.add_argument(
        "--batch-size",
        type=int,
        default=default_market_data.batch_size,
    )
    argument_parser.add_argument(
        "--hidden-layer-size",
        type=int,
        default=default_architecture.hidden_layer_size,
    )
    argument_parser.add_argument(
        "--latent-layer-size",
        type=int,
        default=default_architecture.latent_layer_size,
    )
    argument_parser.add_argument(
        "--epochs",
        type=int,
        default=default_optimization.number_of_epochs,
    )
    argument_parser.add_argument(
        "--learning-rate",
        type=float,
        default=default_optimization.learning_rate,
    )
    argument_parser.add_argument(
        "--random-seed",
        type=int,
        default=default_optimization.random_seed,
    )
    argument_parser.add_argument(
        "--contamination",
        type=float,
        default=default_training_config.contamination,
    )
    argument_parser.add_argument(
        "--project-name",
        type=str,
        default=default_tracking.project_name,
    )
    argument_parser.add_argument(
        "--output-directory",
        type=str,
        default=default_tracking.output_directory,
    )
    argument_parser.add_argument(
        "--env-file-path",
        type=str,
        default=default_tracking.env_file_path,
    )
    argument_parser.add_argument(
        "--device",
        type=str,
        default=default_training_config.device,
    )
    return argument_parser


def build_training_config(
    arguments: argparse.Namespace,
) -> AutoencoderTrainingConfig:
    """Build the training configuration from parsed arguments.

    Parameters
    ----------
    arguments : argparse.Namespace
        Arguments parsed by the parser of `build_argument_parser`.

    Returns
    -------
    AutoencoderTrainingConfig
        Validated training configuration.

    Raises
    ------
    ValueError
        If any argument value fails the configuration validation.
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
    """Parse the arguments, train the autoencoder and report the result.

    Parameters
    ----------
    command_line_arguments : list[str] | None
        Arguments to parse. When `None`, `sys.argv` is used.

    Returns
    -------
    AutoencoderTrainingResult
        Result of the training run.

    Raises
    ------
    SystemExit
        If `--ticker` is missing or an argument cannot be parsed.
    ValueError
        If the resulting configuration is invalid.

    Example
    -------
    >>> training_result = main(["--ticker", "AAPL", "--epochs", "100"])
    """
    arguments = build_argument_parser().parse_args(command_line_arguments)
    training_config = build_training_config(arguments)

    trainer = AutoencoderTrainer(
        ticker=arguments.ticker,
        config=training_config,
    )
    training_result = trainer.train()

    print(
        f"ticker={arguments.ticker} | "
        f"final_train_loss={training_result.training_loss_history[-1]} | "
        f"validation_loss={training_result.validation_loss} | "
        f"anomaly_threshold={training_result.anomaly_threshold} | "
        f"model_file_path={training_result.model_file_path}"
    )
    return training_result


if __name__ == "__main__":
    main()
