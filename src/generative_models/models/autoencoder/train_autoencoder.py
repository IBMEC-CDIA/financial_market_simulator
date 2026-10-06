"""Command-line script to train the simple autoencoder for a ticker.

Example
-------
Train with the default configuration::

    uv run python -m generative_models.models.autoencoder.train_autoencoder \
        --ticker AAPL

Override part of the configuration::

    uv run python -m generative_models.models.autoencoder.train_autoencoder \
        --ticker PETR4.SA --start-date 2018-01-01 --epochs 100
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
    """Build the command-line parser for the training script.

    Every optional argument defaults to the value defined in the
    corresponding configuration dataclass.

    Returns
    -------
    argparse.ArgumentParser
        Parser with a required `--ticker` argument and optional
        overrides for the training configuration.
    """
    market_data_defaults = MarketDataConfig()
    architecture_defaults = AutoencoderArchitectureConfig()
    optimization_defaults = OptimizationConfig()
    tracking_defaults = TrackingConfig()
    training_defaults = AutoencoderTrainingConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Train a simple dense autoencoder on the daily log-returns "
            "of a ticker and log the experiment to wandb."
        ),
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument(
        "--start-date",
        default=market_data_defaults.start_date,
    )
    parser.add_argument(
        "--end-date",
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
        default=tracking_defaults.project_name,
    )
    parser.add_argument(
        "--output-directory",
        default=tracking_defaults.output_directory,
    )
    parser.add_argument(
        "--env-file-path",
        default=tracking_defaults.env_file_path,
    )
    parser.add_argument("--device", default=training_defaults.device)
    return parser


def build_training_config(
    arguments: argparse.Namespace,
) -> AutoencoderTrainingConfig:
    """Build the training configuration from parsed arguments.

    Parameters
    ----------
    arguments : argparse.Namespace
        Arguments parsed by the parser returned by
        `build_argument_parser`.

    Returns
    -------
    AutoencoderTrainingConfig
        Configuration populated with the parsed values.
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
    """Parse the arguments, train the autoencoder and report results.

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
    >>> result = main(["--ticker", "AAPL", "--epochs", "10"])
    """
    arguments = build_argument_parser().parse_args(command_line_arguments)
    training_config = build_training_config(arguments)

    trainer = AutoencoderTrainer(
        ticker=arguments.ticker,
        config=training_config,
    )
    training_result = trainer.train()

    print(
        f"Ticker: {arguments.ticker} | "
        f"Final train MSE: {training_result.training_loss_history[-1]:.6f}"
        f" | Validation MSE: {training_result.validation_loss:.6f} | "
        f"Anomaly threshold: {training_result.anomaly_threshold:.6f} | "
        f"Model saved to: {training_result.model_file_path}"
    )
    return training_result


if __name__ == "__main__":
    main()
