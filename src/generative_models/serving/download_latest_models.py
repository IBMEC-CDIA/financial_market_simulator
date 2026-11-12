"""Command-line script that downloads the latest registered models.

The script fetches the version of the model artifact of each ticker that is
registered under the experiment of the model, from wandb, and writes the
files that the local serving registry
reads: the weights `<name>.pt` and the report `<name>-overfitting.json`.
It is used by the build workflow before the Docker image is built.
"""

# pylint: disable=duplicate-code

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from generative_models.models.base_autoencoder.trainer import (
    MODEL_NAME_PREFIX,
    build_ticker_slug,
)
from generative_models.models.model_names import (
    derive_experiment_name,
    load_artifact_prefix,
    load_project_name,
)
from generative_models.serving.model_families import (
    find_missing_serving_keys,
    resolve_model_family,
)
from generative_models.serving.wandb_autoencoder_registry import (
    WandbAutoencoderRegistry,
)

REPORT_SUFFIX = "-overfitting.json"
DEFAULT_OUTPUT_DIRECTORY = "models"
DEFAULT_ALIAS = "latest"


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the parser of the download script.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the options `--tickers`, `--model`,
        `--output-directory` and `--entity`.

    Example
    -------
    >>> arguments = build_argument_parser().parse_args(
    ...     ["--tickers", "AAPL,MSFT"]
    ... )
    >>> arguments.output_directory
    'models'
    """
    parser = argparse.ArgumentParser(
        description="Download the latest registered autoencoders from wandb.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated ticker symbols, for example PETR4.SA,AAPL.",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory where the weights and reports are written.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="wandb entity (user or team). Defaults to the account entity.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model package whose models are downloaded, for example "
            "convolutional_autoencoder. Its experiment name and artifact "
            "prefix are derived from it."
        ),
    )

    return parser


def parse_ticker_list(ticker_text: str) -> list[str]:
    """Split a comma-separated list of tickers.

    Parameters
    ----------
    ticker_text : str
        Tickers separated by commas. Spaces around each ticker and empty
        items are ignored.

    Returns
    -------
    list[str]
        Tickers in the order they were given.

    Raises
    ------
    ValueError
        If no ticker is given.

    Example
    -------
    >>> parse_ticker_list("PETR4.SA, AAPL,")
    ['PETR4.SA', 'AAPL']
    """
    tickers = [
        ticker.strip() for ticker in ticker_text.split(",") if ticker.strip()
    ]
    if not tickers:
        raise ValueError("At least one ticker is required.")

    return tickers


def build_model_name(
    ticker: str,
    name_prefix: str = MODEL_NAME_PREFIX,
) -> str:
    """Return the artifact and file name of the model of a ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol, for example `"PETR4.SA"`.
    name_prefix : str
        Prefix of the model family, for example `"simple-autoencoder-"`
        or `"convolutional-autoencoder-"`. It must match the prefix used
        by the trainer of that model.

    Returns
    -------
    str
        Name in the form `<name_prefix><ticker_slug>`.

    Example
    -------
    >>> build_model_name("PETR4.SA")
    'simple-autoencoder-petr4.sa'
    >>> build_model_name("PETR4.SA", "convolutional-autoencoder-")
    'convolutional-autoencoder-petr4.sa'
    """
    return name_prefix + build_ticker_slug(ticker)


def build_report_data(
    metadata: dict,
    model_file_name: str,
) -> dict:
    """Build the overfitting report of a model from its artifact metadata.

    The artifact does not store the test losses, so `test_loss` and
    `baseline_test_loss` are `None`. The overfitting ratio is computed from
    the validation and training losses, and is `None` when the training
    loss is zero or missing.

    Parameters
    ----------
    metadata : dict
        Metadata of the model artifact, as logged by the trainer.
    model_file_name : str
        File name of the weights, stored in `model_file_path`.

    Returns
    -------
    dict
        Values of an `OverfittingReport`, ready to be saved as JSON.

    Example
    -------
    >>> report = build_report_data(metadata, "simple-autoencoder-aapl.pt")
    >>> report["test_loss"] is None
    True
    """
    train_evaluation_loss = metadata.get("train_evaluation_loss")
    validation_loss = metadata.get("validation_loss")
    overfitting_ratio = None
    if train_evaluation_loss and validation_loss is not None:
        overfitting_ratio = validation_loss / train_evaluation_loss

    return {
        "ticker": metadata["ticker"],
        "model_file_path": model_file_name,
        "final_train_loss": metadata.get("final_train_loss"),
        "train_evaluation_loss": train_evaluation_loss,
        "validation_loss": validation_loss,
        "overfitting_ratio": overfitting_ratio,
        "test_loss": None,
        "baseline_test_loss": None,
        "window_size": metadata["window_size"],
        "hidden_layer_size": metadata.get("hidden_layer_size"),
        "latent_layer_size": metadata.get("latent_layer_size"),
        "return_train_mean": metadata["return_train_mean"],
        "return_train_std": metadata["return_train_std"],
        "anomaly_threshold": metadata["anomaly_threshold"],
        "model_family": resolve_model_family(metadata),
        "architecture": metadata.get("architecture"),
    }


def download_latest_model(
    registry: WandbAutoencoderRegistry,
    ticker: str,
    output_directory: Path,
    alias: str = DEFAULT_ALIAS,
    name_prefix: str = MODEL_NAME_PREFIX,
) -> Path:
    """Download the model of a ticker with an alias and write its files.

    Parameters
    ----------
    registry : WandbAutoencoderRegistry
        Registry whose wandb client and project are used.
    ticker : str
        Ticker symbol of the model.
    output_directory : Path
        Directory where `<name>.pt` and `<name>-overfitting.json` are
        written. It must already exist.
    alias : str
        Alias of the model version to download, `"latest"` by default.
        An experiment name selects the models registered under it.
    name_prefix : str
        Prefix of the artifact name of the model family, for example
        `"convolutional-autoencoder-"`. Defaults to the simple
        autoencoder prefix.

    Returns
    -------
    Path
        Path of the written report, `<name>-overfitting.json`.

    Raises
    ------
    ValueError
        If the artifact lacks metadata required for inference.
    FileNotFoundError
        If the artifact contains no `.pt` file.

    Example
    -------
    >>> report_path = download_latest_model(
    ...     registry,
    ...     "AAPL",
    ...     Path("models"),
    ...     alias="aapl-baseline",
    ... )
    """
    model_name = build_model_name(ticker, name_prefix)
    artifact_path = f"{registry.project_path}/{model_name}:{alias}"
    artifact = registry.api.artifact(artifact_path, type="model")

    missing_keys = find_missing_serving_keys(artifact.metadata)
    if missing_keys:
        raise ValueError(
            f"Artifact '{artifact_path}' is missing metadata required "
            f"for inference: {', '.join(missing_keys)}."
        )

    with tempfile.TemporaryDirectory() as download_directory:
        artifact.download(root=download_directory)
        weight_paths = sorted(Path(download_directory).glob("*.pt"))
        if not weight_paths:
            raise FileNotFoundError(
                f"Artifact '{artifact_path}' has no .pt weights file."
            )
        weight_file_name = f"{model_name}.pt"
        shutil.copyfile(
            weight_paths[0],
            output_directory / weight_file_name,
        )

    report_data = build_report_data(artifact.metadata, weight_file_name)
    report_path = output_directory / f"{model_name}{REPORT_SUFFIX}"
    report_path.write_text(
        json.dumps(report_data, indent=2),
        encoding="utf-8",
    )

    return report_path


def main(
    command_line_arguments: list[str] | None = None,
) -> list[Path]:
    """Download the latest model of every ticker given on the command line.

    Parameters
    ----------
    command_line_arguments : list[str] | None
        Arguments to parse. When `None`, the arguments of `sys.argv` are
        used.

    Returns
    -------
    list[Path]
        Paths of the written reports, in the order of the tickers.

    Raises
    ------
    SystemExit
        If the arguments are malformed (raised by `argparse`).

    Example
    -------
    >>> main(["--tickers", "AAPL,MSFT", "--output-directory", "models"])
    """
    arguments = build_argument_parser().parse_args(command_line_arguments)
    tickers = parse_ticker_list(arguments.tickers)
    experiment_name = derive_experiment_name(arguments.model)
    name_prefix = load_artifact_prefix(arguments.model)
    project_name = load_project_name(arguments.model)
    output_directory = Path(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    registry = WandbAutoencoderRegistry(
        project_name=project_name,
        entity=arguments.entity,
    )

    report_paths = []
    for ticker in tickers:
        report_path = download_latest_model(
            registry,
            ticker,
            output_directory,
            alias=experiment_name,
            name_prefix=name_prefix,
        )
        print(f"Downloaded the latest model of {ticker}: {report_path}")
        report_paths.append(report_path)

    return report_paths


if __name__ == "__main__":
    main()
