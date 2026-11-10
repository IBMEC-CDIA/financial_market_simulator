"""Unit tests for the script that downloads the latest registered models."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from generative_models.models.base_autoencoder.overfitting import (
    load_overfitting_report,
)
from generative_models.serving import download_latest_models
from generative_models.serving.download_latest_models import (
    build_argument_parser,
    build_model_name,
    build_report_data,
    download_latest_model,
    main,
    parse_ticker_list,
)

PROJECT_PATH = "team/financial-market-simulator"


def _metadata(**overrides) -> dict:
    """Build the metadata of a model artifact, with optional changes."""
    data = {
        "ticker": "AAPL",
        "window_size": 10,
        "hidden_layer_size": 8,
        "latent_layer_size": 4,
        "return_train_mean": 0.001,
        "return_train_std": 0.02,
        "anomaly_threshold": 1.3,
        "final_train_loss": 0.5,
        "train_evaluation_loss": 0.4,
        "validation_loss": 0.6,
    }
    data.update(overrides)

    return data


def _build_fake_artifact(
    metadata: dict,
    weight_file_names: list[str],
) -> MagicMock:
    """Build an artifact double whose download writes the given files."""

    def download(root: str) -> str:
        """Write the fake files into the download directory."""
        for file_name in weight_file_names:
            (Path(root) / file_name).write_bytes(b"weights")

        return root

    artifact = MagicMock()
    artifact.metadata = metadata
    artifact.download.side_effect = download

    return artifact


def _registry_with(artifact: MagicMock) -> MagicMock:
    """Build a registry double that returns the given artifact."""
    registry = MagicMock()
    registry.project_path = PROJECT_PATH
    registry.api.artifact.return_value = artifact

    return registry


def test_parse_ticker_list_ignores_spaces_and_empty_items() -> None:
    """Check that the ticker list keeps the order and drops blanks."""
    assert parse_ticker_list("PETR4.SA, AAPL,") == ["PETR4.SA", "AAPL"]


def test_parse_ticker_list_requires_a_ticker() -> None:
    """Check that an empty list is refused."""
    with pytest.raises(ValueError):
        parse_ticker_list(" , ")


def test_build_model_name_uses_the_trainer_slug() -> None:
    """Check the artifact name of a ticker with a dot."""
    assert build_model_name("PETR4.SA") == "simple-autoencoder-petr4.sa"


def test_build_argument_parser_defaults_to_the_models_directory() -> None:
    """Check the default output directory of the script."""
    arguments = build_argument_parser().parse_args(["--tickers", "AAPL"])

    assert arguments.output_directory == "models"
    assert arguments.entity is None
    assert arguments.alias == "latest"


def test_build_report_data_computes_the_overfitting_ratio() -> None:
    """Check that the ratio is validation over training loss."""
    report = build_report_data(_metadata(), "simple-autoencoder-aapl.pt")

    assert report["overfitting_ratio"] == pytest.approx(1.5)
    assert report["test_loss"] is None
    assert report["baseline_test_loss"] is None
    assert report["model_file_path"] == "simple-autoencoder-aapl.pt"


def test_build_report_data_leaves_the_ratio_empty_without_training_loss() -> (
    None
):
    """Check that a zero training loss gives no ratio instead of an error."""
    report = build_report_data(
        _metadata(train_evaluation_loss=0.0),
        "simple-autoencoder-aapl.pt",
    )

    assert report["overfitting_ratio"] is None


def test_download_latest_model_writes_weights_and_report(tmp_path) -> None:
    """Check that the files are written where the local registry reads them."""
    artifact = _build_fake_artifact(_metadata(), ["weights.pt"])
    registry = _registry_with(artifact)

    report_path = download_latest_model(registry, "AAPL", tmp_path)

    registry.api.artifact.assert_called_once_with(
        f"{PROJECT_PATH}/simple-autoencoder-aapl:latest",
        type="model",
    )
    assert (tmp_path / "simple-autoencoder-aapl.pt").read_bytes() == b"weights"
    assert report_path == tmp_path / "simple-autoencoder-aapl-overfitting.json"
    report = load_overfitting_report(report_path)
    assert report.ticker == "AAPL"
    assert report.window_size == 10
    assert report.anomaly_threshold == pytest.approx(1.3)


def test_download_latest_model_refuses_incomplete_metadata(tmp_path) -> None:
    """Check that an artifact without the serving keys is refused."""
    metadata = _metadata()
    del metadata["hidden_layer_size"]
    registry = _registry_with(_build_fake_artifact(metadata, ["weights.pt"]))

    with pytest.raises(ValueError, match="hidden_layer_size"):
        download_latest_model(registry, "AAPL", tmp_path)

    assert not list(tmp_path.iterdir())


def test_download_latest_model_refuses_an_artifact_without_weights(
    tmp_path,
) -> None:
    """Check that an artifact with no `.pt` file is refused."""
    registry = _registry_with(_build_fake_artifact(_metadata(), ["notes.txt"]))

    with pytest.raises(FileNotFoundError):
        download_latest_model(registry, "AAPL", tmp_path)


def test_main_downloads_every_ticker(tmp_path, monkeypatch, capsys) -> None:
    """Check that the script downloads each ticker into the directory."""
    artifact = _build_fake_artifact(_metadata(), ["weights.pt"])
    registry = _registry_with(artifact)
    monkeypatch.setattr(
        download_latest_models,
        "WandbAutoencoderRegistry",
        MagicMock(return_value=registry),
    )
    output_directory = tmp_path / "models"

    report_paths = main(
        [
            "--tickers",
            "AAPL,MSFT",
            "--output-directory",
            str(output_directory),
        ]
    )

    assert len(report_paths) == 2
    assert all(path.exists() for path in report_paths)
    assert "Downloaded the latest model of AAPL" in capsys.readouterr().out
    assert json.loads(report_paths[0].read_text(encoding="utf-8"))["ticker"]


def test_download_model_of_an_experiment_uses_its_alias(tmp_path) -> None:
    """Check that the alias of the experiment selects the artifact version."""
    artifact = _build_fake_artifact(_metadata(), ["weights.pt"])
    registry = _registry_with(artifact)

    download_latest_model(
        registry,
        "AAPL",
        tmp_path,
        alias="aapl-baseline",
    )

    registry.api.artifact.assert_called_once_with(
        f"{PROJECT_PATH}/simple-autoencoder-aapl:aapl-baseline",
        type="model",
    )


def test_main_passes_the_alias_to_every_download(
    tmp_path, monkeypatch
) -> None:
    """Check that the --alias option is used for each ticker."""
    artifact = _build_fake_artifact(_metadata(), ["weights.pt"])
    registry = _registry_with(artifact)
    monkeypatch.setattr(
        download_latest_models,
        "WandbAutoencoderRegistry",
        MagicMock(return_value=registry),
    )

    main(
        [
            "--tickers",
            "AAPL,MSFT",
            "--output-directory",
            str(tmp_path / "models"),
            "--alias",
            "aapl-baseline",
        ]
    )

    requested_paths = [
        call.args[0] for call in registry.api.artifact.call_args_list
    ]
    assert requested_paths == [
        f"{PROJECT_PATH}/simple-autoencoder-aapl:aapl-baseline",
        f"{PROJECT_PATH}/simple-autoencoder-msft:aapl-baseline",
    ]
