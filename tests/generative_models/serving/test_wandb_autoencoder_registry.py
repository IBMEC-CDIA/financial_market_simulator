"""Unit tests for WandbAutoencoderRegistry."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder
from generative_models.serving.autoencoder_predictor import (
    AutoencoderPredictor,
)
from generative_models.serving.wandb_autoencoder_registry import (
    WandbAutoencoderRegistry,
)

API_PATH = "generative_models.serving.wandb_autoencoder_registry.wandb.Api"
PROJECT_PATH = "test-entity/test-project"


def _make_metadata(ticker: str) -> dict[str, object]:
    """Build the artifact metadata logged by AutoencoderTrainer."""
    return {
        "ticker": ticker,
        "window_size": 5,
        "hidden_layer_size": 8,
        "latent_layer_size": 2,
        "return_train_mean": 0.001,
        "return_train_std": 0.02,
        "anomaly_threshold": 1.5,
    }


def _make_artifact(
    name: str,
    created_at: str,
    metadata: dict[str, object],
) -> MagicMock:
    """Build a fake artifact whose download writes model weights."""
    artifact = MagicMock()
    artifact.name = name.split(":")[0] + ":latest"
    artifact.version = name.split(":")[1]
    artifact.created_at = created_at
    artifact.metadata = metadata

    def download(root: str) -> str:
        Path(root).mkdir(parents=True, exist_ok=True)
        model = SimpleAutoencoder(
            input_dim=5,
            hidden_dim=8,
            latent_dim=2,
        )
        torch.save(model.state_dict(), Path(root) / "weights.pt")
        return root

    artifact.download.side_effect = download
    return artifact


def _make_registry(
    mock_api_class: MagicMock,
    tmp_path: Path,
    artifacts: dict[str, MagicMock],
) -> WandbAutoencoderRegistry:
    """Build a registry whose wandb API serves the given artifacts."""
    mock_api = mock_api_class.return_value
    mock_api.artifact_collections.return_value = [
        SimpleNamespace(name=name.split(":")[0]) for name in artifacts
    ] + [SimpleNamespace(name="unrelated-model")]
    mock_api.artifact.side_effect = lambda path, **_: artifacts[
        path.removeprefix(f"{PROJECT_PATH}/")
    ]

    env_file_path = tmp_path / ".env"
    env_file_path.write_text("", encoding="utf-8")
    return WandbAutoencoderRegistry(
        project_name="test-project",
        entity="test-entity",
        download_directory=str(tmp_path / "downloads"),
        env_file_path=str(env_file_path),
        device="cpu",
    )


@patch(API_PATH)
def test_list_available_artifacts_skips_incomplete_metadata(
    mock_api_class, tmp_path
) -> None:
    """Check that artifacts without inference metadata are ignored."""
    complete_artifact = _make_artifact(
        "simple-autoencoder-aapl:v0",
        "2026-10-01T00:00:00Z",
        _make_metadata("AAPL"),
    )
    incomplete_artifact = _make_artifact(
        "simple-autoencoder-outlier-trading:v3",
        "2026-10-02T00:00:00Z",
        {"ticker": "AAPL", "window_size": 10},
    )
    registry = _make_registry(
        mock_api_class,
        tmp_path,
        {
            "simple-autoencoder-aapl:latest": complete_artifact,
            "simple-autoencoder-outlier-trading:latest": incomplete_artifact,
        },
    )

    assert registry.list_available_artifacts() == [complete_artifact]


@patch(API_PATH)
def test_load_latest_model_picks_newest_artifact(
    mock_api_class, tmp_path
) -> None:
    """Check that the most recently created artifact is loaded."""
    registry = _make_registry(
        mock_api_class,
        tmp_path,
        {
            "simple-autoencoder-aapl:latest": _make_artifact(
                "simple-autoencoder-aapl:v2",
                "2026-10-01T00:00:00Z",
                _make_metadata("AAPL"),
            ),
            "simple-autoencoder-petr4.sa:latest": _make_artifact(
                "simple-autoencoder-petr4.sa:v0",
                "2026-10-05T00:00:00Z",
                _make_metadata("PETR4.SA"),
            ),
        },
    )

    predictor = registry.load_latest_model()

    assert isinstance(predictor, AutoencoderPredictor)
    assert predictor.metadata.ticker == "PETR4.SA"
    assert predictor.metadata.artifact_name == (
        "simple-autoencoder-petr4.sa:v0"
    )
    assert predictor.metadata.anomaly_threshold == 1.5
    assert registry.get_model("PETR4.SA") is predictor
    assert (tmp_path / "downloads" / "simple-autoencoder-petr4.sa-v0").is_dir()


@patch(API_PATH)
def test_load_latest_model_raises_without_artifacts(
    mock_api_class, tmp_path
) -> None:
    """Check that an empty project raises LookupError."""
    registry = _make_registry(mock_api_class, tmp_path, {})

    with pytest.raises(LookupError, match="No servable"):
        registry.load_latest_model()


@patch(API_PATH)
def test_load_model_uses_ticker_artifact(mock_api_class, tmp_path) -> None:
    """Check that load_model reads the latest artifact of the ticker."""
    registry = _make_registry(
        mock_api_class,
        tmp_path,
        {
            "simple-autoencoder-aapl:latest": _make_artifact(
                "simple-autoencoder-aapl:v1",
                "2026-10-01T00:00:00Z",
                _make_metadata("AAPL"),
            ),
        },
    )

    predictor = registry.load_model("AAPL")

    mock_api_class.return_value.artifact.assert_called_once_with(
        f"{PROJECT_PATH}/simple-autoencoder-aapl:latest",
        type="model",
    )
    assert predictor.metadata.window_size == 5
    assert registry.list_loaded_tickers() == ["AAPL"]


@patch(API_PATH)
def test_load_model_rejects_incomplete_metadata(
    mock_api_class, tmp_path
) -> None:
    """Check that a ticker artifact without statistics is rejected."""
    registry = _make_registry(
        mock_api_class,
        tmp_path,
        {
            "simple-autoencoder-aapl:latest": _make_artifact(
                "simple-autoencoder-aapl:v0",
                "2026-10-01T00:00:00Z",
                {"ticker": "AAPL"},
            ),
        },
    )

    with pytest.raises(ValueError, match="missing metadata"):
        registry.load_model("AAPL")


@patch(API_PATH)
def test_load_all_models_loads_every_ticker(mock_api_class, tmp_path) -> None:
    """Check that every servable artifact is loaded into memory."""
    registry = _make_registry(
        mock_api_class,
        tmp_path,
        {
            "simple-autoencoder-aapl:latest": _make_artifact(
                "simple-autoencoder-aapl:v0",
                "2026-10-01T00:00:00Z",
                _make_metadata("AAPL"),
            ),
            "simple-autoencoder-msft:latest": _make_artifact(
                "simple-autoencoder-msft:v0",
                "2026-10-02T00:00:00Z",
                _make_metadata("MSFT"),
            ),
        },
    )

    assert registry.load_all_models() == ["AAPL", "MSFT"]
    assert registry.list_loaded_tickers() == ["AAPL", "MSFT"]


def test_get_model_raises_for_unloaded_ticker(tmp_path) -> None:
    """Check that requesting an unloaded ticker raises KeyError."""
    registry = WandbAutoencoderRegistry(
        download_directory=str(tmp_path),
        device="cpu",
    )

    with pytest.raises(KeyError, match="load_model"):
        registry.get_model("AAPL")
