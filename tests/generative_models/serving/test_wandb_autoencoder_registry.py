"""Unit tests for WandbAutoencoderRegistry."""

from pathlib import Path
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


def _make_metadata(ticker: str) -> dict[str, object]:
    """Build complete artifact metadata for a ticker."""
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
    collection_name: str,
    version: str,
    created_at: str,
    metadata: dict[str, object],
    model: SimpleAutoencoder | None = None,
) -> MagicMock:
    """Build a fake artifact whose download writes a state_dict."""
    weights_model = model or SimpleAutoencoder(
        input_dim=5, hidden_dim=8, latent_dim=2
    )

    def download(root: str) -> str:
        """Write the weights file into the download directory."""
        Path(root).mkdir(parents=True, exist_ok=True)
        torch.save(weights_model.state_dict(), Path(root) / "model.pt")
        return root

    artifact = MagicMock()
    artifact.name = f"{collection_name}:{version}"
    artifact.version = version
    artifact.created_at = created_at
    artifact.metadata = metadata
    artifact.download.side_effect = download
    return artifact


def _make_collection(collection_name: str) -> MagicMock:
    """Build a fake artifact collection."""
    collection = MagicMock()
    collection.name = collection_name
    return collection


def _make_registry(tmp_path: Path) -> WandbAutoencoderRegistry:
    """Build a registry writing into a temporary directory."""
    env_file_path = tmp_path / ".env"
    env_file_path.write_text("", encoding="utf-8")
    return WandbAutoencoderRegistry(
        project_name="test-project",
        entity="test-entity",
        download_directory=str(tmp_path / "downloads"),
        env_file_path=str(env_file_path),
        device="cpu",
    )


def _configure_api(
    mock_api_class: MagicMock,
    artifacts_by_collection: dict[str, MagicMock],
) -> MagicMock:
    """Make the fake API list and return the given artifacts."""
    mock_api = mock_api_class.return_value
    mock_api.artifact_collections.return_value = [
        _make_collection(collection_name)
        for collection_name in artifacts_by_collection
    ]

    def get_artifact(artifact_path: str, **_kwargs: object) -> MagicMock:
        """Return the artifact of the requested collection."""
        collection_name = artifact_path.split("/")[-1].split(":")[0]
        return artifacts_by_collection[collection_name]

    mock_api.artifact.side_effect = get_artifact
    return mock_api


@patch(API_PATH)
def test_api_is_created_lazily_once(mock_api_class, tmp_path) -> None:
    """CA-01: the client is created on first access and then reused."""
    registry = _make_registry(tmp_path)
    mock_api_class.assert_not_called()

    first_api = registry.api
    second_api = registry.api

    assert first_api is second_api
    mock_api_class.assert_called_once()


@patch(API_PATH)
def test_project_path(mock_api_class, tmp_path) -> None:
    """CA-02: project_path uses the entity or the API default."""
    mock_api_class.return_value.default_entity = "default-entity"
    registry = _make_registry(tmp_path)

    assert registry.project_path == "test-entity/test-project"
    registry.entity = None
    assert registry.project_path == "default-entity/test-project"


@patch(API_PATH)
def test_list_available_artifacts_filters(mock_api_class, tmp_path) -> None:
    """CA-03: other collections and incomplete metadata are skipped."""
    valid_artifact = _make_artifact(
        "simple-autoencoder-aapl", "v1", "2026-10-01", _make_metadata("AAPL")
    )
    _configure_api(
        mock_api_class,
        {
            "simple-autoencoder-aapl": valid_artifact,
            "other-model": _make_artifact(
                "other-model", "v0", "2026-10-02", _make_metadata("X")
            ),
            "simple-autoencoder-msft": _make_artifact(
                "simple-autoencoder-msft", "v0", "2026-10-03", {"ticker": "M"}
            ),
        },
    )

    available_artifacts = _make_registry(tmp_path).list_available_artifacts()

    assert available_artifacts == [valid_artifact]


@patch(API_PATH)
def test_load_latest_model_picks_newest(mock_api_class, tmp_path) -> None:
    """CA-04: the artifact with the latest created_at is loaded."""
    _configure_api(
        mock_api_class,
        {
            "simple-autoencoder-aapl": _make_artifact(
                "simple-autoencoder-aapl",
                "v1",
                "2026-10-01",
                _make_metadata("AAPL"),
            ),
            "simple-autoencoder-msft": _make_artifact(
                "simple-autoencoder-msft",
                "v0",
                "2026-10-05",
                _make_metadata("MSFT"),
            ),
        },
    )

    predictor = _make_registry(tmp_path).load_latest_model()

    assert predictor.metadata.ticker == "MSFT"


@patch(API_PATH)
def test_load_latest_model_without_artifacts(
    mock_api_class,
    tmp_path,
) -> None:
    """CA-05: no servable artifact raises LookupError."""
    _configure_api(mock_api_class, {})

    with pytest.raises(LookupError, match="No servable autoencoder"):
        _make_registry(tmp_path).load_latest_model()


@patch(API_PATH)
def test_load_model_returns_predictor(mock_api_class, tmp_path) -> None:
    """CA-06: load_model fetches the ticker artifact and its metadata."""
    mock_api = _configure_api(
        mock_api_class,
        {
            "simple-autoencoder-petr4.sa": _make_artifact(
                "simple-autoencoder-petr4.sa",
                "v2",
                "2026-10-01",
                _make_metadata("PETR4.SA"),
            )
        },
    )

    predictor = _make_registry(tmp_path).load_model("PETR4.SA")

    mock_api.artifact.assert_called_once_with(
        "test-entity/test-project/simple-autoencoder-petr4.sa:latest",
        type="model",
    )
    assert isinstance(predictor, AutoencoderPredictor)
    assert predictor.metadata.window_size == 5
    assert predictor.metadata.return_train_mean == 0.001
    assert predictor.metadata.return_train_std == 0.02
    assert predictor.metadata.anomaly_threshold == 1.5


@patch(API_PATH)
def test_artifact_name_is_versioned(mock_api_class, tmp_path) -> None:
    """CA-07: artifact_name carries the resolved version."""
    _configure_api(
        mock_api_class,
        {
            "simple-autoencoder-petr4.sa": _make_artifact(
                "simple-autoencoder-petr4.sa",
                "v2",
                "2026-10-01",
                _make_metadata("PETR4.SA"),
            )
        },
    )

    predictor = _make_registry(tmp_path).load_model("PETR4.SA")

    assert predictor.metadata.artifact_name == (
        "simple-autoencoder-petr4.sa:v2"
    )


@patch(API_PATH)
def test_model_is_rebuilt_with_saved_weights(
    mock_api_class,
    tmp_path,
) -> None:
    """CA-08: architecture and weights match the stored artifact."""
    torch.manual_seed(3)
    stored_model = SimpleAutoencoder(input_dim=5, hidden_dim=8, latent_dim=2)
    _configure_api(
        mock_api_class,
        {
            "simple-autoencoder-aapl": _make_artifact(
                "simple-autoencoder-aapl",
                "v0",
                "2026-10-01",
                _make_metadata("AAPL"),
                model=stored_model,
            )
        },
    )

    predictor = _make_registry(tmp_path).load_model("AAPL")

    assert predictor.model.encoder[0].out_features == 8
    assert predictor.model.encoder[2].out_features == 2
    for parameter_name, parameter_value in stored_model.state_dict().items():
        assert torch.equal(
            predictor.model.state_dict()[parameter_name], parameter_value
        )


@patch(API_PATH)
def test_load_model_with_incomplete_metadata(
    mock_api_class,
    tmp_path,
) -> None:
    """CA-09: missing metadata keys raise ValueError."""
    _configure_api(
        mock_api_class,
        {
            "simple-autoencoder-aapl": _make_artifact(
                "simple-autoencoder-aapl", "v0", "2026-10-01", {"ticker": "A"}
            )
        },
    )

    with pytest.raises(ValueError, match="missing metadata"):
        _make_registry(tmp_path).load_model("AAPL")


@patch(API_PATH)
def test_artifact_without_weights(mock_api_class, tmp_path) -> None:
    """CA-10: an artifact without a .pt file raises FileNotFoundError."""
    artifact = _make_artifact(
        "simple-autoencoder-aapl", "v0", "2026-10-01", _make_metadata("AAPL")
    )
    artifact.download.side_effect = lambda root: root
    _configure_api(mock_api_class, {"simple-autoencoder-aapl": artifact})

    with pytest.raises(FileNotFoundError, match=r"has no \.pt weights"):
        _make_registry(tmp_path).load_model("AAPL")


@patch(API_PATH)
def test_load_all_models(mock_api_class, tmp_path) -> None:
    """CA-11: every servable artifact is loaded and listed."""
    _configure_api(
        mock_api_class,
        {
            "simple-autoencoder-aapl": _make_artifact(
                "simple-autoencoder-aapl",
                "v0",
                "2026-10-01",
                _make_metadata("AAPL"),
            ),
            "simple-autoencoder-msft": _make_artifact(
                "simple-autoencoder-msft",
                "v0",
                "2026-10-02",
                _make_metadata("MSFT"),
            ),
        },
    )
    registry = _make_registry(tmp_path)

    loaded_tickers = registry.load_all_models()

    assert loaded_tickers == ["AAPL", "MSFT"]
    assert registry.list_loaded_tickers() == ["AAPL", "MSFT"]


@patch(API_PATH)
def test_get_model_of_unloaded_ticker(mock_api_class, tmp_path) -> None:
    """CA-12: get_model raises KeyError without calling the API."""
    registry = _make_registry(tmp_path)

    with pytest.raises(KeyError, match="No model loaded for ticker"):
        registry.get_model("AAPL")

    mock_api_class.assert_not_called()
