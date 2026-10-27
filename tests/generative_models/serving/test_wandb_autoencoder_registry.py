"""Unit tests for WandbAutoencoderRegistry."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from generative_models.models.autoencoder.autoencoder import (
    SimpleAutoencoder,
)
from generative_models.serving.autoencoder_predictor import (
    AutoencoderPredictor,
)
from generative_models.serving.wandb_autoencoder_registry import (
    WandbAutoencoderRegistry,
    build_versioned_artifact_name,
)

API_PATH = "generative_models.serving.wandb_autoencoder_registry.wandb.Api"


def _make_metadata(
    ticker: str = "PETR4.SA",
    window_size: int = 5,
    hidden_layer_size: int = 8,
    latent_layer_size: int = 2,
) -> dict[str, object]:
    """Build a complete set of artifact metadata."""
    return {
        "ticker": ticker,
        "window_size": window_size,
        "hidden_layer_size": hidden_layer_size,
        "latent_layer_size": latent_layer_size,
        "return_train_mean": 0.001,
        "return_train_std": 0.02,
        "anomaly_threshold": 0.5,
    }


def _make_artifact(
    collection_name: str,
    version: str,
    created_at: str,
    metadata: dict[str, object],
    weights_file_name: str | None = "model.pt",
) -> MagicMock:
    """Build a fake wandb artifact that writes real weights on download."""
    artifact = MagicMock()
    artifact.name = f"{collection_name}:{version}"
    artifact.version = version
    artifact.created_at = created_at
    artifact.metadata = metadata

    def download(root: str) -> str:
        """Write a state_dict into the root directory, like wandb does."""
        if weights_file_name is not None:
            torch.manual_seed(0)
            state_dict = SimpleAutoencoder(
                input_dim=metadata["window_size"],
                hidden_dim=metadata["hidden_layer_size"],
                latent_dim=metadata["latent_layer_size"],
            ).state_dict()
            Path(root).mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, Path(root) / weights_file_name)

        return root

    artifact.download.side_effect = download

    return artifact


def _make_registry(tmp_path, entity: str | None = "test-entity"):
    """Build a registry that stores downloads inside tmp_path."""
    return WandbAutoencoderRegistry(
        project_name="test-project",
        entity=entity,
        download_directory=str(tmp_path / "downloads"),
        env_file_path=str(tmp_path / ".env"),
        device="cpu",
    )


def _attach_api(api_class: MagicMock, collection_names, artifacts_by_name):
    """Configure the mocked wandb.Api used by a registry."""
    api = api_class.return_value
    api.default_entity = "default-entity"
    collections = []
    for collection_name in collection_names:
        collection = MagicMock()
        collection.name = collection_name
        collections.append(collection)
    api.artifact_collections.return_value = collections

    def get_artifact(name: str, **_options: str) -> MagicMock:
        """Return the artifact registered under the requested name."""
        return artifacts_by_name[name]

    api.artifact.side_effect = get_artifact

    return api


@patch(API_PATH)
def test_constructor_does_not_create_the_api_client(
    api_class,
    tmp_path,
) -> None:
    """Check that the client is created lazily, on first use of `api`."""
    registry = _make_registry(tmp_path)

    api_class.assert_not_called()
    first_client = registry.api
    second_client = registry.api

    api_class.assert_called_once()
    assert first_client is second_client


@patch(API_PATH)
def test_reset_api_makes_the_next_call_create_a_new_client(
    api_class,
    tmp_path,
) -> None:
    """Check that reset_api discards the client so a new one is created."""
    first_client = MagicMock(name="first_client")
    second_client = MagicMock(name="second_client")
    api_class.side_effect = [first_client, second_client]
    registry = _make_registry(tmp_path)
    assert registry.api is first_client

    registry.reset_api()

    assert registry.api is second_client
    assert api_class.call_count == 2


@patch(API_PATH)
def test_project_path_uses_entity_or_default_entity(
    api_class,
    tmp_path,
) -> None:
    """Check the project path with and without an explicit entity."""
    _attach_api(api_class, [], {})

    explicit_registry = _make_registry(tmp_path, entity="test-entity")
    default_registry = _make_registry(tmp_path, entity=None)

    assert explicit_registry.project_path == "test-entity/test-project"
    assert default_registry.project_path == "default-entity/test-project"


@patch(API_PATH)
def test_list_available_artifacts_skips_invalid_entries(
    api_class, tmp_path
) -> None:
    """Check that foreign collections and incomplete metadata are skipped."""
    valid_artifact = _make_artifact(
        "simple-autoencoder-petr4.sa",
        "v1",
        "2026-09-01",
        _make_metadata(),
    )
    incomplete_metadata = _make_metadata()
    del incomplete_metadata["anomaly_threshold"]
    incomplete_artifact = _make_artifact(
        "simple-autoencoder-aapl",
        "v1",
        "2026-09-02",
        incomplete_metadata,
    )
    artifacts_by_name = {
        "test-entity/test-project/simple-autoencoder-petr4.sa:latest": (
            valid_artifact
        ),
        "test-entity/test-project/simple-autoencoder-aapl:latest": (
            incomplete_artifact
        ),
    }
    _attach_api(
        api_class,
        [
            "simple-autoencoder-petr4.sa",
            "simple-autoencoder-aapl",
            "other-collection",
        ],
        artifacts_by_name,
    )
    registry = _make_registry(tmp_path)

    artifacts = registry.list_available_artifacts()

    assert artifacts == [valid_artifact]


@patch(API_PATH)
def test_load_latest_model_picks_the_most_recent_artifact(
    api_class, tmp_path
) -> None:
    """Check that the artifact with the greatest created_at is loaded."""
    older_artifact = _make_artifact(
        "simple-autoencoder-petr4.sa",
        "v1",
        "2026-09-01T10:00:00",
        _make_metadata(ticker="PETR4.SA"),
    )
    newer_artifact = _make_artifact(
        "simple-autoencoder-aapl",
        "v2",
        "2026-09-10T10:00:00",
        _make_metadata(ticker="AAPL"),
    )
    _attach_api(
        api_class,
        ["simple-autoencoder-petr4.sa", "simple-autoencoder-aapl"],
        {
            "test-entity/test-project/simple-autoencoder-petr4.sa:latest": (
                older_artifact
            ),
            "test-entity/test-project/simple-autoencoder-aapl:latest": (
                newer_artifact
            ),
        },
    )
    registry = _make_registry(tmp_path)

    predictor = registry.load_latest_model()

    assert predictor.metadata.ticker == "AAPL"


@patch(API_PATH)
def test_load_latest_model_raises_when_nothing_is_servable(
    api_class, tmp_path
) -> None:
    """Check that an empty project raises LookupError."""
    _attach_api(api_class, [], {})
    registry = _make_registry(tmp_path)

    with pytest.raises(LookupError, match="No servable autoencoder artifact"):
        registry.load_latest_model()


@patch(API_PATH)
def test_load_model_fetches_the_latest_alias_and_builds_predictor(
    api_class, tmp_path
) -> None:
    """Check the artifact name requested and the predictor values."""
    artifact = _make_artifact(
        "simple-autoencoder-petr4.sa",
        "v2",
        "2026-09-10",
        _make_metadata(ticker="PETR4.SA"),
    )
    api = _attach_api(
        api_class,
        [],
        {
            "test-entity/test-project/simple-autoencoder-petr4.sa:latest": (
                artifact
            ),
        },
    )
    registry = _make_registry(tmp_path)

    predictor = registry.load_model("PETR4.SA")

    api.artifact.assert_called_with(
        "test-entity/test-project/simple-autoencoder-petr4.sa:latest",
        type="model",
    )
    assert isinstance(predictor, AutoencoderPredictor)
    assert predictor.metadata.window_size == 5
    assert predictor.metadata.return_train_mean == 0.001
    assert predictor.metadata.return_train_std == 0.02
    assert predictor.metadata.anomaly_threshold == 0.5


@patch(API_PATH)
def test_load_model_sets_the_versioned_artifact_name(
    api_class,
    tmp_path,
) -> None:
    """Check that the predictor records the version, not the alias."""
    artifact = _make_artifact(
        "simple-autoencoder-petr4.sa",
        "v2",
        "2026-09-10",
        _make_metadata(ticker="PETR4.SA"),
    )
    _attach_api(
        api_class,
        [],
        {
            "test-entity/test-project/simple-autoencoder-petr4.sa:latest": (
                artifact
            ),
        },
    )
    registry = _make_registry(tmp_path)

    predictor = registry.load_model("PETR4.SA")

    assert predictor.metadata.artifact_name == "simple-autoencoder-petr4.sa:v2"


@patch(API_PATH)
def test_load_model_rebuilds_the_saved_architecture(
    api_class,
    tmp_path,
) -> None:
    """Check that the rebuilt model has the metadata sizes and weights."""
    metadata = _make_metadata(hidden_layer_size=8, latent_layer_size=2)
    artifact = _make_artifact(
        "simple-autoencoder-petr4.sa",
        "v1",
        "2026-09-10",
        metadata,
    )
    _attach_api(
        api_class,
        [],
        {
            "test-entity/test-project/simple-autoencoder-petr4.sa:latest": (
                artifact
            ),
        },
    )
    registry = _make_registry(tmp_path)

    predictor = registry.load_model("PETR4.SA")

    assert predictor.model.encoder[0].out_features == 8
    assert predictor.model.encoder[2].out_features == 2
    torch.manual_seed(0)
    expected_state = SimpleAutoencoder(
        input_dim=5,
        hidden_dim=8,
        latent_dim=2,
    ).state_dict()
    for key, expected_tensor in expected_state.items():
        assert torch.equal(predictor.model.state_dict()[key], expected_tensor)


@patch(API_PATH)
def test_load_model_rejects_incomplete_metadata(api_class, tmp_path) -> None:
    """Check that missing metadata keys raise ValueError."""
    incomplete_metadata = _make_metadata()
    del incomplete_metadata["window_size"]
    artifact = _make_artifact(
        "simple-autoencoder-petr4.sa",
        "v1",
        "2026-09-10",
        incomplete_metadata,
    )
    _attach_api(
        api_class,
        [],
        {
            "test-entity/test-project/simple-autoencoder-petr4.sa:latest": (
                artifact
            ),
        },
    )
    registry = _make_registry(tmp_path)

    with pytest.raises(
        ValueError, match="is missing metadata required for inference"
    ):
        registry.load_model("PETR4.SA")


@patch(API_PATH)
def test_load_model_raises_when_artifact_has_no_weights(
    api_class, tmp_path
) -> None:
    """Check that an artifact without a .pt file raises FileNotFoundError."""
    artifact = _make_artifact(
        "simple-autoencoder-petr4.sa",
        "v1",
        "2026-09-10",
        _make_metadata(),
        weights_file_name=None,
    )
    _attach_api(
        api_class,
        [],
        {
            "test-entity/test-project/simple-autoencoder-petr4.sa:latest": (
                artifact
            ),
        },
    )
    registry = _make_registry(tmp_path)

    with pytest.raises(FileNotFoundError, match="has no .pt weights file"):
        registry.load_model("PETR4.SA")


@patch(API_PATH)
def test_load_all_models_returns_the_loaded_tickers(
    api_class,
    tmp_path,
) -> None:
    """Check that every valid artifact is loaded and listed."""
    petr_artifact = _make_artifact(
        "simple-autoencoder-petr4.sa",
        "v1",
        "2026-09-01",
        _make_metadata(ticker="PETR4.SA"),
    )
    aapl_artifact = _make_artifact(
        "simple-autoencoder-aapl",
        "v1",
        "2026-09-02",
        _make_metadata(ticker="AAPL"),
    )
    _attach_api(
        api_class,
        ["simple-autoencoder-petr4.sa", "simple-autoencoder-aapl"],
        {
            "test-entity/test-project/simple-autoencoder-petr4.sa:latest": (
                petr_artifact
            ),
            "test-entity/test-project/simple-autoencoder-aapl:latest": (
                aapl_artifact
            ),
        },
    )
    registry = _make_registry(tmp_path)

    loaded_tickers = registry.load_all_models()

    assert set(loaded_tickers) == {"PETR4.SA", "AAPL"}
    assert set(registry.list_loaded_tickers()) == {"PETR4.SA", "AAPL"}


@patch(API_PATH)
def test_get_model_raises_key_error_without_calling_the_api(
    api_class, tmp_path
) -> None:
    """Check that a ticker that is not in memory raises KeyError."""
    registry = _make_registry(tmp_path)

    with pytest.raises(KeyError, match="Call load_model first"):
        registry.get_model("PETR4.SA")

    api_class.assert_not_called()


def test_build_versioned_artifact_name_drops_the_alias() -> None:
    """Check that the versioned name uses the collection and the version."""
    artifact = MagicMock()
    artifact.name = "simple-autoencoder-aapl:latest"
    artifact.version = "v3"

    assert build_versioned_artifact_name(artifact) == (
        "simple-autoencoder-aapl:v3"
    )
