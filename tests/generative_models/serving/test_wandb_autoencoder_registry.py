"""Unit tests for WandbAutoencoderRegistry."""

from pathlib import Path
from typing import Any
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
)

REGISTRY_MODULE = (
    "generative_models.serving.wandb_autoencoder_registry"
)
PETR_COLLECTION = "simple-autoencoder-petr4.sa"
AAPL_COLLECTION = "simple-autoencoder-aapl"


def build_artifact_metadata(
    ticker: str = "PETR4.SA",
    anomaly_threshold: float = 0.5,
) -> dict[str, Any]:
    """Build the artifact metadata written by the training pipeline.

    Parameters
    ----------
    ticker : str
        Ticker symbol stored in the metadata.
    anomaly_threshold : float
        Reconstruction error above which a window is an outlier.

    Returns
    -------
    dict[str, Any]
        Metadata holding every key inference requires.
    """
    return {
        "ticker": ticker,
        "window_size": 5,
        "hidden_layer_size": 8,
        "latent_layer_size": 2,
        "return_train_mean": 0.001,
        "return_train_std": 0.02,
        "anomaly_threshold": anomaly_threshold,
    }


def build_reference_state_dict() -> dict[str, torch.Tensor]:
    """Build the weights the fake artifact download writes to disk.

    Returns
    -------
    dict[str, torch.Tensor]
        State dict of an autoencoder with hidden size 8 and latent
        size 2.
    """
    torch.manual_seed(3)
    model = SimpleAutoencoder(input_dim=5, hidden_dim=8, latent_dim=2)

    return model.state_dict()


def build_fake_artifact(
    collection_name: str,
    version: str = "v1",
    created_at: str = "2026-01-01T00:00:00",
    metadata: dict[str, Any] | None = None,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> MagicMock:
    """Build a stand-in for a wandb model artifact.

    Parameters
    ----------
    collection_name : str
        Name of the artifact collection, without the version suffix.
    version : str
        Version label reported by the artifact.
    created_at : str
        Creation timestamp used to pick the most recent artifact.
    metadata : dict[str, Any] | None
        Artifact metadata. Defaults to a complete metadata dictionary.
    state_dict : dict[str, torch.Tensor] | None
        Weights written by `download`. When `None`, the download
        writes no `.pt` file at all.

    Returns
    -------
    MagicMock
        Object exposing the artifact attributes the registry reads.
    """
    artifact = MagicMock()
    artifact.name = f"{collection_name}:latest"
    artifact.version = version
    artifact.created_at = created_at
    artifact.metadata = (
        build_artifact_metadata() if metadata is None else metadata
    )

    def download(root: str) -> str:
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        if state_dict is not None:
            torch.save(state_dict, root_path / "weights.pt")
        return str(root_path)

    artifact.download.side_effect = download

    return artifact


def build_collection(collection_name: str) -> MagicMock:
    """Build a stand-in for a wandb artifact collection.

    Parameters
    ----------
    collection_name : str
        Name reported by the collection.

    Returns
    -------
    MagicMock
        Object exposing the `name` attribute the registry reads.
    """
    collection = MagicMock()
    collection.name = collection_name

    return collection


def build_registry(
    download_directory: Path,
    entity: str | None = "test-entity",
) -> WandbAutoencoderRegistry:
    """Build a registry pointing at the test project.

    Parameters
    ----------
    download_directory : Path
        Directory where artifacts are downloaded.
    entity : str | None
        Entity owning the project, or `None` to use the default one.

    Returns
    -------
    WandbAutoencoderRegistry
        Registry configured for CPU inference.
    """
    return WandbAutoencoderRegistry(
        project_name="test-project",
        entity=entity,
        download_directory=str(download_directory),
        device="cpu",
    )


def test_api_client_is_created_lazily_and_reused(tmp_path: Path) -> None:
    """CA-01: the constructor never builds the client, and api caches it."""
    with patch(f"{REGISTRY_MODULE}.wandb.Api") as api_class:
        registry = build_registry(tmp_path)

        api_class.assert_not_called()

        first_client = registry.api
        second_client = registry.api

        api_class.assert_called_once()
        assert first_client is second_client


def test_project_path_uses_the_default_entity(tmp_path: Path) -> None:
    """CA-02: project_path falls back to the default entity of the api."""
    with patch(f"{REGISTRY_MODULE}.wandb.Api") as api_class:
        api_class.return_value.default_entity = "default-entity"

        assert build_registry(tmp_path, entity=None).project_path == (
            "default-entity/test-project"
        )
        assert build_registry(tmp_path).project_path == (
            "test-entity/test-project"
        )


def test_list_available_artifacts_filters_collections(
    tmp_path: Path,
) -> None:
    """CA-03: collections without the prefix and bad metadata are skipped."""
    complete_artifact = build_fake_artifact(PETR_COLLECTION)
    incomplete_artifact = build_fake_artifact(
        AAPL_COLLECTION,
        metadata={"ticker": "AAPL"},
    )

    def artifact_lookup(artifact_name: str, **_: Any) -> MagicMock:
        if PETR_COLLECTION in artifact_name:
            return complete_artifact
        return incomplete_artifact

    with patch(f"{REGISTRY_MODULE}.wandb.Api") as api_class:
        api_class.return_value.artifact_collections.return_value = [
            build_collection(PETR_COLLECTION),
            build_collection("feed-forward-regression-model"),
            build_collection(AAPL_COLLECTION),
        ]
        api_class.return_value.artifact.side_effect = artifact_lookup

        available_artifacts = build_registry(
            tmp_path
        ).list_available_artifacts()

    assert available_artifacts == [complete_artifact]


def test_load_latest_model_picks_the_most_recent_artifact(
    tmp_path: Path,
) -> None:
    """CA-04: the artifact with the newest created_at is loaded."""
    state_dict = build_reference_state_dict()
    older_artifact = build_fake_artifact(
        PETR_COLLECTION,
        created_at="2026-01-01T00:00:00",
        state_dict=state_dict,
    )
    newer_artifact = build_fake_artifact(
        AAPL_COLLECTION,
        created_at="2026-05-01T00:00:00",
        metadata=build_artifact_metadata(ticker="AAPL"),
        state_dict=state_dict,
    )

    def artifact_lookup(artifact_name: str, **_: Any) -> MagicMock:
        if PETR_COLLECTION in artifact_name:
            return older_artifact
        return newer_artifact

    with patch(f"{REGISTRY_MODULE}.wandb.Api") as api_class:
        api_class.return_value.artifact_collections.return_value = [
            build_collection(PETR_COLLECTION),
            build_collection(AAPL_COLLECTION),
        ]
        api_class.return_value.artifact.side_effect = artifact_lookup

        predictor = build_registry(tmp_path).load_latest_model()

    assert predictor.metadata.ticker == "AAPL"


def test_load_latest_model_without_artifacts_raises_lookup_error(
    tmp_path: Path,
) -> None:
    """CA-05: an empty project raises LookupError."""
    with patch(f"{REGISTRY_MODULE}.wandb.Api") as api_class:
        api_class.return_value.artifact_collections.return_value = []

        with pytest.raises(LookupError, match="No servable autoencoder"):
            build_registry(tmp_path).load_latest_model()


def test_load_model_returns_predictor_built_from_metadata(
    tmp_path: Path,
) -> None:
    """CA-06: the ticker artifact is fetched and wrapped in a predictor."""
    artifact = build_fake_artifact(
        PETR_COLLECTION,
        state_dict=build_reference_state_dict(),
    )

    with patch(f"{REGISTRY_MODULE}.wandb.Api") as api_class:
        api_class.return_value.artifact.return_value = artifact

        predictor = build_registry(tmp_path).load_model("PETR4.SA")

    requested_name = api_class.return_value.artifact.call_args.args[0]

    assert requested_name == (
        "test-entity/test-project/simple-autoencoder-petr4.sa:latest"
    )
    assert isinstance(predictor, AutoencoderPredictor)
    assert predictor.metadata.window_size == 5
    assert predictor.metadata.return_train_mean == 0.001
    assert predictor.metadata.return_train_std == 0.02
    assert predictor.metadata.anomaly_threshold == 0.5


def test_artifact_name_is_versioned(tmp_path: Path) -> None:
    """CA-07: the predictor records the versioned name, not latest."""
    artifact = build_fake_artifact(
        PETR_COLLECTION,
        version="v2",
        state_dict=build_reference_state_dict(),
    )

    with patch(f"{REGISTRY_MODULE}.wandb.Api") as api_class:
        api_class.return_value.artifact.return_value = artifact

        predictor = build_registry(tmp_path).load_model("PETR4.SA")

    assert predictor.metadata.artifact_name == (
        "simple-autoencoder-petr4.sa:v2"
    )


def test_loaded_model_matches_the_saved_architecture_and_weights(
    tmp_path: Path,
) -> None:
    """CA-08: the model is rebuilt from the metadata with the same weights."""
    state_dict = build_reference_state_dict()
    artifact = build_fake_artifact(
        PETR_COLLECTION,
        state_dict=state_dict,
    )

    with patch(f"{REGISTRY_MODULE}.wandb.Api") as api_class:
        api_class.return_value.artifact.return_value = artifact

        predictor = build_registry(tmp_path).load_model("PETR4.SA")

    assert predictor.model.encoder[0].out_features == 8
    assert predictor.model.encoder[2].out_features == 2
    assert torch.equal(
        predictor.model.encoder[0].weight,
        state_dict["encoder.0.weight"],
    )


def test_load_model_with_incomplete_metadata_raises_value_error(
    tmp_path: Path,
) -> None:
    """CA-09: missing metadata keys raise ValueError."""
    artifact = build_fake_artifact(
        PETR_COLLECTION,
        metadata={"ticker": "PETR4.SA"},
    )

    with patch(f"{REGISTRY_MODULE}.wandb.Api") as api_class:
        api_class.return_value.artifact.return_value = artifact

        with pytest.raises(ValueError, match="is missing metadata"):
            build_registry(tmp_path).load_model("PETR4.SA")


def test_artifact_without_weights_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """CA-10: an artifact without a .pt file raises FileNotFoundError."""
    artifact = build_fake_artifact(PETR_COLLECTION)

    with patch(f"{REGISTRY_MODULE}.wandb.Api") as api_class:
        api_class.return_value.artifact.return_value = artifact

        with pytest.raises(FileNotFoundError, match="has no .pt"):
            build_registry(tmp_path).load_model("PETR4.SA")


def test_load_all_models_loads_every_valid_artifact(tmp_path: Path) -> None:
    """CA-11: every valid artifact is loaded and listed afterwards."""
    state_dict = build_reference_state_dict()
    petr_artifact = build_fake_artifact(
        PETR_COLLECTION,
        state_dict=state_dict,
    )
    aapl_artifact = build_fake_artifact(
        AAPL_COLLECTION,
        metadata=build_artifact_metadata(ticker="AAPL"),
        state_dict=state_dict,
    )

    def artifact_lookup(artifact_name: str, **_: Any) -> MagicMock:
        if PETR_COLLECTION in artifact_name:
            return petr_artifact
        return aapl_artifact

    with patch(f"{REGISTRY_MODULE}.wandb.Api") as api_class:
        api_class.return_value.artifact_collections.return_value = [
            build_collection(PETR_COLLECTION),
            build_collection(AAPL_COLLECTION),
        ]
        api_class.return_value.artifact.side_effect = artifact_lookup

        registry = build_registry(tmp_path)
        loaded_tickers = registry.load_all_models()

    assert loaded_tickers == ["PETR4.SA", "AAPL"]
    assert registry.list_loaded_tickers() == ["PETR4.SA", "AAPL"]


def test_get_model_of_unloaded_ticker_raises_key_error(
    tmp_path: Path,
) -> None:
    """CA-12: get_model never touches the api for an unloaded ticker."""
    with patch(f"{REGISTRY_MODULE}.wandb.Api") as api_class:
        registry = build_registry(tmp_path)

        with pytest.raises(KeyError, match="No model loaded for ticker"):
            registry.get_model("PETR4.SA")

        api_class.assert_not_called()
