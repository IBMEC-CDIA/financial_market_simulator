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

WANDB_API_PATH = (
    "generative_models.serving.wandb_autoencoder_registry.wandb.Api"
)


def build_metadata(ticker: str) -> dict[str, object]:
    """Create complete artifact metadata for a ticker.

    Parameters
    ----------
    ticker : str
        Ticker stored in the metadata.

    Returns
    -------
    dict[str, object]
        Metadata with every key required for inference.
    """
    return {
        "ticker": ticker,
        "window_size": 5,
        "hidden_layer_size": 8,
        "latent_layer_size": 2,
        "return_train_mean": 0.001,
        "return_train_std": 0.02,
        "anomaly_threshold": 0.75,
        "final_train_loss": 0.1,
    }


def build_fake_artifact(
    collection_name: str,
    metadata: dict[str, object],
    version: str = "v2",
    created_at: str = "2026-09-01T00:00:00",
    write_weights: bool = True,
) -> MagicMock:
    """Create a fake wandb artifact whose download writes weights.

    Parameters
    ----------
    collection_name : str
        Name of the artifact collection.
    metadata : dict[str, object]
        Metadata of the artifact.
    version : str
        Version alias of the artifact.
    created_at : str
        Creation timestamp of the artifact.
    write_weights : bool
        Whether `download` writes a `.pt` file.

    Returns
    -------
    MagicMock
        Fake artifact with a `state_dict` attribute holding the
        weights written by `download`.
    """
    torch.manual_seed(3)
    source_model = SimpleAutoencoder(input_dim=5, hidden_dim=8, latent_dim=2)
    fake_artifact = MagicMock()
    fake_artifact.name = f"{collection_name}:latest"
    fake_artifact.version = version
    fake_artifact.created_at = created_at
    fake_artifact.metadata = metadata
    fake_artifact.state_dict = source_model.state_dict()

    def download(root: str) -> str:
        Path(root).mkdir(parents=True, exist_ok=True)
        if write_weights:
            torch.save(
                source_model.state_dict(),
                Path(root) / f"{collection_name}.pt",
            )
        return root

    fake_artifact.download.side_effect = download
    return fake_artifact


def build_fake_collection(collection_name: str) -> MagicMock:
    """Create a fake wandb artifact collection.

    Parameters
    ----------
    collection_name : str
        Name of the collection.

    Returns
    -------
    MagicMock
        Fake collection with the given `name`.
    """
    fake_collection = MagicMock()
    fake_collection.name = collection_name
    return fake_collection


def build_registry(
    tmp_path: Path,
    project_name: str = "test-project",
) -> WandbAutoencoderRegistry:
    """Create a registry that downloads into a temporary directory.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory of the test.
    project_name : str
        wandb project name.

    Returns
    -------
    WandbAutoencoderRegistry
        Registry on CPU with `entity="test-entity"`.
    """
    return WandbAutoencoderRegistry(
        project_name=project_name,
        entity="test-entity",
        download_directory=str(tmp_path / "downloads"),
        env_file_path=str(tmp_path / ".env"),
        device="cpu",
    )


def configure_project(
    mock_api_class: MagicMock,
    artifacts_by_collection: dict[str, MagicMock | None],
) -> MagicMock:
    """Make the mocked wandb client serve a set of collections.

    Parameters
    ----------
    mock_api_class : MagicMock
        Patched `wandb.Api` class.
    artifacts_by_collection : dict[str, MagicMock | None]
        Latest artifact of each collection name.

    Returns
    -------
    MagicMock
        Mocked wandb client.
    """
    mock_api = mock_api_class.return_value
    mock_api.artifact_collections.return_value = [
        build_fake_collection(collection_name)
        for collection_name in artifacts_by_collection
    ]

    def get_artifact(artifact_path: str, **_: object) -> MagicMock | None:
        collection_name = artifact_path.split("/")[-1].split(":")[0]
        return artifacts_by_collection[collection_name]

    mock_api.artifact.side_effect = get_artifact
    return mock_api


def test_api_is_created_lazily_and_reused(tmp_path: Path) -> None:
    """CA-01: check that the wandb client is created once, on demand."""
    with patch(WANDB_API_PATH) as mock_api_class:
        registry = build_registry(tmp_path)
        mock_api_class.assert_not_called()

        first_api = registry.api
        second_api = registry.api

    mock_api_class.assert_called_once_with()
    assert first_api is second_api


def test_project_path_uses_entity(tmp_path: Path) -> None:
    """CA-02: check the project path with and without an entity."""
    with patch(WANDB_API_PATH) as mock_api_class:
        mock_api_class.return_value.default_entity = "default-entity"
        registry_without_entity = WandbAutoencoderRegistry(
            project_name="test-project",
            env_file_path=str(tmp_path / ".env"),
            device="cpu",
        )

        assert (
            registry_without_entity.project_path
            == "default-entity/test-project"
        )
        assert (
            build_registry(tmp_path).project_path
            == "test-entity/test-project"
        )


def test_list_available_artifacts_filters_collections(
    tmp_path: Path,
) -> None:
    """CA-03: check that foreign and incomplete artifacts are skipped."""
    incomplete_metadata = build_metadata("AAPL")
    del incomplete_metadata["anomaly_threshold"]
    valid_artifact = build_fake_artifact(
        "simple-autoencoder-petr4.sa",
        build_metadata("PETR4.SA"),
    )
    with patch(WANDB_API_PATH) as mock_api_class:
        mock_api = configure_project(
            mock_api_class,
            {
                "simple-autoencoder-petr4.sa": valid_artifact,
                "simple-autoencoder-aapl": build_fake_artifact(
                    "simple-autoencoder-aapl",
                    incomplete_metadata,
                ),
                "regression-model-aapl": None,
            },
        )

        available_artifacts = build_registry(
            tmp_path
        ).list_available_artifacts()

    assert available_artifacts == [valid_artifact]
    mock_api.artifact_collections.assert_called_once_with(
        project_name="test-entity/test-project",
        type_name="model",
    )
    requested_paths = [
        call.args[0] for call in mock_api.artifact.call_args_list
    ]
    assert "test-entity/test-project/regression-model-aapl:latest" not in (
        requested_paths
    )


def test_load_latest_model_picks_newest_artifact(tmp_path: Path) -> None:
    """CA-04: check that the newest created_at artifact is loaded."""
    with patch(WANDB_API_PATH) as mock_api_class:
        configure_project(
            mock_api_class,
            {
                "simple-autoencoder-aapl": build_fake_artifact(
                    "simple-autoencoder-aapl",
                    build_metadata("AAPL"),
                    created_at="2026-08-01T00:00:00",
                ),
                "simple-autoencoder-petr4.sa": build_fake_artifact(
                    "simple-autoencoder-petr4.sa",
                    build_metadata("PETR4.SA"),
                    created_at="2026-09-01T00:00:00",
                ),
            },
        )

        predictor = build_registry(tmp_path).load_latest_model()

    assert predictor.metadata.ticker == "PETR4.SA"


def test_load_latest_model_without_artifacts_raises(
    tmp_path: Path,
) -> None:
    """CA-05: check the error when no servable artifact exists."""
    with patch(WANDB_API_PATH) as mock_api_class:
        configure_project(mock_api_class, {})

        with pytest.raises(
            LookupError,
            match="No servable autoencoder artifact found in "
            "'test-entity/test-project'",
        ):
            build_registry(tmp_path).load_latest_model()


def test_load_model_builds_predictor_from_metadata(tmp_path: Path) -> None:
    """CA-06, CA-07, CA-08: check path, metadata, name and weights."""
    fake_artifact = build_fake_artifact(
        "simple-autoencoder-petr4.sa",
        build_metadata("PETR4.SA"),
        version="v2",
    )
    with patch(WANDB_API_PATH) as mock_api_class:
        mock_api_class.return_value.artifact.return_value = fake_artifact

        predictor = build_registry(tmp_path).load_model("PETR4.SA")

    mock_api_class.return_value.artifact.assert_called_once_with(
        "test-entity/test-project/simple-autoencoder-petr4.sa:latest",
        type="model",
    )
    assert isinstance(predictor, AutoencoderPredictor)
    assert predictor.metadata.window_size == 5
    assert predictor.metadata.return_train_mean == 0.001
    assert predictor.metadata.return_train_std == 0.02
    assert predictor.metadata.anomaly_threshold == 0.75
    assert predictor.metadata.artifact_name == (
        "simple-autoencoder-petr4.sa:v2"
    )
    assert predictor.model.encoder[0].out_features == 8
    assert predictor.model.encoder[2].out_features == 2
    for parameter_name, parameter_value in (
        predictor.model.state_dict().items()
    ):
        assert torch.equal(
            parameter_value,
            fake_artifact.state_dict[parameter_name],
        )
    assert (tmp_path / "downloads" / "simple-autoencoder-petr4.sa-v2").is_dir()


def test_load_model_with_incomplete_metadata_raises(
    tmp_path: Path,
) -> None:
    """CA-09: check the error for missing required metadata."""
    incomplete_metadata = build_metadata("AAPL")
    del incomplete_metadata["window_size"]
    with patch(WANDB_API_PATH) as mock_api_class:
        mock_api_class.return_value.artifact.return_value = (
            build_fake_artifact("simple-autoencoder-aapl", incomplete_metadata)
        )

        with pytest.raises(
            ValueError,
            match="is missing metadata required for inference",
        ):
            build_registry(tmp_path).load_model("AAPL")


def test_artifact_without_weights_raises(tmp_path: Path) -> None:
    """CA-10: check the error when the artifact has no .pt file."""
    with patch(WANDB_API_PATH) as mock_api_class:
        mock_api_class.return_value.artifact.return_value = (
            build_fake_artifact(
                "simple-autoencoder-aapl",
                build_metadata("AAPL"),
                write_weights=False,
            )
        )

        with pytest.raises(FileNotFoundError, match="has no .pt weights"):
            build_registry(tmp_path).load_model("AAPL")


def test_load_all_models_loads_every_artifact(tmp_path: Path) -> None:
    """CA-11: check that every valid artifact is loaded and listed."""
    with patch(WANDB_API_PATH) as mock_api_class:
        configure_project(
            mock_api_class,
            {
                "simple-autoencoder-aapl": build_fake_artifact(
                    "simple-autoencoder-aapl",
                    build_metadata("AAPL"),
                ),
                "simple-autoencoder-petr4.sa": build_fake_artifact(
                    "simple-autoencoder-petr4.sa",
                    build_metadata("PETR4.SA"),
                ),
            },
        )
        registry = build_registry(tmp_path)

        loaded_tickers = registry.load_all_models()

    assert loaded_tickers == ["AAPL", "PETR4.SA"]
    assert registry.list_loaded_tickers() == ["AAPL", "PETR4.SA"]
    assert registry.get_model("AAPL").metadata.ticker == "AAPL"


def test_get_model_of_unloaded_ticker_raises(tmp_path: Path) -> None:
    """CA-12: check the KeyError without any wandb access."""
    with patch(WANDB_API_PATH) as mock_api_class:
        registry = build_registry(tmp_path)

        with pytest.raises(KeyError, match="No model loaded for ticker"):
            registry.get_model("AAPL")

    mock_api_class.assert_not_called()
