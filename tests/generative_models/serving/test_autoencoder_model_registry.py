"""Unit tests for AutoencoderModelRegistry."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from wandb.errors import CommError

from generative_models.models.autoencoder.simple_autoencoder import (
    SimpleAutoencoder,
)
from generative_models.serving.autoencoder_model_registry import (
    AutoencoderModelRegistry,
)

MODULE_PATH = "generative_models.serving.autoencoder_model_registry"


def _build_state_dict_writer(
    input_dim: int = 10,
    hidden_dim: int = 16,
    latent_dim: int = 4,
):
    """Build a fake `artifact.download` that writes a model file."""

    def download(root: str) -> str:
        torch.manual_seed(0)
        autoencoder = SimpleAutoencoder(input_dim, hidden_dim, latent_dim)
        torch.save(
            autoencoder.state_dict(), f"{root}/simple_autoencoder_X.pt"
        )
        return root

    return download


def _build_registry(**kwargs) -> AutoencoderModelRegistry:
    return AutoencoderModelRegistry(entity="test-entity", **kwargs)


@pytest.fixture(name="mock_api")
def fixture_mock_api():
    """Patch wandb.Api and make every artifact download a model file."""
    with patch(f"{MODULE_PATH}.wandb") as mock_wandb:
        api = mock_wandb.Api.return_value
        api.artifact.return_value.download.side_effect = (
            _build_state_dict_writer()
        )
        yield api


def test_constructor_starts_with_no_models_and_no_wandb_call() -> None:
    """Check that construction is lazy: nothing is loaded or fetched."""
    with patch(f"{MODULE_PATH}.wandb") as mock_wandb:
        registry = _build_registry()

    assert not registry.models
    assert not registry.list_loaded_tickers()
    mock_wandb.Api.assert_not_called()


def test_load_model_reads_artifact_of_the_ticker(
    mock_api
) -> None:
    """Check the artifact path built from entity, project and ticker."""
    registry = _build_registry()

    registry.load_model("AAPL")

    mock_api.artifact.assert_called_once_with(
        "test-entity/autoencoder-ticker-models/simple-autoencoder-aapl:latest"
    )


def test_load_model_uses_default_entity_when_not_given(
    mock_api
) -> None:
    """Check that the authenticated user's entity is the fallback."""
    mock_api.default_entity = "default-user"
    registry = AutoencoderModelRegistry(artifact_alias="v0")

    registry.load_model("MSFT")

    mock_api.artifact.assert_called_once_with(
        "default-user/autoencoder-ticker-models/simple-autoencoder-msft:v0"
    )


def test_load_model_keeps_model_in_memory_in_eval_mode(
    mock_api
) -> None:
    """Check that a loaded model is stored and set to evaluation mode."""
    registry = _build_registry()

    autoencoder = registry.load_model("AAPL")

    assert mock_api.artifact.return_value.download.called
    assert isinstance(autoencoder, SimpleAutoencoder)
    assert not autoencoder.training
    assert registry.list_loaded_tickers() == ["AAPL"]


def test_load_model_restores_downloaded_weights(mock_api) -> None:
    """Check that the loaded model reproduces the artifact weights."""
    registry = _build_registry()
    torch.manual_seed(0)
    expected_model = SimpleAutoencoder(10, 16, 4)
    input_tensor = torch.randn(3, 10)

    loaded_model = registry.load_model("AAPL")

    assert torch.allclose(
        loaded_model(input_tensor), expected_model(input_tensor)
    )
    assert mock_api is not None


def test_load_model_infers_architecture_from_weights(
    mock_api
) -> None:
    """Check that window, hidden and latent sizes come from the file."""
    mock_api.artifact.return_value.download.side_effect = (
        _build_state_dict_writer(input_dim=20, hidden_dim=8, latent_dim=2)
    )
    registry = _build_registry()

    autoencoder = registry.load_model("MSFT")

    assert autoencoder.encoder[0].in_features == 20
    assert autoencoder.encoder[0].out_features == 8
    assert autoencoder.encoder[2].out_features == 2


def test_load_model_raises_lookup_error_when_artifact_is_missing(
    mock_api
) -> None:
    """Check that a ticker without artifact raises a clear error."""
    mock_api.artifact.side_effect = CommError("artifact not found")
    registry = _build_registry()

    with pytest.raises(LookupError, match="AAPL"):
        registry.load_model("AAPL")

    assert not registry.models


def test_get_model_selects_model_by_ticker(mock_api) -> None:
    """Check that each ticker returns its own distinct model."""
    writers = {
        "aapl": _build_state_dict_writer(hidden_dim=16),
        "msft": _build_state_dict_writer(hidden_dim=8),
    }

    def fake_artifact(artifact_path: str) -> MagicMock:
        ticker = artifact_path.split("simple-autoencoder-")[1].split(":")[0]
        artifact = MagicMock()
        artifact.download.side_effect = writers[ticker]
        return artifact

    mock_api.artifact.side_effect = fake_artifact
    registry = _build_registry()
    registry.load_model("AAPL")
    registry.load_model("MSFT")

    assert registry.get_model("AAPL").encoder[0].out_features == 16
    assert registry.get_model("MSFT").encoder[0].out_features == 8


def test_get_model_is_case_insensitive(mock_api) -> None:
    """Check that the ticker is normalized to upper case."""
    registry = _build_registry()
    registry.load_model("aapl")

    assert registry.get_model(" aapl ") is registry.get_model("AAPL")
    assert mock_api.artifact.call_count == 1


def test_get_model_does_not_contact_wandb_again(mock_api) -> None:
    """Check that a loaded model is served from memory."""
    registry = _build_registry()
    autoencoder = registry.load_model("AAPL")
    mock_api.reset_mock()

    assert registry.get_model("AAPL") is autoencoder
    mock_api.artifact.assert_not_called()


def test_get_model_raises_for_unloaded_ticker(mock_api) -> None:
    """Check that requesting a ticker that was not loaded raises."""
    registry = _build_registry()

    with pytest.raises(KeyError, match="AAPL"):
        registry.get_model("AAPL")

    mock_api.artifact.assert_not_called()


def test_load_all_models_loads_every_model_artifact(
    mock_api
) -> None:
    """Check that all model collections of the project are loaded."""
    mock_api.artifact_type.return_value.collections.return_value = [
        SimpleNamespace(name="simple-autoencoder-nvda"),
        SimpleNamespace(name="simple-autoencoder-aapl"),
        SimpleNamespace(name="other-model-msft"),
    ]
    registry = _build_registry()

    loaded_tickers = registry.load_all_models()

    mock_api.artifact_type.assert_called_once_with(
        "model", "test-entity/autoencoder-ticker-models"
    )
    assert loaded_tickers == ["AAPL", "NVDA"]
    assert registry.list_loaded_tickers() == ["AAPL", "NVDA"]


def test_load_all_models_returns_empty_list_without_artifacts(
    mock_api
) -> None:
    """Check that a project without models loads nothing."""
    mock_api.artifact_type.return_value.collections.return_value = []
    registry = _build_registry()

    assert registry.load_all_models() == []
    mock_api.artifact.assert_not_called()
