"""Unit tests for the train_autoencoders script."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "train_autoencoders.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "train_autoencoders", SCRIPT_PATH
)
train_autoencoders = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(train_autoencoders)


def _fake_train_model(self, ticker) -> None:
    self.training_results[ticker] = SimpleNamespace(validation_loss=0.5)


def test_main_builds_manager_from_script_configuration() -> None:
    """Check that the manager receives the script constants."""
    with patch.object(
        train_autoencoders.AutoencoderTickerModelManager,
        "train_model",
        _fake_train_model,
    ):
        manager = train_autoencoders.main()

    assert manager.tickers == train_autoencoders.TICKERS
    assert manager.training_config is train_autoencoders.TRAINING_CONFIG
    assert manager.project_name == train_autoencoders.PROJECT_NAME


def test_main_trains_every_configured_ticker() -> None:
    """Check that each ticker is trained once, in the configured order."""
    with patch.object(
        train_autoencoders.AutoencoderTickerModelManager,
        "train_model",
        _fake_train_model,
    ):
        manager = train_autoencoders.main()

    assert list(manager.training_results) == list(train_autoencoders.TICKERS)
