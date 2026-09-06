"""Unit tests for WandbExperimentTracker."""

from unittest.mock import MagicMock, patch

import torch
from torch import nn

from generative_models.tracking.wandb_tracker import WandbExperimentTracker


def _make_tracker(monkeypatch, tmp_path) -> WandbExperimentTracker:
    """Build a tracker instance backed by an empty temporary .env file."""
    env_file_path = tmp_path / ".env"
    env_file_path.write_text("", encoding="utf-8")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    return WandbExperimentTracker(
        project_name="test-project",
        config={"learning_rate": 0.001},
        env_file_path=str(env_file_path),
    )


def test_init_reads_missing_api_key_as_none(monkeypatch, tmp_path) -> None:
    """Check that no WANDB_API_KEY in the environment yields None."""
    tracker = _make_tracker(monkeypatch, tmp_path)

    assert tracker.wandb_api_key is None
    assert tracker.wandb_mode == "disabled"
    assert tracker.wandb_run is None


@patch("generative_models.tracking.wandb_tracker.wandb")
def test_start_run_uses_anonymous_login_without_api_key(
    mock_wandb, monkeypatch, tmp_path
) -> None:
    """Check that start_run logs in anonymously when no key is set."""
    tracker = _make_tracker(monkeypatch, tmp_path)

    tracker.start_run()

    mock_wandb.login.assert_called_once_with(
        anonymous="allow",
        timeout=tracker.login_timeout_seconds,
    )
    mock_wandb.init.assert_called_once_with(
        project="test-project",
        mode="online",
        config={"learning_rate": 0.001},
    )
    assert tracker.wandb_run is mock_wandb.init.return_value


@patch("generative_models.tracking.wandb_tracker.wandb")
def test_start_run_falls_back_to_disabled_on_login_error(
    mock_wandb, monkeypatch, tmp_path
) -> None:
    """Check that a failed login disables tracking instead of raising."""
    mock_wandb.login.side_effect = RuntimeError("network error")
    tracker = _make_tracker(monkeypatch, tmp_path)

    tracker.start_run()

    assert tracker.wandb_mode == "disabled"
    mock_wandb.init.assert_called_once_with(
        project="test-project",
        mode="disabled",
        config={"learning_rate": 0.001},
    )


@patch("generative_models.tracking.wandb_tracker.wandb")
def test_log_metrics_forwards_to_wandb(
    mock_wandb, monkeypatch, tmp_path
) -> None:
    """Check that log_metrics forwards the metrics dict to wandb.log."""
    tracker = _make_tracker(monkeypatch, tmp_path)

    tracker.log_metrics({"epoch": 1, "train/loss_mse": 0.0123})

    mock_wandb.log.assert_called_once_with(
        {"epoch": 1, "train/loss_mse": 0.0123}
    )


@patch("generative_models.tracking.wandb_tracker.wandb")
def test_log_model_skips_registration_without_active_run(
    mock_wandb, monkeypatch, tmp_path
) -> None:
    """Check that log_model is a no-op when no run has been started."""
    tracker = _make_tracker(monkeypatch, tmp_path)
    model = nn.Linear(2, 1)
    model_file_path = tmp_path / "model.pt"

    tracker.log_model(
        model=model,
        model_name="test-model",
        model_file_path=str(model_file_path),
    )

    mock_wandb.Artifact.assert_not_called()
    assert not model_file_path.exists()


@patch("generative_models.tracking.wandb_tracker.wandb")
def test_log_model_saves_weights_and_logs_artifact(
    mock_wandb, monkeypatch, tmp_path
) -> None:
    """Check that log_model saves the state_dict and logs an artifact."""
    tracker = _make_tracker(monkeypatch, tmp_path)
    tracker.wandb_run = MagicMock()
    model = nn.Linear(2, 1)
    model_file_path = tmp_path / "model.pt"

    tracker.log_model(
        model=model,
        model_name="test-model",
        model_file_path=str(model_file_path),
        metadata={"hidden_layer_size": 8},
    )

    assert model_file_path.exists()
    saved_state_dict = torch.load(model_file_path)
    assert saved_state_dict.keys() == model.state_dict().keys()

    mock_wandb.Artifact.assert_called_once_with(
        name="test-model",
        type="model",
        metadata={"hidden_layer_size": 8},
    )
    artifact = mock_wandb.Artifact.return_value
    artifact.add_file.assert_called_once_with(str(model_file_path))
    tracker.wandb_run.log_artifact.assert_called_once_with(artifact)


@patch("generative_models.tracking.wandb_tracker.wandb")
def test_finish_run_finishes_active_run(
    _mock_wandb, monkeypatch, tmp_path
) -> None:
    """Check that finish_run calls finish on the active wandb run."""
    tracker = _make_tracker(monkeypatch, tmp_path)
    tracker.wandb_run = MagicMock()

    tracker.finish_run()

    tracker.wandb_run.finish.assert_called_once()


def test_finish_run_without_active_run_does_not_raise(
    monkeypatch, tmp_path
) -> None:
    """Check that finish_run is a no-op when no run has been started."""
    tracker = _make_tracker(monkeypatch, tmp_path)

    tracker.finish_run()
