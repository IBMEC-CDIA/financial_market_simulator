"""Tests of the training of the regression forecaster."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generative_models.serving.forecast import load_regression_forecaster
from generative_models.serving.train_regression_forecaster import (
    TrainingConfig,
    build_windows,
    parse_arguments,
    save_regression_artifact,
    train_regression_forecaster,
)


def random_walk_closes(size: int, seed: int = 0) -> pd.Series:
    """Closing prices of a random walk with a small drift."""
    generator = np.random.default_rng(seed)
    returns = generator.normal(0.0005, 0.01, size)
    values = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=size))


def test_windows_pair_each_input_with_the_value_that_follows_it():
    """Check that windows pair each input with the value that follows it."""
    inputs, targets = build_windows(np.arange(5.0), 2)
    assert inputs.tolist() == [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
    assert targets.ravel().tolist() == [2.0, 3.0, 4.0]


def test_a_series_shorter_than_the_window_gives_no_windows():
    """Check that a series shorter than the window gives no windows."""
    inputs, targets = build_windows(np.arange(2.0), 5)
    assert inputs.shape == (0, 5)
    assert targets.shape == (0, 1)


def test_training_returns_finite_losses_and_normalization():
    """Check that training returns finite losses and normalization."""
    result = train_regression_forecaster(
        random_walk_closes(200),
        TrainingConfig(window_size=5, hidden_layer_size=4, epochs=2),
    )
    assert np.isfinite(result.validation_loss)
    assert np.isfinite(result.baseline_validation_loss)
    assert result.return_train_std > 0
    assert not result.model.training


def test_a_flat_series_is_refused():
    """Check that a flat series is refused."""
    flat = pd.Series(np.full(100, 50.0))
    with pytest.raises(ValueError, match="no variation"):
        train_regression_forecaster(
            flat,
            TrainingConfig(window_size=5, epochs=1),
        )


def test_a_too_short_series_is_refused():
    """Check that a too short series is refused."""
    with pytest.raises(ValueError, match="too short"):
        train_regression_forecaster(
            random_walk_closes(8),
            TrainingConfig(window_size=5, epochs=1),
        )


def test_the_saved_artifact_loads_with_the_same_normalization(tmp_path: Path):
    """Check that the saved artifact loads with the same normalization."""
    result = train_regression_forecaster(
        random_walk_closes(200),
        TrainingConfig(window_size=5, hidden_layer_size=4, epochs=2),
    )
    path = save_regression_artifact(
        result,
        "msft",
        tmp_path,
        TrainingConfig(window_size=5, hidden_layer_size=4),
    )
    assert path == tmp_path / "regression" / "MSFT.pt"
    loaded = load_regression_forecaster("MSFT", tmp_path)
    assert loaded is not None
    assert loaded.input_window_size == 5
    assert loaded.return_train_mean == pytest.approx(result.return_train_mean)
    assert loaded.return_train_std == pytest.approx(result.return_train_std)
    assert loaded.artifact_name == "regression-MSFT"


def test_the_command_line_defaults_follow_the_notebook():
    """Check that the command line defaults follow the notebook."""
    options = parse_arguments(["--ticker", "AAPL"])
    assert options.window_size == 30
    assert options.hidden_layer_size == 64
    assert options.learning_rate == pytest.approx(0.001)
    assert options.output_directory == Path("models")
