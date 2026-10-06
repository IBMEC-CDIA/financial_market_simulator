"""Unit tests for AutoencoderPredictor."""

import numpy as np
import pytest
import torch

from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPredictor,
)


def _make_predictor(anomaly_threshold: float = 1.0) -> AutoencoderPredictor:
    """Build a predictor around an untrained autoencoder."""
    torch.manual_seed(0)
    return AutoencoderPredictor(
        model=SimpleAutoencoder(input_dim=5),
        metadata=AutoencoderModelMetadata(
            ticker="AAPL",
            window_size=5,
            return_train_mean=0.0,
            return_train_std=0.01,
            anomaly_threshold=anomaly_threshold,
            artifact_name="simple-autoencoder-aapl:v0",
        ),
    )


def test_predict_returns_one_score_per_window() -> None:
    """Check the output sizes and the window end indices."""
    predictor = _make_predictor()
    log_return_values = np.random.default_rng(0).normal(0, 0.01, size=20)

    prediction = predictor.predict(log_return_values)

    assert prediction.reconstruction_scores.shape == (16,)
    assert prediction.is_outlier.shape == (16,)
    np.testing.assert_array_equal(
        prediction.window_end_indices,
        np.arange(4, 20),
    )


def test_predict_flags_scores_above_threshold() -> None:
    """Check that the outlier flag follows the anomaly threshold."""
    predictor = _make_predictor(anomaly_threshold=0.0)

    prediction = predictor.predict(torch.randn(12) * 0.01)

    np.testing.assert_array_equal(
        prediction.is_outlier,
        prediction.reconstruction_scores > 0.0,
    )


def test_predict_normalizes_with_training_statistics() -> None:
    """Check that the scores match a manual normalization."""
    predictor = _make_predictor()
    log_return_values = torch.randn(5) * 0.01

    prediction = predictor.predict(log_return_values)

    normalized_window = (log_return_values / 0.01).unsqueeze(0)
    with torch.no_grad():
        expected_score = (
            (predictor.model(normalized_window) - normalized_window) ** 2
        ).mean()
    assert prediction.reconstruction_scores[0] == pytest.approx(
        expected_score.item(),
        rel=1e-5,
    )


def test_predict_raises_when_series_is_too_short() -> None:
    """Check that fewer log-returns than the window size are rejected."""
    predictor = _make_predictor()

    with pytest.raises(ValueError, match="At least 5"):
        predictor.predict(np.zeros(4))


def test_predict_from_prices_uses_log_returns() -> None:
    """Check that prices and their log-returns give the same result."""
    predictor = _make_predictor()
    price_values = 100.0 * np.exp(
        np.cumsum(np.random.default_rng(1).normal(0, 0.01, size=15))
    )

    price_prediction = predictor.predict_from_prices(price_values)
    return_prediction = predictor.predict(np.diff(np.log(price_values)))

    np.testing.assert_allclose(
        price_prediction.reconstruction_scores,
        return_prediction.reconstruction_scores,
        rtol=1e-4,
    )
