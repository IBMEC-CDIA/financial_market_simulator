"""Unit tests for AutoencoderPredictor."""

import numpy as np
import pytest
import torch

from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder
from generative_models.models.autoencoder.trainer import (
    compute_reconstruction_scores,
)
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPredictor,
)


def _make_predictor(
    anomaly_threshold: float = 1.0,
    return_train_mean: float = 0.001,
    return_train_std: float = 0.02,
) -> AutoencoderPredictor:
    """Build a predictor around an untrained autoencoder."""
    torch.manual_seed(0)
    metadata = AutoencoderModelMetadata(
        ticker="PETR4.SA",
        window_size=5,
        return_train_mean=return_train_mean,
        return_train_std=return_train_std,
        anomaly_threshold=anomaly_threshold,
        artifact_name="simple-autoencoder-petr4.sa:v0",
    )
    return AutoencoderPredictor(
        model=SimpleAutoencoder(input_dim=5),
        metadata=metadata,
    )


def _make_log_returns(number_of_returns: int) -> np.ndarray:
    """Build a synthetic log-return series."""
    random_generator = np.random.default_rng(1)
    return random_generator.normal(0, 0.02, number_of_returns).astype(
        np.float32
    )


def test_constructor_sets_eval_mode() -> None:
    """CA-01: the model is stored in evaluation mode."""
    predictor = _make_predictor()

    assert predictor.model.training is False


def test_predict_returns_one_entry_per_window() -> None:
    """CA-02: 12 log-returns yield 8 windows ending at indices 4 to 11."""
    prediction = _make_predictor().predict(_make_log_returns(12))

    assert prediction.reconstruction_scores.shape == (8,)
    assert prediction.is_outlier.shape == (8,)
    assert prediction.window_end_indices.tolist() == list(range(4, 12))


def test_threshold_controls_outlier_flags() -> None:
    """CA-03: threshold 0 flags every window, 1e9 flags none."""
    log_returns = _make_log_returns(12)

    low_prediction = _make_predictor(anomaly_threshold=0.0).predict(
        log_returns
    )
    high_prediction = _make_predictor(anomaly_threshold=1e9).predict(
        log_returns
    )

    assert low_prediction.is_outlier.all()
    assert not high_prediction.is_outlier.any()


def test_score_equal_to_threshold_is_not_outlier() -> None:
    """CA-04: a score exactly equal to the threshold is not an outlier."""
    log_returns = _make_log_returns(12)
    first_prediction = _make_predictor().predict(log_returns)
    first_score = first_prediction.reconstruction_scores[0]

    prediction = _make_predictor(
        anomaly_threshold=float(first_score)
    ).predict(log_returns)

    assert not prediction.is_outlier[0]


def test_predict_normalizes_with_metadata() -> None:
    """CA-05: scores match manually normalized windows."""
    log_returns = _make_log_returns(12)
    predictor = _make_predictor(return_train_mean=0.01, return_train_std=0.5)

    prediction = predictor.predict(log_returns)

    normalized_returns = (torch.tensor(log_returns) - 0.01) / 0.5
    manual_windows = torch.stack(
        [normalized_returns[index:index + 5] for index in range(8)]
    )
    expected_scores = compute_reconstruction_scores(
        predictor.model, manual_windows
    )
    np.testing.assert_allclose(
        prediction.reconstruction_scores, expected_scores, rtol=1e-6
    )


def test_predict_raises_with_short_series() -> None:
    """CA-06: fewer log-returns than window_size raise ValueError."""
    with pytest.raises(ValueError, match="At least 5 log-returns are"):
        _make_predictor().predict(_make_log_returns(4))


def test_predict_from_prices_matches_predict() -> None:
    """CA-07: prices give the same scores as their log-returns."""
    random_generator = np.random.default_rng(2)
    price_values = 100 * np.exp(
        np.cumsum(random_generator.normal(0, 0.01, 13))
    )
    predictor = _make_predictor()

    price_prediction = predictor.predict_from_prices(price_values)
    return_prediction = predictor.predict(np.diff(np.log(price_values)))

    np.testing.assert_allclose(
        price_prediction.reconstruction_scores,
        return_prediction.reconstruction_scores,
        rtol=1e-4,
    )


def test_predict_accepts_inputs_without_changing_them() -> None:
    """CA-08: arrays, tensors and read-only arrays are not modified."""
    predictor = _make_predictor()
    log_returns = _make_log_returns(12)
    read_only_returns = log_returns.copy()
    read_only_returns.setflags(write=False)
    tensor_returns = torch.tensor(log_returns, requires_grad=True)

    array_scores = predictor.predict(log_returns).reconstruction_scores
    read_only_scores = predictor.predict(
        read_only_returns
    ).reconstruction_scores
    tensor_scores = predictor.predict(tensor_returns).reconstruction_scores

    np.testing.assert_allclose(array_scores, read_only_scores)
    np.testing.assert_allclose(array_scores, tensor_scores)
    np.testing.assert_array_equal(log_returns, read_only_returns)
    assert torch.equal(tensor_returns.detach(), torch.tensor(log_returns))
