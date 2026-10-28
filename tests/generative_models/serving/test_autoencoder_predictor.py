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

WINDOW_SIZE = 5


def build_predictor(
    anomaly_threshold: float = 0.5,
    return_train_mean: float = 0.0,
    return_train_std: float = 1.0,
) -> AutoencoderPredictor:
    """Create a predictor around an untrained autoencoder.

    Parameters
    ----------
    anomaly_threshold : float
        Threshold stored in the metadata.
    return_train_mean : float
        Normalization mean stored in the metadata.
    return_train_std : float
        Normalization standard deviation stored in the metadata.

    Returns
    -------
    AutoencoderPredictor
        Predictor with a seeded `SimpleAutoencoder(input_dim=5)`.
    """
    torch.manual_seed(0)
    metadata = AutoencoderModelMetadata(
        ticker="PETR4.SA",
        window_size=WINDOW_SIZE,
        return_train_mean=return_train_mean,
        return_train_std=return_train_std,
        anomaly_threshold=anomaly_threshold,
        artifact_name="simple-autoencoder-petr4.sa:v0",
    )
    return AutoencoderPredictor(
        model=SimpleAutoencoder(input_dim=WINDOW_SIZE),
        metadata=metadata,
    )


def build_log_returns(number_of_values: int) -> np.ndarray:
    """Create a reproducible synthetic log-return series.

    Parameters
    ----------
    number_of_values : int
        Length of the series.

    Returns
    -------
    np.ndarray
        Float32 array of small random log-returns.
    """
    random_generator = np.random.default_rng(7)
    return random_generator.normal(0.0, 0.02, number_of_values).astype(
        np.float32
    )


def test_constructor_puts_model_in_eval_mode() -> None:
    """CA-01: check that the model is in evaluation mode."""
    predictor = build_predictor()

    assert predictor.model.training is False


def test_predict_returns_one_entry_per_window() -> None:
    """CA-02: check the window count and window end indices."""
    prediction = build_predictor().predict(build_log_returns(12))

    assert prediction.reconstruction_scores.shape == (8,)
    assert prediction.is_outlier.shape == (8,)
    np.testing.assert_array_equal(
        prediction.window_end_indices,
        np.arange(4, 12),
    )


def test_threshold_controls_outlier_flags() -> None:
    """CA-03: check that the threshold decides every outlier flag."""
    log_returns = build_log_returns(12)

    zero_threshold_prediction = build_predictor(0.0).predict(log_returns)
    high_threshold_prediction = build_predictor(1e9).predict(log_returns)

    assert zero_threshold_prediction.is_outlier.all()
    assert not high_threshold_prediction.is_outlier.any()


def test_score_equal_to_threshold_is_not_outlier() -> None:
    """CA-04: check that the comparison with the threshold is strict."""
    log_returns = build_log_returns(WINDOW_SIZE)
    first_score = build_predictor().predict(log_returns)
    predictor = build_predictor(
        float(first_score.reconstruction_scores[0])
    )

    prediction = predictor.predict(log_returns)

    assert not prediction.is_outlier[0]


def test_predict_normalizes_with_metadata_statistics() -> None:
    """CA-05: check that the training statistics normalize the input."""
    log_returns = build_log_returns(12)
    predictor = build_predictor(
        return_train_mean=0.001,
        return_train_std=0.02,
    )
    normalized_windows = torch.tensor(
        (log_returns - 0.001) / 0.02
    ).unfold(0, WINDOW_SIZE, 1)

    prediction = predictor.predict(log_returns)

    np.testing.assert_allclose(
        prediction.reconstruction_scores,
        compute_reconstruction_scores(predictor.model, normalized_windows),
        rtol=1e-5,
    )


def test_predict_rejects_short_series() -> None:
    """CA-06: check the error for fewer log-returns than window_size."""
    with pytest.raises(
        ValueError,
        match="At least 5 log-returns are required",
    ):
        build_predictor().predict(build_log_returns(4))


def test_predict_from_prices_matches_log_returns() -> None:
    """CA-07: check that prices are converted into log-returns."""
    predictor = build_predictor()
    price_values = 100.0 * np.exp(np.cumsum(build_log_returns(13)))

    price_prediction = predictor.predict_from_prices(price_values)
    log_return_prediction = predictor.predict(np.diff(np.log(price_values)))

    np.testing.assert_allclose(
        price_prediction.reconstruction_scores,
        log_return_prediction.reconstruction_scores,
        rtol=1e-4,
        atol=1e-6,
    )


def test_predict_from_prices_rejects_short_series() -> None:
    """Check the error for fewer prices than window_size + 1."""
    with pytest.raises(ValueError, match="At least 5 log-returns"):
        build_predictor().predict_from_prices(np.ones(WINDOW_SIZE))


def test_predict_accepts_several_input_types() -> None:
    """CA-08: check numpy, tensor and read-only inputs stay unchanged."""
    predictor = build_predictor()
    log_returns = build_log_returns(12)
    read_only_log_returns = log_returns.copy()
    read_only_log_returns.setflags(write=False)
    gradient_tensor = torch.tensor(log_returns, requires_grad=True)
    expected_scores = predictor.predict(log_returns).reconstruction_scores

    for log_return_input in (read_only_log_returns, gradient_tensor):
        prediction = predictor.predict(log_return_input)
        np.testing.assert_allclose(
            prediction.reconstruction_scores,
            expected_scores,
        )

    np.testing.assert_array_equal(read_only_log_returns, log_returns)
    np.testing.assert_array_equal(
        gradient_tensor.detach().numpy(),
        log_returns,
    )
