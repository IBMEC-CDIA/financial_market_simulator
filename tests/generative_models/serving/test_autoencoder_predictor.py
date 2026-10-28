"""Unit tests for AutoencoderPredictor."""

import numpy as np
import pytest
import torch

from generative_models.models.autoencoder.autoencoder import (
    SimpleAutoencoder,
)
from generative_models.models.autoencoder.trainer import (
    compute_reconstruction_scores,
)
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPredictor,
)

WINDOW_SIZE = 5
RETURN_TRAIN_MEAN = 0.001
RETURN_TRAIN_STD = 0.02


def build_metadata(anomaly_threshold: float) -> AutoencoderModelMetadata:
    """Build model metadata for the tests.

    Parameters
    ----------
    anomaly_threshold : float
        Reconstruction error above which a window is an outlier.

    Returns
    -------
    AutoencoderModelMetadata
        Metadata describing a model trained on PETR4.SA.
    """
    return AutoencoderModelMetadata(
        ticker="PETR4.SA",
        window_size=WINDOW_SIZE,
        return_train_mean=RETURN_TRAIN_MEAN,
        return_train_std=RETURN_TRAIN_STD,
        anomaly_threshold=anomaly_threshold,
        artifact_name="simple-autoencoder-petr4.sa:v1",
    )


def build_predictor(anomaly_threshold: float = 0.5) -> AutoencoderPredictor:
    """Build a predictor wrapping an untrained autoencoder.

    Parameters
    ----------
    anomaly_threshold : float
        Reconstruction error above which a window is an outlier.

    Returns
    -------
    AutoencoderPredictor
        Predictor on CPU, built from a fixed random seed.
    """
    torch.manual_seed(42)
    model = SimpleAutoencoder(input_dim=WINDOW_SIZE)

    return AutoencoderPredictor(
        model=model,
        metadata=build_metadata(anomaly_threshold),
        device="cpu",
    )


def build_log_returns(number_of_returns: int) -> np.ndarray:
    """Build a synthetic log-return series.

    Parameters
    ----------
    number_of_returns : int
        Number of log-returns to generate.

    Returns
    -------
    np.ndarray
        One-dimensional array of synthetic log-returns.
    """
    random_generator = np.random.default_rng(11)

    return random_generator.normal(
        loc=0.0,
        scale=0.02,
        size=number_of_returns,
    ).astype(np.float32)


def test_constructor_puts_the_model_in_evaluation_mode() -> None:
    """CA-01: the constructor leaves the model in evaluation mode."""
    predictor = build_predictor()

    assert predictor.model.training is False


def test_predict_returns_one_entry_per_window() -> None:
    """CA-02: 12 log-returns produce 8 windows ending at indices 4 to 11."""
    predictor = build_predictor()

    prediction = predictor.predict(build_log_returns(12))

    assert prediction.reconstruction_scores.shape == (8,)
    assert prediction.is_outlier.shape == (8,)
    assert np.array_equal(
        prediction.window_end_indices,
        np.arange(4, 12),
    )


def test_threshold_decides_which_windows_are_outliers() -> None:
    """CA-03: a zero threshold flags every window and a huge one none."""
    log_return_values = build_log_returns(12)

    every_window = build_predictor(anomaly_threshold=0.0).predict(
        log_return_values
    )
    no_window = build_predictor(anomaly_threshold=1e9).predict(
        log_return_values
    )

    assert bool(every_window.is_outlier.all())
    assert not bool(no_window.is_outlier.any())


def test_score_equal_to_the_threshold_is_not_an_outlier() -> None:
    """CA-04: a score exactly equal to the threshold is not an outlier."""
    log_return_values = build_log_returns(12)
    reference = build_predictor(anomaly_threshold=0.0).predict(
        log_return_values
    )
    first_score = float(reference.reconstruction_scores[0])

    prediction = build_predictor(anomaly_threshold=first_score).predict(
        log_return_values
    )

    assert bool(prediction.is_outlier[0]) is False


def test_normalization_uses_the_training_statistics() -> None:
    """CA-05: scores match windows normalized with the metadata."""
    predictor = build_predictor()
    log_return_values = build_log_returns(12)

    prediction = predictor.predict(log_return_values)

    normalized_values = (
        log_return_values - RETURN_TRAIN_MEAN
    ) / RETURN_TRAIN_STD
    expected_windows = torch.from_numpy(
        np.lib.stride_tricks.sliding_window_view(
            normalized_values.astype(np.float32),
            window_shape=WINDOW_SIZE,
        ).copy()
    )
    expected_scores = compute_reconstruction_scores(
        predictor.model,
        expected_windows,
    )

    assert np.allclose(
        prediction.reconstruction_scores,
        expected_scores,
        atol=1e-6,
    )


def test_series_shorter_than_the_window_raises_value_error() -> None:
    """CA-06: fewer log-returns than the window size raises ValueError."""
    predictor = build_predictor()

    with pytest.raises(
        ValueError,
        match="At least 5 log-returns are required",
    ):
        predictor.predict(build_log_returns(4))


def test_predict_from_prices_matches_predict_on_log_returns() -> None:
    """CA-07: scoring prices equals scoring their log-returns."""
    predictor = build_predictor()
    price_values = 100.0 * np.exp(np.cumsum(build_log_returns(12)))

    from_prices = predictor.predict_from_prices(price_values)
    from_log_returns = predictor.predict(np.diff(np.log(price_values)))

    assert np.allclose(
        from_prices.reconstruction_scores,
        from_log_returns.reconstruction_scores,
        atol=1e-6,
    )


def test_predict_accepts_every_input_type_without_mutating_it() -> None:
    """CA-08: arrays, tensors and read-only arrays work and stay intact."""
    predictor = build_predictor()
    log_return_values = build_log_returns(12)

    array_prediction = predictor.predict(log_return_values.copy())

    tensor_input = torch.from_numpy(log_return_values.copy())
    tensor_input.requires_grad_(True)
    tensor_prediction = predictor.predict(tensor_input)

    read_only_values = log_return_values.copy()
    read_only_values.setflags(write=False)
    read_only_prediction = predictor.predict(read_only_values)

    assert np.allclose(
        array_prediction.reconstruction_scores,
        tensor_prediction.reconstruction_scores,
    )
    assert np.allclose(
        array_prediction.reconstruction_scores,
        read_only_prediction.reconstruction_scores,
    )
    assert np.array_equal(read_only_values, log_return_values)
