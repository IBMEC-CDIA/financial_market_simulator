"""Unit tests for AutoencoderPredictor."""

import numpy as np
import pytest
import torch

from generative_models.models.autoencoder.trainer import (
    compute_reconstruction_scores,
)


def test_constructor_puts_model_in_eval_mode(make_predictor) -> None:
    """Check that the predictor leaves the model in evaluation mode."""
    predictor = make_predictor()

    assert predictor.model.training is False


def test_predict_returns_one_window_per_position(make_predictor) -> None:
    """Check the number of windows and their end indices for 12 returns."""
    predictor = make_predictor()
    log_return_values = np.linspace(-0.02, 0.02, 12)

    prediction = predictor.predict(log_return_values)

    assert len(prediction.reconstruction_scores) == 8
    assert len(prediction.is_outlier) == 8
    assert prediction.window_end_indices.tolist() == list(range(4, 12))


def test_threshold_zero_flags_every_window_and_high_threshold_none(
    make_predictor,
) -> None:
    """Check the outlier flags with extreme thresholds."""
    log_return_values = np.linspace(-0.02, 0.02, 12)

    all_outliers = make_predictor(anomaly_threshold=0.0).predict(
        log_return_values
    )
    no_outliers = make_predictor(anomaly_threshold=1e9).predict(
        log_return_values
    )

    assert bool(all_outliers.is_outlier.all())
    assert not bool(no_outliers.is_outlier.any())


def test_score_equal_to_threshold_is_not_an_outlier(make_predictor) -> None:
    """Check that the comparison with the threshold is strict."""
    log_return_values = np.linspace(-0.02, 0.02, 12)
    scores = make_predictor().predict(log_return_values).reconstruction_scores
    predictor = make_predictor(anomaly_threshold=float(scores[0]))

    prediction = predictor.predict(log_return_values)

    assert not bool(prediction.is_outlier[0])


def test_predict_uses_saved_normalization_statistics(make_predictor) -> None:
    """Check that scores match a manual normalization with the metadata."""
    mean_value = 0.001
    std_value = 0.02
    predictor = make_predictor(
        return_train_mean=mean_value,
        return_train_std=std_value,
    )
    log_return_values = np.linspace(-0.03, 0.03, 12)

    prediction = predictor.predict(log_return_values)

    normalized_tensor = (
        torch.tensor(log_return_values, dtype=torch.float32) - mean_value
    ) / std_value
    manual_windows = torch.stack(
        [normalized_tensor[start : start + 5] for start in range(8)]
    )
    expected_scores = compute_reconstruction_scores(
        predictor.model,
        manual_windows,
    )
    np.testing.assert_allclose(
        prediction.reconstruction_scores,
        expected_scores,
        rtol=1e-5,
    )


def test_predict_raises_when_too_few_log_returns(make_predictor) -> None:
    """Check that fewer log-returns than the window size are rejected."""
    predictor = make_predictor()

    with pytest.raises(
        ValueError, match="At least 5 log-returns are required"
    ):
        predictor.predict(np.array([0.01, 0.02, 0.03, 0.04]))


def test_predict_from_prices_requires_window_size_plus_one_prices(
    make_predictor,
) -> None:
    """Check that too few prices give a message in terms of prices."""
    predictor = make_predictor()

    with pytest.raises(
        ValueError,
        match="At least 6 prices are required, got 5. The model window size",
    ):
        predictor.predict_from_prices(np.linspace(100.0, 101.0, 5))


def test_predict_from_prices_matches_predict_on_log_returns(
    make_predictor,
) -> None:
    """Check that prices give the same scores as their log-returns."""
    predictor = make_predictor()
    price_values = 100 * np.exp(np.cumsum(np.full(12, 0.01)))

    from_prices = predictor.predict_from_prices(price_values)
    from_log_returns = predictor.predict(np.diff(np.log(price_values)))

    np.testing.assert_allclose(
        from_prices.reconstruction_scores,
        from_log_returns.reconstruction_scores,
        rtol=1e-5,
        atol=1e-6,
    )


def test_predict_accepts_read_only_arrays_and_tensors(make_predictor) -> None:
    """Check that read-only inputs work and are never modified."""
    predictor = make_predictor()
    read_only_values = np.linspace(-0.02, 0.02, 12)
    read_only_values.setflags(write=False)
    tensor_values = torch.linspace(-0.02, 0.02, 12, requires_grad=True)
    original_values = read_only_values.copy()

    array_prediction = predictor.predict(read_only_values)
    tensor_prediction = predictor.predict(tensor_values)

    np.testing.assert_array_equal(read_only_values, original_values)
    assert len(array_prediction.reconstruction_scores) == 8
    assert len(tensor_prediction.reconstruction_scores) == 8


def test_predict_rejects_two_dimensional_input(make_predictor) -> None:
    """Check that a series must be one-dimensional."""
    predictor = make_predictor()

    with pytest.raises(ValueError, match="one-dimensional"):
        predictor.predict(np.zeros((2, 6)))
