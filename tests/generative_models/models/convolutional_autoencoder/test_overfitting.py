"""Unit tests for the overfitting report and its evaluation script."""

# pylint: disable=duplicate-code

import json
from pathlib import Path

import pytest

from generative_models.models.convolutional_autoencoder.autoencoder import (
    ConvolutionalAutoencoder,
)
from generative_models.models.convolutional_autoencoder.overfitting import (
    build_overfitting_report,
    build_report_path,
    build_step_summary,
    is_better_than_baseline,
    is_overfitting_within_limit,
    load_overfitting_report,
    main,
    save_overfitting_report,
)
from generative_models.models.convolutional_autoencoder.trainer import (
    ConvolutionalAutoencoderTrainingResult,
)

MODEL_FILE_PATH = Path("artifacts/convolutional-autoencoder-aapl.pt")

EXPECTED_REPORT_KEYS = {
    "ticker",
    "model_file_path",
    "final_train_loss",
    "train_evaluation_loss",
    "validation_loss",
    "overfitting_ratio",
    "test_loss",
    "baseline_test_loss",
    "window_size",
    "hidden_layer_size",
    "latent_layer_size",
    "return_train_mean",
    "return_train_std",
    "anomaly_threshold",
}


def _make_result(
    train_evaluation_loss: float = 0.2,
    validation_loss: float = 0.3,
    test_loss: float | None = 0.4,
    baseline_test_loss: float | None = 0.9,
) -> ConvolutionalAutoencoderTrainingResult:
    """Build a training result with the given errors."""
    return ConvolutionalAutoencoderTrainingResult(
        model=ConvolutionalAutoencoder(input_dim=5),
        training_loss_history=[0.5, 0.25],
        validation_loss=validation_loss,
        train_evaluation_loss=train_evaluation_loss,
        anomaly_threshold=0.9,
        return_train_mean=0.0,
        return_train_std=1.0,
        model_file_path=MODEL_FILE_PATH,
        test_loss=test_loss,
        baseline_test_loss=baseline_test_loss,
        window_size=5,
        hidden_layer_size=16,
        latent_layer_size=4,
    )


def test_build_overfitting_report_computes_the_ratio() -> None:
    """Check that the ratio is the validation loss over the train loss."""
    report = build_overfitting_report("AAPL", _make_result(0.2, 0.3))

    assert report.ticker == "AAPL"
    assert report.final_train_loss == 0.25
    assert report.overfitting_ratio == pytest.approx(1.5)


def test_build_overfitting_report_carries_the_serving_fields() -> None:
    """Check that the sizes and statistics for serving are reported."""
    report = build_overfitting_report("AAPL", _make_result())

    assert report.window_size == 5
    assert report.hidden_layer_size == 16
    assert report.latent_layer_size == 4
    assert report.return_train_mean == pytest.approx(0.0)
    assert report.return_train_std == pytest.approx(1.0)
    assert report.anomaly_threshold == pytest.approx(0.9)


def test_report_has_exactly_the_keys_of_the_specification(
    tmp_path,
) -> None:
    """Check the JSON keys the evaluate job of the pipeline reads."""
    report = build_overfitting_report("AAPL", _make_result())
    report_path = save_overfitting_report(report, tmp_path / "aapl.json")

    report_data = json.loads(report_path.read_text(encoding="utf-8"))

    assert set(report_data.keys()) == EXPECTED_REPORT_KEYS


def test_build_overfitting_report_rejects_non_positive_train_loss() -> None:
    """Check that a zero training loss is rejected, the ratio would fail."""
    with pytest.raises(ValueError, match="must be positive"):
        build_overfitting_report("AAPL", _make_result(0.0, 0.3))


def test_build_report_path_is_next_to_the_model_file() -> None:
    """Check the report name derived from the weights file name."""
    report_path = build_report_path(MODEL_FILE_PATH)

    assert report_path == Path(
        "artifacts/convolutional-autoencoder-aapl-overfitting.json"
    )


def test_save_and_load_report_round_trip(tmp_path) -> None:
    """Check that a saved report is read back with the same values."""
    report = build_overfitting_report("AAPL", _make_result())
    report_path = save_overfitting_report(
        report,
        tmp_path / "reports" / "aapl.json",
    )

    assert load_overfitting_report(report_path) == report


def test_is_overfitting_within_limit_is_inclusive() -> None:
    """Check that a ratio equal to the limit passes the check."""
    report = build_overfitting_report("AAPL", _make_result(0.2, 0.3))

    assert is_overfitting_within_limit(report, 1.5)
    assert not is_overfitting_within_limit(report, 1.4)


def test_is_better_than_baseline_needs_both_losses() -> None:
    """Check that a missing test loss fails the baseline comparison."""
    with_losses = build_overfitting_report("AAPL", _make_result())
    without_test_loss = build_overfitting_report(
        "AAPL",
        _make_result(test_loss=None),
    )

    assert is_better_than_baseline(with_losses)
    assert not is_better_than_baseline(without_test_loss)


def test_main_returns_true_for_a_valid_report(tmp_path) -> None:
    """Check that a report within the limits passes the check."""
    report = build_overfitting_report("AAPL", _make_result())
    report_path = save_overfitting_report(report, tmp_path / "aapl.json")

    assert main(["--report-path", str(report_path)]) is True


def test_main_returns_false_when_ratio_exceeds_limit(tmp_path) -> None:
    """Check that the script fails when the ratio is above the limit."""
    report = build_overfitting_report("AAPL", _make_result(0.2, 0.3))
    report_path = save_overfitting_report(report, tmp_path / "aapl.json")

    passed = main(
        [
            "--report-path",
            str(report_path),
            "--max-overfitting-ratio",
            "1.0",
        ]
    )

    assert passed is False


def test_main_returns_false_when_baseline_is_not_beaten(tmp_path) -> None:
    """Check that a test loss above the baseline fails the check."""
    report = build_overfitting_report(
        "AAPL",
        _make_result(test_loss=0.9, baseline_test_loss=0.4),
    )
    report_path = save_overfitting_report(report, tmp_path / "aapl.json")

    assert main(["--report-path", str(report_path)]) is False


def test_main_returns_false_without_a_test_loss(tmp_path) -> None:
    """Check that a report without a test loss does not pass."""
    report = build_overfitting_report(
        "AAPL",
        _make_result(test_loss=None, baseline_test_loss=None),
    )
    report_path = save_overfitting_report(report, tmp_path / "aapl.json")

    assert main(["--report-path", str(report_path)]) is False


def test_build_step_summary_reports_result_and_ratio() -> None:
    """Check that the Markdown table has the result and the ratio."""
    report = build_overfitting_report("AAPL", _make_result(0.2, 0.3))

    summary = build_step_summary(report, 1.0, passed=False)

    assert "Result: **failed**" in summary
    assert "| Validation / train ratio | 1.500 |" in summary
    assert "| Limit | 1.000 |" in summary


def test_main_appends_summary_to_github_step_summary(
    tmp_path,
    monkeypatch,
) -> None:
    """Check that the summary is written when GITHUB_STEP_SUMMARY is set."""
    report = build_overfitting_report("AAPL", _make_result(0.2, 0.3))
    report_path = save_overfitting_report(report, tmp_path / "aapl.json")
    summary_path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))

    main(["--report-path", str(report_path)])

    assert "## Overfitting check" in summary_path.read_text(
        encoding="utf-8"
    )


def test_main_prints_a_summary_line(tmp_path, capsys) -> None:
    """Check that the summary line has the ticker and the ratio."""
    report = build_overfitting_report("AAPL", _make_result(0.2, 0.3))
    report_path = save_overfitting_report(report, tmp_path / "aapl.json")

    main(["--report-path", str(report_path)])

    printed_output = capsys.readouterr().out
    assert "ticker=AAPL" in printed_output
    assert "overfitting_ratio=1.500" in printed_output
