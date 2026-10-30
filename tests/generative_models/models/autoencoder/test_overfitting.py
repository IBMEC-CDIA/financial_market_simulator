"""Unit tests for the overfitting report and its evaluation script."""

from pathlib import Path

import pytest

from generative_models.models.autoencoder.autoencoder import (
    SimpleAutoencoder,
)
from generative_models.models.autoencoder.overfitting import (
    build_overfitting_report,
    build_report_path,
    is_overfitting_within_limit,
    load_overfitting_report,
    main,
    save_overfitting_report,
)
from generative_models.models.autoencoder.trainer import (
    AutoencoderTrainingResult,
)


def _make_result(
    train_evaluation_loss: float = 0.2,
    validation_loss: float = 0.3,
) -> AutoencoderTrainingResult:
    """Build a training result with the given errors."""
    return AutoencoderTrainingResult(
        model=SimpleAutoencoder(input_dim=5),
        training_loss_history=[0.5, 0.25],
        validation_loss=validation_loss,
        train_evaluation_loss=train_evaluation_loss,
        anomaly_threshold=0.9,
        return_train_mean=0.0,
        return_train_std=1.0,
        model_file_path=Path("artifacts/simple-autoencoder-aapl.pt"),
    )


def test_build_overfitting_report_computes_validation_to_train_ratio() -> None:
    """Check that the ratio is the validation loss over the train loss."""
    report = build_overfitting_report("AAPL", _make_result(0.2, 0.3))

    assert report.ticker == "AAPL"
    assert report.final_train_loss == 0.25
    assert report.overfitting_ratio == pytest.approx(1.5)


def test_build_overfitting_report_rejects_non_positive_train_loss() -> None:
    """Check that a zero training loss is rejected, as the ratio would fail."""
    with pytest.raises(ValueError, match="must be positive"):
        build_overfitting_report("AAPL", _make_result(0.0, 0.3))


def test_build_report_path_is_next_to_the_model_file() -> None:
    """Check the report name derived from the weights file name."""
    report_path = build_report_path("artifacts/simple-autoencoder-aapl.pt")

    assert report_path == Path(
        "artifacts/simple-autoencoder-aapl-overfitting.json"
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


def test_main_prints_a_summary_line(tmp_path, capsys) -> None:
    """Check that the summary line has the ticker and the ratio."""
    report = build_overfitting_report("AAPL", _make_result(0.2, 0.3))
    report_path = save_overfitting_report(report, tmp_path / "aapl.json")

    main(["--report-path", str(report_path)])

    printed_output = capsys.readouterr().out
    assert "ticker=AAPL" in printed_output
    assert "overfitting_ratio=1.500" in printed_output
