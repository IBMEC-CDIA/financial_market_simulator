"""Overfitting check that compares the training and validation errors."""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from generative_models.models.autoencoder.trainer import (
    AutoencoderTrainingResult,
)

DEFAULT_MAX_OVERFITTING_RATIO = 1.5
REPORT_SUFFIX = "-overfitting.json"


@dataclass
# pylint: disable-next=too-many-instance-attributes
class OverfittingReport:
    """Errors of a trained model on the training and validation sets.

    Parameters
    ----------
    ticker : str
        Ticker symbol the model was trained on.
    model_file_path : str
        Local file where the trained weights were saved.
    final_train_loss : float
        Training loss of the last epoch, averaged during training.
    train_evaluation_loss : float
        Mean squared error on the training set after training.
    validation_loss : float
        Mean squared error on the validation set after training.
    overfitting_ratio : float
        `validation_loss / train_evaluation_loss`. A value well above 1
        means the model reconstructs the training data much better than
        data it has not seen.
    test_loss : float | None
        Mean squared error on the held-out test set. `None` when the run
        had no test split.
    baseline_test_loss : float | None
        Mean squared error of the naive zero prediction on the test set.
    """

    ticker: str
    model_file_path: str
    final_train_loss: float
    train_evaluation_loss: float
    validation_loss: float
    overfitting_ratio: float
    test_loss: float | None = None
    baseline_test_loss: float | None = None


def build_overfitting_report(
    ticker: str,
    result: AutoencoderTrainingResult,
) -> OverfittingReport:
    """Build the overfitting report from the result of a training run.

    Parameters
    ----------
    ticker : str
        Ticker symbol the model was trained on.
    result : AutoencoderTrainingResult
        Result returned by `AutoencoderTrainer.train`.

    Returns
    -------
    OverfittingReport
        Report with the errors and the validation to training ratio.

    Raises
    ------
    ValueError
        If the training error is not positive, so the ratio is undefined.

    Example
    -------
    >>> report = build_overfitting_report("AAPL", result)
    >>> report.overfitting_ratio
    1.2
    """
    train_evaluation_loss = result.train_evaluation_loss
    if train_evaluation_loss <= 0:
        raise ValueError(
            "The training loss must be positive to compute the ratio."
        )

    return OverfittingReport(
        ticker=ticker,
        model_file_path=str(result.model_file_path),
        final_train_loss=result.training_loss_history[-1],
        train_evaluation_loss=train_evaluation_loss,
        validation_loss=result.validation_loss,
        overfitting_ratio=(
            result.validation_loss / train_evaluation_loss
        ),
        test_loss=result.test_loss,
        baseline_test_loss=result.baseline_test_loss,
    )


def build_report_path(model_file_path: str | Path) -> Path:
    """Return the path of the report saved next to a model file.

    Parameters
    ----------
    model_file_path : str | Path
        Path of the trained weights, for example
        `artifacts/simple-autoencoder-aapl.pt`.

    Returns
    -------
    Path
        Path in the form
        `artifacts/simple-autoencoder-aapl-overfitting.json`.
    """
    model_path = Path(model_file_path)

    return model_path.with_name(model_path.stem + REPORT_SUFFIX)


def save_overfitting_report(
    report: OverfittingReport,
    output_path: str | Path,
) -> Path:
    """Write the report as a JSON file, creating its directory if needed.

    Parameters
    ----------
    report : OverfittingReport
        Report to write.
    output_path : str | Path
        Path of the JSON file.

    Returns
    -------
    Path
        Path of the written file.
    """
    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(asdict(report), indent=2) + "\n",
        encoding="utf-8",
    )

    return file_path


def load_overfitting_report(
    input_path: str | Path,
) -> OverfittingReport:
    """Read a report written by `save_overfitting_report`.

    Parameters
    ----------
    input_path : str | Path
        Path of the JSON file.

    Returns
    -------
    OverfittingReport
        Report stored in the file.
    """
    report_data = json.loads(Path(input_path).read_text(encoding="utf-8"))

    return OverfittingReport(**report_data)


def is_overfitting_within_limit(
    report: OverfittingReport,
    max_overfitting_ratio: float,
) -> bool:
    """Tell whether the validation to training ratio is acceptable.

    Parameters
    ----------
    report : OverfittingReport
        Report of the training run.
    max_overfitting_ratio : float
        Largest accepted `overfitting_ratio`, inclusive.

    Returns
    -------
    bool
        `True` when the ratio is at most `max_overfitting_ratio`.

    Example
    -------
    >>> is_overfitting_within_limit(report, max_overfitting_ratio=1.5)
    True
    """
    return report.overfitting_ratio <= max_overfitting_ratio


def is_better_than_baseline(report: OverfittingReport) -> bool:
    """Tell whether the model beats the naive baseline on the test set.

    Parameters
    ----------
    report : OverfittingReport
        Report of the training run.

    Returns
    -------
    bool
        `True` when the test error is smaller than the error of the naive
        zero prediction. `False` when either value is missing.

    Example
    -------
    >>> is_better_than_baseline(report)
    True
    """
    if report.test_loss is None or report.baseline_test_loss is None:
        return False

    return report.test_loss < report.baseline_test_loss


def format_optional_loss(loss: float | None) -> str:
    """Format a loss for the summary table, or `n/a` when it is missing.

    Parameters
    ----------
    loss : float | None
        Loss value, or `None`.

    Returns
    -------
    str
        The loss with six decimals, or `n/a`.
    """
    if loss is None:
        return "n/a"

    return f"{loss:.6f}"


def build_step_summary(
    report: OverfittingReport,
    max_overfitting_ratio: float,
    passed: bool,
) -> str:
    """Build the Markdown table shown on the summary page of a CI job.

    Parameters
    ----------
    report : OverfittingReport
        Report of the training run.
    max_overfitting_ratio : float
        Limit that the ratio was checked against.
    passed : bool
        Result of the check.

    Returns
    -------
    str
        Markdown with the result and the errors of the run.
    """
    status = "passed" if passed else "failed"

    return "\n".join(
        [
            "## Overfitting check",
            "",
            f"Result: **{status}**",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Ticker | {report.ticker} |",
            f"| Train error (MSE) | {report.train_evaluation_loss:.6f} |",
            f"| Validation error (MSE) | {report.validation_loss:.6f} |",
            f"| Validation / train ratio | {report.overfitting_ratio:.3f} |",
            f"| Limit | {max_overfitting_ratio:.3f} |",
            f"| Test error (MSE) | {format_optional_loss(report.test_loss)} |",
            "| Naive baseline error (MSE) | "
            f"{format_optional_loss(report.baseline_test_loss)} |",
            "",
        ]
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the parser of the overfitting evaluation script.

    Returns
    -------
    argparse.ArgumentParser
        Parser that accepts `--report-path` (required) and
        `--max-overfitting-ratio`.
    """
    parser = argparse.ArgumentParser(
        description="Check the overfitting report of a training run.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        required=True,
        help="JSON report written by the training script.",
    )
    parser.add_argument(
        "--max-overfitting-ratio",
        type=float,
        default=DEFAULT_MAX_OVERFITTING_RATIO,
        help="Largest accepted validation to training loss ratio.",
    )

    return parser


def main(
    command_line_arguments: list[str] | None = None,
) -> bool:
    """Print the overfitting report and tell whether it passes the check.

    Parameters
    ----------
    command_line_arguments : list[str] | None
        Arguments to parse. When `None`, the arguments of `sys.argv`
        are used.

    Returns
    -------
    bool
        `True` when the ratio is within the limit. A summary line is also
        printed.

    Example
    -------
    >>> main(["--report-path", "artifacts/model-overfitting.json"])
    True
    """
    parser = build_argument_parser()
    arguments = parser.parse_args(command_line_arguments)
    report = load_overfitting_report(arguments.report_path)
    is_within_limit = is_overfitting_within_limit(
        report,
        arguments.max_overfitting_ratio,
    )
    beats_baseline = is_better_than_baseline(report)
    passed = is_within_limit and beats_baseline

    print(
        f"ticker={report.ticker} "
        f"train_loss={report.train_evaluation_loss:.6f} "
        f"validation_loss={report.validation_loss:.6f} "
        f"test_loss={format_optional_loss(report.test_loss)} "
        "baseline_test_loss="
        f"{format_optional_loss(report.baseline_test_loss)} "
        f"overfitting_ratio={report.overfitting_ratio:.3f} "
        f"max_overfitting_ratio={arguments.max_overfitting_ratio:.3f} "
        f"beats_baseline={beats_baseline} "
        f"passed={passed}"
    )

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        summary_markdown = build_step_summary(
            report,
            arguments.max_overfitting_ratio,
            passed,
        )
        with Path(summary_path).open("a", encoding="utf-8") as summary_file:
            summary_file.write(summary_markdown)

    return passed


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
