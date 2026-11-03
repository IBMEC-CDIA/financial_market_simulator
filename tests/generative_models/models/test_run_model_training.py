"""Unit tests for the command that trains the model of several tickers."""

# pylint: disable=duplicate-code

import json
from unittest.mock import MagicMock, patch

import pytest

from generative_models.models.run_model_training import (
    build_argument_parser,
    build_tickers_configuration,
    load_model_module,
    main,
    parse_ticker_list,
    validate_model_name,
)
from generative_models.models.training_options import TrainingOptions

MANAGER_PATH = "generative_models.models.run_model_training.TickerModelManager"


def _make_result(model_file_path) -> MagicMock:
    """Build a fake training result with real numbers for the reports."""
    result = MagicMock()
    result.training_loss_history = [0.5, 0.4]
    result.train_evaluation_loss = 0.4
    result.validation_loss = 0.6
    result.anomaly_threshold = 1.2
    result.window_size = 10
    result.hidden_layer_size = 16
    result.latent_layer_size = 4
    result.return_train_mean = 0.001
    result.return_train_std = 0.02
    result.test_loss = 0.3
    result.baseline_test_loss = 0.9
    result.model_file_path = model_file_path

    return result


def test_parse_ticker_list_trims_and_drops_empty_items() -> None:
    """Check that spaces and empty items are removed from the list."""
    assert parse_ticker_list(" AAPL, MSFT ,,PETR4.SA,") == [
        "AAPL",
        "MSFT",
        "PETR4.SA",
    ]


def test_tickers_are_required() -> None:
    """Check that argparse rejects a call without --tickers."""
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args([])


def test_default_model_is_the_base_autoencoder() -> None:
    """Check that the base autoencoder is used when no model is given."""
    arguments = build_argument_parser().parse_args(["--tickers", "AAPL"])

    assert arguments.model == "base_autoencoder"


@pytest.mark.parametrize("model_name", ["../models", "Base", "a-b", ""])
def test_invalid_model_names_are_rejected(model_name: str) -> None:
    """Check that names with path or upper-case characters are refused."""
    with pytest.raises(ValueError, match="Invalid model name"):
        validate_model_name(model_name)


def test_load_model_module_imports_the_model_package() -> None:
    """Check that the trainer factory of a model package is found."""
    module = load_model_module("base_autoencoder", "trainer_factory")

    assert hasattr(module, "build_trainer")


def test_tickers_configuration_gives_every_ticker_the_same_dates() -> None:
    """Check that each ticker receives the start and end dates."""
    configuration = build_tickers_configuration(
        ["AAPL", "MSFT"],
        "2020-01-01",
        "2026-01-01",
    )

    assert configuration == {
        "AAPL": {"start_date": "2020-01-01", "end_date": "2026-01-01"},
        "MSFT": {"start_date": "2020-01-01", "end_date": "2026-01-01"},
    }


@patch(MANAGER_PATH)
def test_main_trains_every_ticker_and_writes_its_report(
    mock_manager_class,
    tmp_path,
) -> None:
    """Check that the manager trains all tickers and each one gets a report."""
    manager = mock_manager_class.return_value
    manager.get_model.side_effect = lambda ticker: _make_result(
        tmp_path / f"simple-autoencoder-{ticker.lower()}.pt"
    )

    results = main(
        [
            "--tickers",
            "AAPL,MSFT",
            "--output-directory",
            str(tmp_path),
        ]
    )

    manager.train_all_models.assert_called_once_with()
    assert list(results) == ["AAPL", "MSFT"]
    for ticker in ["aapl", "msft"]:
        report = json.loads(
            (tmp_path / f"simple-autoencoder-{ticker}-overfitting.json")
            .read_text(encoding="utf-8")
        )
        assert report["ticker"].lower() == ticker


@patch(MANAGER_PATH)
def test_main_passes_the_options_to_the_trainer_factory(
    mock_manager_class,
    tmp_path,
) -> None:
    """Check that the trainer factory receives the shared options."""
    mock_manager_class.return_value.get_model.return_value = _make_result(
        tmp_path / "simple-autoencoder-aapl.pt"
    )

    main(
        [
            "--tickers",
            "AAPL",
            "--epochs",
            "7",
            "--output-directory",
            str(tmp_path),
        ]
    )

    trainer_factory = mock_manager_class.call_args.kwargs["trainer_factory"]
    assert trainer_factory.keywords["options"] == TrainingOptions(
        epochs=7,
        output_directory=str(tmp_path),
    )
    tickers_configuration = mock_manager_class.call_args.kwargs["tickers"]
    assert tickers_configuration["AAPL"]["start_date"] == "2020-01-01"


def test_main_rejects_an_invalid_model_before_training(tmp_path) -> None:
    """Check that an invalid model name stops the run before any training."""
    with patch(MANAGER_PATH) as mock_manager_class:
        with pytest.raises(ValueError, match="Invalid model name"):
            main(
                [
                    "--model",
                    "../models",
                    "--tickers",
                    "AAPL",
                    "--output-directory",
                    str(tmp_path),
                ]
            )

    mock_manager_class.assert_not_called()
