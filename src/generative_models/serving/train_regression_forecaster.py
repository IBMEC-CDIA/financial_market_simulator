"""Train the regression forecaster of a ticker and save its artifact.

The artifact is written to `<output_directory>/regression/<TICKER>.pt`, the
path that the `GET /forecast` route reads. It follows the notebook
`notebooks/regression_log_return_training.ipynb`: a window of normalized
log-returns is the input and the next normalized log-return is the target.
"""

import argparse
import copy
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from generative_models.data.market_data import fetch_price_series
from generative_models.models.regression.feed_forward_regression_model import (
    FeedForwardRegressionModel,
)
from generative_models.serving.forecast import (
    log_returns_of,
    regression_artifact_path,
)
from generative_models.tracking.wandb_tracker import WandbExperimentTracker

TRAIN_FRACTION = 0.8
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_WINDOW_SIZE = 30
DEFAULT_HIDDEN_LAYER_SIZE = 64
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42
DEFAULT_PROJECT_NAME = "regression-forecaster"


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of training one regression forecaster.

    Parameters
    ----------
    model : nn.Module
        Trained model in evaluation mode, with the weights of the epoch that had
        the lowest validation loss (early stopping).
    return_train_mean : float
        Mean of the training log-returns, used to normalize.
    return_train_std : float
        Standard deviation of the training log-returns.
    final_train_loss : float
        Mean squared error on the training windows after the last epoch.
    validation_loss : float
        Mean squared error on the validation windows.
    baseline_validation_loss : float
        Mean squared error of predicting the training mean on the validation
        windows. A useful model must have a lower validation loss.
    """

    model: nn.Module
    return_train_mean: float
    return_train_std: float
    final_train_loss: float
    validation_loss: float
    baseline_validation_loss: float


def build_windows(
    normalized_returns: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a series into input windows and the value that follows each one.

    Parameters
    ----------
    normalized_returns : np.ndarray
        Normalized log-returns in chronological order.
    window_size : int
        Number of past values in each input window.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Inputs of shape `(n, window_size)` and targets of shape `(n, 1)`.
        Both are empty when the series is shorter than `window_size + 1`.

    Example
    -------
    >>> inputs, targets = build_windows(np.arange(5.0), 2)
    >>> targets.ravel().tolist()
    [2.0, 3.0, 4.0]
    """
    count = len(normalized_returns) - window_size
    if count <= 0:
        return np.empty((0, window_size)), np.empty((0, 1))
    inputs = np.stack(
        [normalized_returns[start : start + window_size] for start in range(count)]
    )
    targets = normalized_returns[window_size:].reshape(-1, 1)
    return inputs.astype(np.float32), targets.astype(np.float32)


def train_regression_forecaster(
    closes: pd.Series,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hidden_layer_size: int = DEFAULT_HIDDEN_LAYER_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = DEFAULT_SEED,
) -> TrainingResult:
    """Train the regression model on the log-returns of a price series.

    The first 80% of the returns train the model and the rest validates it.
    Both parts are normalized with the mean and the standard deviation of the
    training part only, so no information of the validation part leaks into
    the training.

    Parameters
    ----------
    closes : pd.Series
        Closing prices in chronological order.
    window_size : int
        Number of past log-returns given as input.
    hidden_layer_size : int
        Units in each hidden layer of the model.
    epochs : int
        Number of passes over the training windows.
    learning_rate : float
        Learning rate of the Adam optimizer.
    seed : int
        Seed of the random generator used to shuffle the windows.

    Returns
    -------
    TrainingResult
        Trained model, normalization statistics and the losses.

    Raises
    ------
    ValueError
        When the series is too short to build training and validation
        windows.

    Example
    -------
    >>> result = train_regression_forecaster(closes, epochs=5)  # doctest: +SKIP
    """
    returns = log_returns_of(closes.to_numpy(dtype=float))
    split_index = int(len(returns) * TRAIN_FRACTION)
    train_returns = returns[:split_index]
    validation_returns = returns[split_index:]
    mean = float(train_returns.mean())
    std = float(train_returns.std())
    if std <= 0:
        raise ValueError("The training returns have no variation.")
    train_inputs, train_targets = build_windows(
        (train_returns - mean) / std,
        window_size,
    )
    validation_inputs, validation_targets = build_windows(
        (validation_returns - mean) / std,
        window_size,
    )
    if len(train_inputs) == 0 or len(validation_inputs) == 0:
        raise ValueError("The price series is too short to train the model.")

    torch.manual_seed(seed)
    model = FeedForwardRegressionModel(
        input_window_size=window_size,
        hidden_layer_size=hidden_layer_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    train_x = torch.from_numpy(train_inputs)
    train_y = torch.from_numpy(train_targets)
    generator = torch.Generator().manual_seed(seed)
    validation_x = torch.from_numpy(validation_inputs)
    validation_y = torch.from_numpy(validation_targets)
    final_train_loss = float("nan")
    best_validation_loss = float("inf")
    best_state = None
    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(order), DEFAULT_BATCH_SIZE):
            batch = order[start : start + DEFAULT_BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_function(model(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        final_train_loss = epoch_loss / len(order)
        model.eval()
        with torch.no_grad():
            epoch_validation_loss = loss_function(
                model(validation_x), validation_y
            ).item()
        if epoch_validation_loss < best_validation_loss:
            best_validation_loss = epoch_validation_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_loss = loss_function(model(validation_x), validation_y).item()
    baseline_validation_loss = float(
        np.mean((validation_targets - train_targets.mean()) ** 2)
    )
    return TrainingResult(
        model=model,
        return_train_mean=mean,
        return_train_std=std,
        final_train_loss=final_train_loss,
        validation_loss=float(validation_loss),
        baseline_validation_loss=baseline_validation_loss,
    )


def save_regression_artifact(
    result: TrainingResult,
    ticker: str,
    output_directory: Path,
    window_size: int,
    hidden_layer_size: int,
) -> Path:
    """Save a trained regression forecaster in the format the route reads.

    Parameters
    ----------
    result : TrainingResult
        Output of `train_regression_forecaster`.
    ticker : str
        Ticker the model was trained on.
    output_directory : Path
        Models directory; the artifact goes to its `regression` folder.
    window_size : int
        Input window size used in training.
    hidden_layer_size : int
        Hidden layer size used in training.

    Returns
    -------
    Path
        Path of the written artifact.

    Example
    -------
    >>> save_regression_artifact(result, "AAPL", Path("models"), 30, 64)  # doctest: +SKIP
    PosixPath('models/regression/AAPL.pt')
    """
    path = regression_artifact_path(ticker, output_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": result.model.state_dict(),
        "input_window_size": window_size,
        "hidden_layer_size": hidden_layer_size,
        "return_train_mean": result.return_train_mean,
        "return_train_std": result.return_train_std,
        "artifact_name": f"regression-{ticker.upper()}",
    }
    torch.save(payload, path)
    return path


def parse_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    """Read the command-line options of the training script.

    Parameters
    ----------
    arguments : list[str] | None
        Arguments to parse. Defaults to the process arguments.

    Returns
    -------
    argparse.Namespace
        Parsed options.

    Example
    -------
    >>> parse_arguments(["--ticker", "AAPL"]).ticker
    'AAPL'
    """
    parser = argparse.ArgumentParser(
        description="Train and save the regression forecaster of a ticker.",
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument(
        "--hidden-layer-size", type=int, default=DEFAULT_HIDDEN_LAYER_SIZE
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-directory", type=Path, default=Path("models"))
    parser.add_argument("--project-name", default=DEFAULT_PROJECT_NAME)
    return parser.parse_args(arguments)


def main(
    arguments: list[str] | None = None,
) -> Path:
    """Download the prices, train the forecaster, track it and save it.

    Parameters
    ----------
    arguments : list[str] | None
        Command-line arguments. Defaults to the process arguments.

    Returns
    -------
    Path
        Path of the saved artifact.

    Example
    -------
    >>> main(["--ticker", "AAPL", "--epochs", "5"])  # doctest: +SKIP
    """
    options = parse_arguments(arguments)
    ticker = options.ticker.upper()
    prices = fetch_price_series(ticker, options.start_date, options.end_date)
    config = {
        "ticker": ticker,
        "start_date": options.start_date,
        "end_date": options.end_date,
        "window_size": options.window_size,
        "hidden_layer_size": options.hidden_layer_size,
        "epochs": options.epochs,
        "learning_rate": options.learning_rate,
        "batch_size": DEFAULT_BATCH_SIZE,
        "random_seed": options.seed,
        "optimizer": "Adam",
        "loss_function": "MSE",
        "target": "log_return",
    }
    tracker = WandbExperimentTracker(
        project_name=options.project_name,
        config=config,
    )
    tracker.start_run()
    result = train_regression_forecaster(
        prices["close_price"],
        window_size=options.window_size,
        hidden_layer_size=options.hidden_layer_size,
        epochs=options.epochs,
        learning_rate=options.learning_rate,
        seed=options.seed,
    )
    tracker.log_metrics(
        {
            "final_train_loss": result.final_train_loss,
            "validation_loss": result.validation_loss,
            "baseline_validation_loss": result.baseline_validation_loss,
        }
    )
    path = save_regression_artifact(
        result,
        ticker,
        options.output_directory,
        options.window_size,
        options.hidden_layer_size,
    )
    tracker.finish_run()
    print(
        f"{ticker}: validation loss {result.validation_loss:.6f} "
        f"(baseline {result.baseline_validation_loss:.6f}). Saved to {path}."
    )
    return path


if __name__ == "__main__":
    main()
