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
class TrainingConfig:
    """Hyperparameters of one training run.

    Parameters
    ----------
    window_size : int
        Number of past log-returns given as input.
    hidden_layer_size : int
        Units in each hidden layer of the model.
    epochs : int
        Maximum number of passes over the training windows.
    learning_rate : float
        Learning rate of the Adam optimizer.
    seed : int
        Seed of the random generators of the run.
    """

    window_size: int = DEFAULT_WINDOW_SIZE
    hidden_layer_size: int = DEFAULT_HIDDEN_LAYER_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class WindowedSplit:
    """Normalized training and validation windows of one series.

    Parameters
    ----------
    train_inputs : np.ndarray
        Input windows of the training part, shape `(n, window_size)`.
    train_targets : np.ndarray
        Next values of the training windows, shape `(n, 1)`.
    validation_inputs : np.ndarray
        Input windows of the validation part.
    validation_targets : np.ndarray
        Next values of the validation windows.
    return_train_mean : float
        Mean of the training log-returns, used to normalize.
    return_train_std : float
        Standard deviation of the training log-returns.
    """

    train_inputs: np.ndarray
    train_targets: np.ndarray
    validation_inputs: np.ndarray
    validation_targets: np.ndarray
    return_train_mean: float
    return_train_std: float


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of training one regression forecaster.

    Parameters
    ----------
    model : nn.Module
        Trained model in evaluation mode, with the weights of the epoch that
        had the lowest validation loss (early stopping).
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
        [
            normalized_returns[start : start + window_size]
            for start in range(count)
        ]
    )
    targets = normalized_returns[window_size:].reshape(-1, 1)
    return inputs.astype(np.float32), targets.astype(np.float32)


def split_and_normalize(
    closes: pd.Series,
    window_size: int,
) -> WindowedSplit:
    """Build the normalized training and validation windows of a series.

    The first 80% of the returns train the model and the rest validates it.
    Both parts are normalized with the mean and the standard deviation of the
    training part only, so no information of the validation part leaks into
    the training.

    Parameters
    ----------
    closes : pd.Series
        Closing prices in chronological order.
    window_size : int
        Number of past log-returns in each input window.

    Returns
    -------
    WindowedSplit
        Windows of both parts and the normalization statistics.

    Raises
    ------
    ValueError
        When the training returns have no variation, or the series is too
        short to build windows for both parts.

    Example
    -------
    >>> split = split_and_normalize(closes, 30)  # doctest: +SKIP
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
    return WindowedSplit(
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        return_train_mean=mean,
        return_train_std=std,
    )


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    generator: torch.Generator,
) -> float:
    """Run one pass over the training windows in shuffled mini-batches.

    Parameters
    ----------
    model : nn.Module
        Model being trained.
    optimizer : torch.optim.Optimizer
        Optimizer that updates the weights of the model.
    inputs : torch.Tensor
        Input windows of the training part.
    targets : torch.Tensor
        Next values of the training windows.
    generator : torch.Generator
        Random generator that shuffles the windows.

    Returns
    -------
    float
        Mean squared error over the training windows of this epoch.

    Example
    -------
    >>> train_one_epoch(model, optimizer, x, y, generator)  # doctest: +SKIP
    """
    loss_function = nn.MSELoss()
    model.train()
    order = torch.randperm(len(inputs), generator=generator)
    total_loss = 0.0
    for start in range(0, len(order), DEFAULT_BATCH_SIZE):
        batch = order[start : start + DEFAULT_BATCH_SIZE]
        optimizer.zero_grad()
        loss = loss_function(model(inputs[batch]), targets[batch])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
    return total_loss / len(order)


def evaluate_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Mean squared error of the model on a set of windows, without training.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    inputs : torch.Tensor
        Input windows.
    targets : torch.Tensor
        Expected next values.

    Returns
    -------
    float
        Mean squared error.

    Example
    -------
    >>> evaluate_loss(model, x_val, y_val)  # doctest: +SKIP
    0.9
    """
    loss_function = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        return loss_function(model(inputs), targets).item()


def fit_model(
    model: nn.Module,
    split: WindowedSplit,
    config: TrainingConfig,
) -> float:
    """Train a model with early stopping on the validation windows.

    After the last epoch, the weights of the epoch with the lowest validation
    loss are restored into the model.

    Parameters
    ----------
    model : nn.Module
        Model to train, updated in place.
    split : WindowedSplit
        Normalized training and validation windows.
    config : TrainingConfig
        Epochs, learning rate and seed of the run.

    Returns
    -------
    float
        Mean squared error on the training windows after the last epoch.

    Example
    -------
    >>> fit_model(model, split, TrainingConfig(epochs=5))  # doctest: +SKIP
    0.02
    """
    torch.manual_seed(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    train_x = torch.from_numpy(split.train_inputs)
    train_y = torch.from_numpy(split.train_targets)
    validation_x = torch.from_numpy(split.validation_inputs)
    validation_y = torch.from_numpy(split.validation_targets)
    best_loss = float("inf")
    best_state = None
    final_train_loss = float("nan")
    for _epoch in range(config.epochs):
        final_train_loss = train_one_epoch(
            model,
            optimizer,
            train_x,
            train_y,
            generator,
        )
        epoch_validation_loss = evaluate_loss(
            model,
            validation_x,
            validation_y,
        )
        if epoch_validation_loss < best_loss:
            best_loss = epoch_validation_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return final_train_loss


def train_regression_forecaster(
    closes: pd.Series,
    config: TrainingConfig = TrainingConfig(),
) -> TrainingResult:
    """Train the regression model on the log-returns of a price series.

    Parameters
    ----------
    closes : pd.Series
        Closing prices in chronological order.
    config : TrainingConfig
        Window size, model size, epochs, learning rate and seed.

    Returns
    -------
    TrainingResult
        Trained model, normalization statistics and the losses.

    Raises
    ------
    ValueError
        When the series cannot be split into training and validation windows.

    Example
    -------
    >>> config = TrainingConfig(epochs=5)  # doctest: +SKIP
    >>> train_regression_forecaster(closes, config)  # doctest: +SKIP
    """
    split = split_and_normalize(closes, config.window_size)
    model = FeedForwardRegressionModel(
        input_window_size=config.window_size,
        hidden_layer_size=config.hidden_layer_size,
    )
    final_train_loss = fit_model(model, split, config)
    validation_loss = evaluate_loss(
        model,
        torch.from_numpy(split.validation_inputs),
        torch.from_numpy(split.validation_targets),
    )
    baseline_validation_loss = float(
        np.mean(
            (split.validation_targets - split.train_targets.mean()) ** 2,
        )
    )
    return TrainingResult(
        model=model,
        return_train_mean=split.return_train_mean,
        return_train_std=split.return_train_std,
        final_train_loss=final_train_loss,
        validation_loss=validation_loss,
        baseline_validation_loss=baseline_validation_loss,
    )


def save_regression_artifact(
    result: TrainingResult,
    ticker: str,
    output_directory: Path,
    config: TrainingConfig,
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
    config : TrainingConfig
        Configuration used in training, with the window and the model size.

    Returns
    -------
    Path
        Path of the written artifact.

    Example
    -------
    >>> save_regression_artifact(result, "AAPL", models, config)  # doctest: +SKIP
    PosixPath('models/regression/AAPL.pt')
    """
    path = regression_artifact_path(ticker, output_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": result.model.state_dict(),
        "input_window_size": config.window_size,
        "hidden_layer_size": config.hidden_layer_size,
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
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
    )
    parser.add_argument(
        "--hidden-layer-size",
        type=int,
        default=DEFAULT_HIDDEN_LAYER_SIZE,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("models"),
    )
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
    prices = fetch_price_series(
        ticker,
        options.start_date,
        options.end_date,
    )
    config = TrainingConfig(
        window_size=options.window_size,
        hidden_layer_size=options.hidden_layer_size,
        epochs=options.epochs,
        learning_rate=options.learning_rate,
        seed=options.seed,
    )
    tracker = WandbExperimentTracker(
        project_name=options.project_name,
        config={
            "ticker": ticker,
            "start_date": options.start_date,
            "end_date": options.end_date,
            "window_size": config.window_size,
            "hidden_layer_size": config.hidden_layer_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "batch_size": DEFAULT_BATCH_SIZE,
            "random_seed": config.seed,
            "optimizer": "Adam",
            "loss_function": "MSE",
            "target": "log_return",
        },
    )
    tracker.start_run()
    result = train_regression_forecaster(prices["close_price"], config)
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
        config,
    )
    tracker.finish_run()
    print(
        f"{ticker}: validation loss {result.validation_loss:.6f} "
        f"(baseline {result.baseline_validation_loss:.6f}). "
        f"Saved to {path}."
    )
    return path


if __name__ == "__main__":
    main()
