"""Training options shared by the models of the project."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingOptions:  # pylint: disable=too-many-instance-attributes
    """Options that every model accepts when it is trained for tickers.

    Parameters
    ----------
    start_date : str
        First date of the historical range, in `"YYYY-MM-DD"` format.
    end_date : str
        Last date of the historical range, in `"YYYY-MM-DD"` format.
    epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate of the optimizer.
    window_size : int
        Number of log-returns in each window.
    output_directory : str
        Directory where the weights and the reports are saved.
    device : str
        Device for training, for example `"cpu"` or `"cuda"`.
    experiment_name : str
        Name of the experiment the models are registered under. See
        `TrackingConfig.experiment_name` for the allowed format.

    Example
    -------
    >>> options = TrainingOptions(epochs=50, learning_rate=0.01)
    >>> options.window_size
    10
    """

    start_date: str = "2020-01-01"
    end_date: str = "2026-09-15"
    epochs: int = 300
    learning_rate: float = 0.001
    window_size: int = 10
    output_directory: str = "artifacts"
    device: str = "cpu"
    experiment_name: str = "default"
