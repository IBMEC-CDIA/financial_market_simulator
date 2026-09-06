"""Experiment tracking utilities backed by Weights & Biases."""

import os

import torch
import wandb
from dotenv import load_dotenv
from torch import nn


class WandbExperimentTracker:
    """Experiment tracker wrapper around Weights & Biases (wandb).

    This class centralizes login, run initialization, metric logging,
    model registration and run finalization, following the same
    tracking pattern used in the training loop of the base notebook:
    an anonymous or authenticated login attempt, a `wandb.init` call
    with a reproducible `config` dictionary, per-epoch `wandb.log`
    calls, optional model artifact logging, and an explicit `finish`
    call at the end of training.

    The wandb API key is read from an environment variable using
    `python-dotenv`, so it can be provided locally through a `.env`
    file instead of being hardcoded in the notebook.

    Parameters
    ----------
    project_name : str
        Name of the wandb project where the run will be logged.
    config : dict[str, object]
        Dictionary with the experiment configuration (hyperparameters,
        dataset settings, random seed, etc.) that will be stored
        alongside the run for reproducibility.
    env_file_path : str
        Path to the `.env` file containing the `WANDB_API_KEY`
        variable. Defaults to `.env` in the current directory.
    login_timeout_seconds : int
        Maximum time, in seconds, allowed for the `wandb.login` call
        before falling back to disabled tracking.

    Example
    -------
    >>> tracker = WandbExperimentTracker(
    ...     project_name="pytorch-tensores-log-return",
    ...     config={
    ...         "ticker": TICKER_SYMBOL,
    ...         "random_seed": RANDOM_SEED,
    ...         "epochs": number_of_epochs,
    ...         "learning_rate": learning_rate,
    ...     },
    ... )
    >>> tracker.start_run()
    >>> tracker.log_metrics({"epoch": 1, "train/loss_mse": 0.0123})
    >>> tracker.log_model(
    ...     model=regression_model,
    ...     model_name="feed-forward-regression-model",
    ...     model_file_path="regression_model.pt",
    ... )
    >>> tracker.finish_run()
    """

    def __init__(
        self,
        project_name: str,
        config: dict[str, object],
        env_file_path: str = ".env",
        login_timeout_seconds: int = 10,
    ) -> None:
        """Load the wandb API key from the .env file and store settings."""
        load_dotenv(dotenv_path=env_file_path)

        self.project_name = project_name
        self.config = config
        self.login_timeout_seconds = login_timeout_seconds
        self.wandb_api_key = os.getenv("WANDB_API_KEY")
        self.wandb_mode = "disabled"
        self.wandb_run = None

    def _login(self) -> None:
        """Attempt to log in to wandb, falling back to disabled mode.

        If a `WANDB_API_KEY` was found in the environment, an
        authenticated login is attempted. Otherwise, an anonymous
        login is attempted. If either attempt fails, tracking is
        disabled and training can proceed without an active run.
        """
        try:
            if self.wandb_api_key:
                wandb.login(
                    key=self.wandb_api_key,
                    timeout=self.login_timeout_seconds,
                )
            else:
                wandb.login(
                    anonymous="allow",
                    timeout=self.login_timeout_seconds,
                )
            self.wandb_mode = "online"
        # pylint: disable-next=broad-exception-caught
        except Exception as login_error:
            print(
                f"[wandb] Could not log in ({login_error}); "
                "falling back to disabled tracking."
            )
            self.wandb_mode = "disabled"

    def start_run(self) -> None:
        """Log in to wandb and initialize a new run with the stored config."""
        self._login()

        self.wandb_run = wandb.init(
            project=self.project_name,
            mode=self.wandb_mode,
            config=self.config,
        )

    def log_metrics(
        self,
        metrics: dict[str, object],
    ) -> None:
        """Log a dictionary of metrics to the current run.

        Parameters
        ----------
        metrics : dict[str, object]
            Mapping from metric name to its value at the current step
            (for example, `{"epoch": 1, "train/loss_mse": 0.0123}`).
        """
        wandb.log(metrics)

    def log_model(
        self,
        model: nn.Module,
        model_name: str,
        model_file_path: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Save the model weights to disk and register as a wandb artifact.

        The model's `state_dict` is saved locally at `model_file_path`
        and then logged as a wandb `Artifact` of type `"model"`,
        attached to the current run. This allows the trained weights
        to be versioned and downloaded later from the wandb dashboard,
        independently of the notebook that produced them.

        Parameters
        ----------
        model : nn.Module
            Trained PyTorch model whose weights will be saved and
            registered.
        model_name : str
            Name used to identify the artifact in wandb (for example,
            `"feed-forward-regression-model"`).
        model_file_path : str
            Local path where the model's `state_dict` will be saved
            (for example, `"regression_model.pt"`).
        metadata : dict[str, object] | None
            Optional dictionary with extra information to attach to
            the artifact (for example, architecture hyperparameters or
            final validation loss). Defaults to `None`.

        Example
        -------
        >>> tracker.log_model(
        ...     model=regression_model,
        ...     model_name="feed-forward-regression-model",
        ...     model_file_path="regression_model.pt",
        ...     metadata={"hidden_layer_size": hidden_layer_size},
        ... )
        """
        if self.wandb_run is None:
            print("[wandb] No active run; skipping model registration.")
            return

        torch.save(model.state_dict(), model_file_path)

        model_artifact = wandb.Artifact(
            name=model_name,
            type="model",
            metadata=metadata,
        )
        model_artifact.add_file(model_file_path)

        self.wandb_run.log_artifact(model_artifact)

    def finish_run(self) -> None:
        """Finish the current wandb run, if one is active."""
        if self.wandb_run is not None:
            self.wandb_run.finish()
