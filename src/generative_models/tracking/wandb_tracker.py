"""Experiment tracking utilities backed by Weights & Biases."""

import os

import torch
import wandb
from dotenv import load_dotenv
from torch import nn
from tqdm.auto import tqdm


class WandbExperimentTracker:
    """Experiment tracker wrapper around Weights & Biases (wandb).

    This class centralizes login, run initialization, metric logging,
    model registration and run finalization, following the same
    tracking pattern used in the training loop of the base notebook:
    an anonymous or authenticated login attempt, a `wandb.init` call
    with a reproducible `config` dictionary, per-epoch metric logging
    through the active run object (also echoed to the console, so
    progress stays visible even when tracking falls back to disabled
    mode), a fully described model artifact logged at the end of
    training, and an explicit `finish` call.

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
        alongside the run for reproducibility. This same dictionary is
        merged into the metadata of every model artifact logged with
        `log_model`.
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
    ...     description="Feed-forward regressor for log-return prediction.",
    ...     metadata={"final_train_loss": training_loss_history[-1]},
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
        tqdm.write(
            f"[wandb] Run started in '{self.wandb_mode}' mode "
            f"(project='{self.project_name}')."
        )

    def log_metrics(
        self,
        metrics: dict[str, object],
    ) -> None:
        """Log a dictionary of metrics to the current run and the console.

        Metrics are logged through the active run object returned by
        `start_run` (`self.wandb_run.log`), instead of the module-level
        `wandb.log`. This avoids a common notebook pitfall: if a
        training cell is re-run without calling `finish_run` first,
        wandb's global run state can end up out of sync with the run
        actually shown on the dashboard, so metrics logged through the
        module-level API silently land on the wrong run (or nowhere),
        leaving the run you are looking at empty. Metrics are also
        printed to the console through `tqdm.write`, so training
        progress stays visible even when tracking falls back to
        disabled mode.

        Parameters
        ----------
        metrics : dict[str, object]
            Mapping from metric name to its value at the current step
            (for example, `{"epoch": 1, "train/loss_mse": 0.0123}`).
        """
        formatted_metrics = " | ".join(
            f"{metric_name}: {_format_metric_value(metric_value)}"
            for metric_name, metric_value in metrics.items()
        )
        tqdm.write(formatted_metrics)

        if self.wandb_run is None:
            print("[wandb] No active run; skipping metric logging.")
            return

        self.wandb_run.log(metrics)

    def log_model(
        self,
        model: nn.Module,
        model_name: str,
        model_file_path: str,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Save the model weights and register a fully described artifact.

        The model's `state_dict` is saved locally at `model_file_path`
        and then logged as a wandb `Artifact` of type `"model"`,
        attached to the current run. The artifact metadata always
        includes the experiment `config` given to the tracker (so
        hyperparameters are never left out), merged with any extra
        `metadata` passed here (for example, the final training loss).
        This allows the trained weights to be versioned, described and
        downloaded later from the wandb dashboard, independently of
        the notebook that produced them.

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
        description : str | None
            Optional human-readable description of the model, stored
            alongside the artifact. Defaults to `None`.
        metadata : dict[str, object] | None
            Optional dictionary with extra information to attach to
            the artifact on top of the experiment `config` (for
            example, the final training loss or dataset statistics).
            Defaults to `None`.

        Example
        -------
        >>> tracker.log_model(
        ...     model=regression_model,
        ...     model_name="feed-forward-regression-model",
        ...     model_file_path="regression_model.pt",
        ...     description="Feed-forward regressor for log-return.",
        ...     metadata={
        ...         "final_train_loss": training_loss_history[-1],
        ...         "return_train_mean": return_train_mean,
        ...         "return_train_std": return_train_std,
        ...     },
        ... )
        """
        if self.wandb_run is None:
            print("[wandb] No active run; skipping model registration.")
            return

        torch.save(model.state_dict(), model_file_path)

        artifact_metadata = {**self.config, **(metadata or {})}
        model_artifact = wandb.Artifact(
            name=model_name,
            type="model",
            description=description,
            metadata=artifact_metadata,
        )
        model_artifact.add_file(model_file_path)

        self.wandb_run.log_artifact(model_artifact)
        tqdm.write(
            f"[wandb] Logged model artifact '{model_name}' "
            f"from '{model_file_path}'."
        )

    def finish_run(self) -> None:
        """Finish the current wandb run, if one is active."""
        if self.wandb_run is not None:
            self.wandb_run.finish()


def _format_metric_value(metric_value: object) -> str:
    """Format a metric value for console display.

    Parameters
    ----------
    metric_value : object
        Value to format. Floats are rendered with six decimal places;
        any other type is converted with `str`.

    Returns
    -------
    str
        Human-readable representation of `metric_value`.
    """
    if isinstance(metric_value, float):
        return f"{metric_value:.6f}"
    return str(metric_value)
