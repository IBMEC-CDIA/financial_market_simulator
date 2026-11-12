"""Model registry that serves the autoencoders saved in a local directory."""

# pylint: disable=duplicate-code

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

from generative_models.models.base_autoencoder.overfitting import (
    load_overfitting_report,
)
from generative_models.models.base_autoencoder.trainer import (
    MODEL_NAME_PREFIX,
    build_ticker_slug,
)
from generative_models.serving.autoencoder_predictor import (
    AutoencoderPredictor,
)
from generative_models.serving.model_families import (
    build_predictor as build_family_predictor,
)
from generative_models.serving.model_families import (
    find_missing_serving_keys,
)
from generative_models.registry.model_registry import ModelRegistry

REPORT_SUFFIX = "-overfitting.json"
LOCAL_VERSION = "local"


@dataclass
class LocalModelArtifact:
    """A model stored in the local directory.

    Parameters
    ----------
    name : str
        Model name, for example `"simple-autoencoder-aapl"`.
    version : str
        Version label. It is always `"local"`.
    created_at : str
        Modification time of the weights file, in ISO 8601 format.
    metadata : dict
        Values of the overfitting report, which carry the architecture and
        the inference parameters.
    weight_path : Path
        Path of the `.pt` file with the trained weights.
    """

    name: str
    version: str
    created_at: str
    metadata: dict
    weight_path: Path


def build_predictor(
    artifact: LocalModelArtifact,
    device: str,
) -> AutoencoderPredictor:
    """Rebuild the autoencoder of an artifact and wrap it in a predictor.

    Parameters
    ----------
    artifact : LocalModelArtifact
        Model found in the model directory.
    device : str
        Device where the model runs, for example `"cpu"`.

    Returns
    -------
    AutoencoderPredictor
        Predictor ready for inference.
    """
    state_dict = torch.load(
        artifact.weight_path,
        map_location=device,
        weights_only=True,
    )

    return build_family_predictor(
        metadata=artifact.metadata,
        state_dict=state_dict,
        device=device,
        artifact_name=f"{artifact.name}:{artifact.version}",
    )


class LocalAutoencoderRegistry(ModelRegistry):
    """Registry of autoencoders read from a directory on disk.

    Each model is a pair of files written by the training: the report
    `<name>-overfitting.json`, which carries the architecture and the
    inference parameters, and the weights `<name>.pt`. A model is rebuilt on
    first use and kept in memory.

    Parameters
    ----------
    model_directory : str
        Directory that holds the model files, for example `models`.
    device : str | None
        Device where the models run. When `None`, `"cuda"` is used if
        available and `"cpu"` otherwise.

    Attributes
    ----------
    model_directory : Path
        Directory that holds the model files.
    device : str
        Device where the models run.
    models : dict[str, AutoencoderPredictor]
        Predictors kept in memory, indexed by ticker.

    Example
    -------
    >>> registry = LocalAutoencoderRegistry("models")
    >>> registry.load_all_models()
    ['AAPL', 'MSFT']
    """

    def __init__(
        self,
        model_directory: str,
        device: str | None = None,
    ) -> None:
        """Store the directory and the device. No file is read yet."""
        super().__init__()
        self.model_directory = Path(model_directory)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def list_available_artifacts(self) -> list[LocalModelArtifact]:
        """List the complete models found in the model directory.

        A report without its weights, or with a missing value, is skipped.

        Returns
        -------
        list[LocalModelArtifact]
            Models sorted by name.
        """
        if not self.model_directory.is_dir():
            return []

        artifacts = []
        report_paths = sorted(self.model_directory.glob(f"*{REPORT_SUFFIX}"))
        for report_path in report_paths:
            name = report_path.name[: -len(REPORT_SUFFIX)]
            weight_path = self.model_directory / f"{name}.pt"
            if not weight_path.is_file():
                continue
            metadata = asdict(load_overfitting_report(report_path))
            if find_missing_serving_keys(metadata):
                continue
            artifacts.append(
                LocalModelArtifact(
                    name=name,
                    version=LOCAL_VERSION,
                    created_at=datetime.fromtimestamp(
                        weight_path.stat().st_mtime
                    ).isoformat(),
                    metadata=metadata,
                    weight_path=weight_path,
                )
            )

        return artifacts

    def load_model(self, ticker: str) -> AutoencoderPredictor:
        """Read the model of a ticker from disk and keep it in memory.

        Parameters
        ----------
        ticker : str
            Ticker symbol, for example `"PETR4.SA"`.

        Returns
        -------
        AutoencoderPredictor
            Predictor of the model stored for the ticker.

        Raises
        ------
        LookupError
            If the directory has no complete model for the ticker.
        """
        model_name = MODEL_NAME_PREFIX + build_ticker_slug(ticker)
        for artifact in self.list_available_artifacts():
            if artifact.name == model_name:
                return self._load_artifact(artifact)

        raise LookupError(
            f"No model for ticker '{ticker}' in '{self.model_directory}'."
        )

    def load_latest_model(self) -> AutoencoderPredictor:
        """Read the most recently written model from disk.

        Returns
        -------
        AutoencoderPredictor
            Predictor of the model with the newest weights file.

        Raises
        ------
        LookupError
            If the directory has no complete model.
        """
        artifacts = self.list_available_artifacts()
        if not artifacts:
            raise LookupError(f"No model found in '{self.model_directory}'.")
        latest_artifact = max(
            artifacts,
            key=lambda artifact: artifact.created_at,
        )

        return self._load_artifact(latest_artifact)

    def load_all_models(self) -> list[str]:
        """Read every complete model of the directory into memory.

        Returns
        -------
        list[str]
            Tickers of the loaded models, in loading order.
        """
        return [
            self._load_artifact(artifact).metadata.ticker
            for artifact in self.list_available_artifacts()
        ]

    def get_model(self, ticker: str) -> AutoencoderPredictor:
        """Return the predictor of a ticker that is already in memory.

        Parameters
        ----------
        ticker : str
            Ticker symbol of a model loaded before.

        Returns
        -------
        AutoencoderPredictor
            Predictor kept in memory for `ticker`.

        Raises
        ------
        KeyError
            If no model was loaded for `ticker`.
        """
        if ticker not in self.models:
            raise KeyError(
                f"No model loaded for ticker '{ticker}'. "
                "Call load_model first."
            )

        return self.models[ticker]

    def list_loaded_tickers(self) -> list[str]:
        """List the tickers whose models are in memory.

        Returns
        -------
        list[str]
            Tickers in the order they were loaded.
        """
        return list(self.models.keys())

    def _load_artifact(
        self,
        artifact: LocalModelArtifact,
    ) -> AutoencoderPredictor:
        """Rebuild the model of an artifact and keep its predictor.

        Parameters
        ----------
        artifact : LocalModelArtifact
            Model found in the model directory.

        Returns
        -------
        AutoencoderPredictor
            Predictor kept in `models`, replacing any earlier predictor of
            the same ticker.
        """
        predictor = build_predictor(artifact, self.device)
        self.models[predictor.metadata.ticker] = predictor

        return predictor
