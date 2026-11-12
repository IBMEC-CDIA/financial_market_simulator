"""Choose the model family of a stored model and rebuild its predictor.

Each model family in `generative_models.models` provides a `serving` module
with two functions. `build_model(architecture)` returns the untrained
network from the constructor arguments saved with the model, and
`score_windows(model, input_tensor)` returns the reconstruction score of
each window. Artifacts without a stored family are served as the base
autoencoder, which keeps its architecture in the legacy keys.
"""

import importlib
from collections.abc import Mapping
from types import ModuleType

import torch

from generative_models.models.model_names import validate_model_name
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPredictor,
)

DEFAULT_MODEL_FAMILY = "base_autoencoder"
SERVING_MODULE_NAME = "serving"
COMMON_SERVING_KEYS = (
    "ticker",
    "window_size",
    "return_train_mean",
    "return_train_std",
    "anomaly_threshold",
)
LEGACY_ARCHITECTURE_KEYS = ("hidden_layer_size", "latent_layer_size")


def resolve_model_family(metadata: Mapping) -> str:
    """Return the model family of a stored model.

    Parameters
    ----------
    metadata : Mapping
        Metadata of the model. The family is read from `model_family`.

    Returns
    -------
    str
        Name of the model package, for example `"lstm_autoencoder"`. A
        model without a stored family is the base autoencoder.

    Example
    -------
    >>> resolve_model_family({})
    'base_autoencoder'
    """
    return metadata.get("model_family") or DEFAULT_MODEL_FAMILY


def resolve_architecture(metadata: Mapping) -> dict[str, int]:
    """Return the constructor arguments of the stored model.

    Parameters
    ----------
    metadata : Mapping
        Metadata of the model. The arguments are read from `architecture`.
        Base autoencoders without it use `window_size`,
        `hidden_layer_size` and `latent_layer_size`.

    Returns
    -------
    dict[str, int]
        Constructor arguments of the model family.

    Raises
    ------
    ValueError
        If a model of another family has no stored architecture.

    Example
    -------
    >>> resolve_architecture(
    ...     {"window_size": 10, "hidden_layer_size": 8, "latent_layer_size": 4}
    ... )
    {'input_dim': 10, 'hidden_dim': 8, 'latent_dim': 4}
    """
    architecture = metadata.get("architecture")
    if architecture:
        return {key: int(value) for key, value in architecture.items()}
    if resolve_model_family(metadata) != DEFAULT_MODEL_FAMILY:
        raise ValueError(
            "The model has no stored architecture: "
            f"'{resolve_model_family(metadata)}' needs one."
        )

    return {
        "input_dim": int(metadata["window_size"]),
        "hidden_dim": int(metadata["hidden_layer_size"]),
        "latent_dim": int(metadata["latent_layer_size"]),
    }


def find_missing_serving_keys(metadata: Mapping) -> list[str]:
    """List the values that inference needs and the metadata does not have.

    Parameters
    ----------
    metadata : Mapping
        Metadata of a stored model.

    Returns
    -------
    list[str]
        Missing keys. The architecture is listed as `architecture`, or as
        the legacy keys of the base autoencoder.

    Example
    -------
    >>> find_missing_serving_keys({"ticker": "AAPL"})[-1]
    'latent_layer_size'
    """
    missing_keys = [
        key for key in COMMON_SERVING_KEYS if metadata.get(key) is None
    ]
    if metadata.get("architecture"):
        return missing_keys
    if resolve_model_family(metadata) == DEFAULT_MODEL_FAMILY:
        missing_keys.extend(
            key
            for key in LEGACY_ARCHITECTURE_KEYS
            if metadata.get(key) is None
        )
    else:
        missing_keys.append("architecture")

    return missing_keys


def load_serving_module(model_family: str) -> ModuleType:
    """Import the serving module of a model family.

    Parameters
    ----------
    model_family : str
        Name of the model package, for example `"gru_autoencoder"`.

    Returns
    -------
    ModuleType
        The `serving` module, with `build_model` and `score_windows`.

    Raises
    ------
    ValueError
        If the family name is invalid.
    ModuleNotFoundError
        If the family has no `serving` module, so it cannot be served.
    """
    validate_model_name(model_family)

    return importlib.import_module(
        f"generative_models.models.{model_family}.{SERVING_MODULE_NAME}"
    )


def build_predictor(
    metadata: Mapping,
    state_dict: Mapping[str, torch.Tensor],
    device: str,
    artifact_name: str,
) -> AutoencoderPredictor:
    """Rebuild a stored model of any family and wrap it in a predictor.

    Parameters
    ----------
    metadata : Mapping
        Metadata of the stored model, with the serving keys.
    state_dict : Mapping[str, torch.Tensor]
        Trained weights of the network.
    device : str
        Device where the model runs, for example `"cpu"`.
    artifact_name : str
        Name of the artifact, kept in the predictor metadata.

    Returns
    -------
    AutoencoderPredictor
        Predictor that scores the windows with the family's scorer.

    Example
    -------
    >>> predictor = build_predictor(
    ...     metadata, weights, "cpu", "simple-autoencoder-aapl:v1"
    ... )
    """
    model_family = resolve_model_family(metadata)
    serving_module = load_serving_module(model_family)
    model = serving_module.build_model(resolve_architecture(metadata))
    model.load_state_dict(state_dict)
    model.to(device)

    predictor_metadata = AutoencoderModelMetadata(
        ticker=str(metadata["ticker"]),
        window_size=int(metadata["window_size"]),
        return_train_mean=float(metadata["return_train_mean"]),
        return_train_std=float(metadata["return_train_std"]),
        anomaly_threshold=float(metadata["anomaly_threshold"]),
        artifact_name=artifact_name,
    )

    return AutoencoderPredictor(
        model=model,
        metadata=predictor_metadata,
        device=device,
        reconstruction_scorer=serving_module.score_windows,
    )
