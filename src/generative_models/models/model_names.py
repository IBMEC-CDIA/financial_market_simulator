"""Names derived from a model package, shared by training and serving.

The experiment name that a model is registered under and the prefix of its
artifact names are both derived from the model package, so the training and
the build never need to be told the same two values again.
"""

import importlib
import re

from generative_models.models.base_autoencoder.config import (
    validate_experiment_name,
)

MODEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
TRAINER_MODULE_NAME = "trainer"
NAME_PREFIX_ATTRIBUTE = "MODEL_NAME_PREFIX"


def validate_model_name(model_name: str) -> None:
    """Check that a model name is a valid package name.

    Parameters
    ----------
    model_name : str
        Name of the model package, for example `"lstm_autoencoder"`.

    Raises
    ------
    ValueError
        If the name is not in `snake_case`, or contains path characters.
    """
    if not MODEL_NAME_PATTERN.match(model_name):
        raise ValueError(
            f"Invalid model name '{model_name}': use lowercase letters, "
            "digits and underscores, starting with a letter."
        )


def derive_experiment_name(model_name: str) -> str:
    """Return the experiment name of a model, from its package name.

    The underscores of the package name become hyphens, so the name is
    the alias the models are registered under in wandb.

    Parameters
    ----------
    model_name : str
        Name of the model package, for example `"convolutional_autoencoder"`.

    Returns
    -------
    str
        Experiment name, for example `"convolutional-autoencoder"`.

    Raises
    ------
    ValueError
        If the model name is invalid, or the derived experiment name is not
        accepted by wandb as an alias.

    Example
    -------
    >>> derive_experiment_name("convolutional_autoencoder")
    'convolutional-autoencoder'
    """
    validate_model_name(model_name)

    return validate_experiment_name(model_name.replace("_", "-"))


def load_artifact_prefix(model_name: str) -> str:
    """Return the prefix of the wandb artifact names of a model.

    The prefix is the `MODEL_NAME_PREFIX` constant of the trainer module of
    the model package, so it is the one the training registers with.

    Parameters
    ----------
    model_name : str
        Name of the model package, for example `"base_autoencoder"`.

    Returns
    -------
    str
        Prefix of the artifact names, for example `"simple-autoencoder-"`.

    Raises
    ------
    ValueError
        If the model name is invalid, or the trainer module does not define
        a string `MODEL_NAME_PREFIX`.
    ModuleNotFoundError
        If the model package has no trainer module.

    Example
    -------
    >>> load_artifact_prefix("base_autoencoder")
    'simple-autoencoder-'
    """
    validate_model_name(model_name)
    trainer_module = importlib.import_module(
        f"generative_models.models.{model_name}.{TRAINER_MODULE_NAME}"
    )
    prefix = getattr(trainer_module, NAME_PREFIX_ATTRIBUTE, None)
    if not isinstance(prefix, str) or not prefix:
        raise ValueError(
            f"The trainer of '{model_name}' must define a non-empty "
            f"{NAME_PREFIX_ATTRIBUTE} string."
        )

    return prefix
