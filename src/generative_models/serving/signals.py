"""Trading signals derived from the outlier detection of the models."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from generative_models.serving.api_schemas import SignalItem
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
)

MAXIMUM_MODEL_NAME_LENGTH = 60
MAXIMUM_REASON_LENGTH = 280


@dataclass(frozen=True)
class SignalDecision:
    """Decision produced by a signal rule for one model.

    Parameters
    ----------
    action : str
        `"buy"`, `"sell"` or `"hold"`.
    confidence : float
        Confidence of the decision, between 0 and 1 inclusive.
    reason : str | None
        Plain explanation based only on the model numbers.
    """

    action: str
    confidence: float
    reason: str | None = None


SignalRule = Callable[[AutoencoderModelMetadata], SignalDecision | None]

# The rule that turns an anomaly into buy, sell or hold is still a team
# decision. Until it is set, no signal is generated and GET /signals returns
# an empty list.
DEFAULT_SIGNAL_RULE: SignalRule | None = None


def build_signals(
    metadatas: list[AutoencoderModelMetadata],
    rule: SignalRule | None,
    generated_at: datetime,
) -> list[SignalItem]:
    """Turn the loaded models into signal items.

    Parameters
    ----------
    metadatas : list[AutoencoderModelMetadata]
        Metadata of the models currently in memory.
    rule : SignalRule | None
        Function that decides the signal of one model. When `None`, no
        signal is generated.
    generated_at : datetime
        Moment shared by every item of the answer.

    Returns
    -------
    list[SignalItem]
        One item for each model that received a decision. Models without a
        decision are left out, and an empty list is returned when there is
        no rule.

    Example
    -------
    >>> build_signals(metadatas, None, datetime.now(UTC))
    []
    """
    if rule is None:
        return []

    items = []
    for metadata in metadatas:
        decision = rule(metadata)
        if decision is None:
            continue
        model_name = metadata.artifact_name
        items.append(
            SignalItem(
                symbol=metadata.ticker,
                action=decision.action,
                confidence=decision.confidence,
                generated_at=generated_at,
                model=(
                    model_name
                    if len(model_name) <= MAXIMUM_MODEL_NAME_LENGTH
                    else None
                ),
                reason=decision.reason,
            )
        )

    return items
