"""Trading signals derived from the outlier detection of the models."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from generative_models.serving.api_schemas import SignalItem
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
)

MAXIMUM_MODEL_NAME_LENGTH = 60
MAXIMUM_REASON_LENGTH = 280
MINIMUM_WINDOWS_FOR_SIGNAL = 2


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


SignalRule = Callable[
    [AutoencoderModelMetadata, pd.DataFrame],
    SignalDecision | None,
]


def clip_confidence(value: float) -> float:
    """Limit a confidence value to the interval from 0 to 1.

    Parameters
    ----------
    value : float
        Raw confidence value.

    Returns
    -------
    float
        `value` limited to the interval from 0 to 1 inclusive.

    Example
    -------
    >>> clip_confidence(1.7)
    1.0
    """
    return min(max(value, 0.0), 1.0)


def default_signal_rule(
    metadata: AutoencoderModelMetadata,
    prediction_table: pd.DataFrame,
) -> SignalDecision | None:
    """Turn the most recent window of a model into a signal.

    The most recent window is compared with the anomaly threshold of the
    model. When it is within the threshold, the action is `"hold"` and
    the confidence grows as the score gets further below the threshold.
    When it is above the threshold, the action follows the last daily
    move of the price: an abnormal drop gives `"buy"`, an abnormal rise
    gives `"sell"`, and no move gives `"hold"`. The confidence grows as
    the score gets further above the threshold.

    Parameters
    ----------
    metadata : AutoencoderModelMetadata
        Training metadata of the model, with the anomaly threshold.
    prediction_table : pd.DataFrame
        Prediction table with the columns `close_price` and
        `reconstruction_score`, one row per window, in date order.

    Returns
    -------
    SignalDecision | None
        The decision, or `None` when there are fewer than two windows or
        the threshold is not positive, so no signal can be computed.

    Example
    -------
    >>> decision = default_signal_rule(metadata, prediction_table)
    >>> decision.action
    'sell'
    """
    threshold = metadata.anomaly_threshold
    if len(prediction_table) < MINIMUM_WINDOWS_FOR_SIGNAL or threshold <= 0:
        return None

    last_score = float(prediction_table["reconstruction_score"].iloc[-1])
    last_close = float(prediction_table["close_price"].iloc[-1])
    previous_close = float(prediction_table["close_price"].iloc[-2])

    if last_score <= threshold:
        return SignalDecision(
            action="hold",
            confidence=clip_confidence(1 - last_score / threshold),
        )

    confidence = clip_confidence((last_score - threshold) / threshold)
    reason = (
        f"Janela de {metadata.window_size} dias com erro de reconstrução "
        "acima do limite."
    )
    if last_close < previous_close:
        return SignalDecision(
            action="buy",
            confidence=confidence,
            reason=reason,
        )
    if last_close > previous_close:
        return SignalDecision(
            action="sell",
            confidence=confidence,
            reason=reason,
        )

    return SignalDecision(action="hold", confidence=confidence, reason=reason)


def build_signals(
    prediction_tables: list[tuple[AutoencoderModelMetadata, pd.DataFrame]],
    rule: SignalRule | None,
    generated_at: datetime,
) -> list[SignalItem]:
    """Turn the recent predictions of the models into signal items.

    Parameters
    ----------
    prediction_tables : list[tuple[AutoencoderModelMetadata, pd.DataFrame]]
        For each model, its metadata and its recent prediction table, as
        built by `build_prediction_table`.
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
    >>> build_signals(tables, default_signal_rule, datetime.now(UTC))
    """
    if rule is None:
        return []

    items = []
    for metadata, prediction_table in prediction_tables:
        decision = rule(metadata, prediction_table)
        if decision is None:
            continue
        items.append(signal_item(metadata, decision, generated_at))

    return items


def signal_item(
    metadata: AutoencoderModelMetadata,
    decision: SignalDecision,
    generated_at: datetime,
) -> SignalItem:
    """Build the public item of one decision of a model.

    Parameters
    ----------
    metadata : AutoencoderModelMetadata
        Training metadata of the model that made the decision.
    decision : SignalDecision
        Decision produced by the signal rule.
    generated_at : datetime
        Moment of the suggestion, in UTC.

    Returns
    -------
    SignalItem
        Suggestion with the model name when it fits the length limit.

    Example
    -------
    >>> signal_item(metadata, decision, datetime.now(UTC))  # doctest: +SKIP
    """
    model_name = metadata.artifact_name
    return SignalItem(
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


def build_signal_history(
    metadata: AutoencoderModelMetadata,
    prediction_table: pd.DataFrame,
    rule: SignalRule | None,
) -> list[SignalItem]:
    """Apply the signal rule to every day of a prediction table.

    The rule is applied to the table up to each day, so each day gets the
    suggestion the model would have given on that day, using only the windows
    known until then. The first days, before the rule has enough windows, get
    no suggestion.

    Parameters
    ----------
    metadata : AutoencoderModelMetadata
        Training metadata of the model.
    prediction_table : pd.DataFrame
        Prediction table, one row per window, indexed by date, in date order.
    rule : SignalRule | None
        Function that decides the signal of one day. When `None`, no signal
        is generated.

    Returns
    -------
    list[SignalItem]
        One item per day with a decision, in date order. Holds are included.

    Example
    -------
    >>> build_signal_history(metadata, table, rule)  # doctest: +SKIP
    """
    if rule is None:
        return []

    items = []
    first_position = MINIMUM_WINDOWS_FOR_SIGNAL - 1
    for position in range(first_position, len(prediction_table)):
        decision = rule(metadata, prediction_table.iloc[: position + 1])
        if decision is None:
            continue
        day = pd.Timestamp(prediction_table.index[position]).date()
        generated_at = datetime(
            day.year,
            day.month,
            day.day,
            tzinfo=UTC,
        )
        items.append(signal_item(metadata, decision, generated_at))

    return items
