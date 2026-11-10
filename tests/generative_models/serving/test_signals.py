"""Unit tests for the trading signals of the models and their schema."""

from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from generative_models.serving.api_schemas import SignalItem
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
)
from generative_models.serving.signals import (
    SignalDecision,
    build_signal_history,
    build_signals,
    clip_confidence,
    default_signal_rule,
)

GENERATED_AT = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
THRESHOLD = 1.0


def _metadata(
    ticker: str,
    artifact_name: str,
    anomaly_threshold: float = THRESHOLD,
) -> AutoencoderModelMetadata:
    """Build the metadata of a model for the tests."""
    return AutoencoderModelMetadata(
        ticker=ticker,
        window_size=10,
        return_train_mean=0.0,
        return_train_std=0.02,
        anomaly_threshold=anomaly_threshold,
        artifact_name=artifact_name,
    )


def _prediction_table(
    closes: list[float],
    scores: list[float],
) -> pd.DataFrame:
    """Build a prediction table with one row per window."""
    dates = pd.date_range("2026-10-01", periods=len(closes), freq="B")

    return pd.DataFrame(
        {
            "close_price": closes,
            "reconstruction_score": scores,
            "is_outlier": [score > THRESHOLD for score in scores],
        },
        index=dates,
    )


def _valid_item(**overrides) -> dict:
    """Build the data of a valid signal item, with optional changes."""
    data = {
        "symbol": "AAPL",
        "action": "hold",
        "confidence": 0.5,
        "generated_at": GENERATED_AT,
        "model": "simple-autoencoder-aapl",
        "reason": "Janela de 10 dias sem erro acima do limite.",
    }
    data.update(overrides)

    return data


def test_no_rule_gives_an_empty_list() -> None:
    """Check that without a signal rule no item is returned."""
    tables = [
        (
            _metadata("AAPL", "simple-autoencoder-aapl:local"),
            _prediction_table([10.0, 11.0], [0.1, 0.2]),
        )
    ]

    assert not build_signals(tables, None, GENERATED_AT)


def test_models_without_a_decision_are_left_out() -> None:
    """Check that only the models with a decision become items."""

    def rule(
        metadata: AutoencoderModelMetadata,
        _prediction_table: pd.DataFrame,
    ) -> SignalDecision | None:
        if metadata.ticker == "AAPL":
            return SignalDecision(action="sell", confidence=0.8)
        return None

    tables = [
        (
            _metadata("AAPL", "simple-autoencoder-aapl:local"),
            _prediction_table([10.0, 11.0], [0.1, 0.2]),
        ),
        (
            _metadata("MSFT", "simple-autoencoder-msft:local"),
            _prediction_table([10.0, 11.0], [0.1, 0.2]),
        ),
    ]

    items = build_signals(tables, rule, GENERATED_AT)

    assert [item.symbol for item in items] == ["AAPL"]
    assert items[0].action == "sell"


def test_build_signals_copies_the_model_and_the_time() -> None:
    """Check the model name and the generation time of each item."""

    def rule(
        _metadata_value: AutoencoderModelMetadata,
        _prediction_table: pd.DataFrame,
    ) -> SignalDecision:
        return SignalDecision(action="buy", confidence=0.6, reason="Teste.")

    items = build_signals(
        [
            (
                _metadata("AAPL", "simple-autoencoder-aapl:local"),
                _prediction_table([10.0, 11.0], [0.1, 0.2]),
            )
        ],
        rule,
        GENERATED_AT,
    )

    assert items[0].model == "simple-autoencoder-aapl:local"
    assert items[0].generated_at == GENERATED_AT
    assert items[0].reason == "Teste."


def test_model_name_longer_than_60_characters_is_not_reported() -> None:
    """Check that a long model name is left empty, not truncated."""

    def rule(
        _metadata_value: AutoencoderModelMetadata,
        _prediction_table: pd.DataFrame,
    ) -> SignalDecision:
        return SignalDecision(action="hold", confidence=0.1)

    long_name = "simple-autoencoder-" + "x" * 60
    items = build_signals(
        [
            (
                _metadata("AAPL", long_name),
                _prediction_table([10.0, 11.0], [0.1, 0.2]),
            )
        ],
        rule,
        GENERATED_AT,
    )

    assert items[0].model is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-0.5, 0.0), (0.3, 0.3), (1.7, 1.0)],
)
def test_clip_confidence_limits_the_value(
    value: float,
    expected: float,
) -> None:
    """Check that the confidence stays between 0 and 1."""
    assert clip_confidence(value) == pytest.approx(expected)


def test_default_rule_holds_when_the_last_window_is_normal() -> None:
    """Check the hold action and its confidence within the threshold."""
    decision = default_signal_rule(
        _metadata("MSFT", "simple-autoencoder-msft:local"),
        _prediction_table([10.0, 10.5], [0.2, 0.25]),
    )

    assert decision.action == "hold"
    assert decision.confidence == pytest.approx(0.75)
    assert decision.reason is None


def test_default_rule_buys_after_an_abnormal_drop() -> None:
    """Check that an outlier with a falling price gives a buy."""
    decision = default_signal_rule(
        _metadata("AAPL", "simple-autoencoder-aapl:local"),
        _prediction_table([10.0, 9.0], [0.5, 1.5]),
    )

    assert decision.action == "buy"
    assert decision.confidence == pytest.approx(0.5)
    assert decision.reason.startswith("Janela de 10 dias")


def test_default_rule_sells_after_an_abnormal_rise() -> None:
    """Check that an outlier with a rising price gives a sell."""
    decision = default_signal_rule(
        _metadata("AAPL", "simple-autoencoder-aapl:local"),
        _prediction_table([10.0, 11.0], [0.5, 1.2]),
    )

    assert decision.action == "sell"
    assert decision.confidence == pytest.approx(0.2)


def test_default_rule_holds_an_outlier_with_a_flat_price() -> None:
    """Check that an outlier without a price move gives a hold."""
    decision = default_signal_rule(
        _metadata("AAPL", "simple-autoencoder-aapl:local"),
        _prediction_table([10.0, 10.0], [0.5, 2.0]),
    )

    assert decision.action == "hold"
    assert decision.confidence == pytest.approx(1.0)
    assert decision.reason is not None


def test_default_rule_needs_two_windows() -> None:
    """Check that a single window gives no signal."""
    decision = default_signal_rule(
        _metadata("AAPL", "simple-autoencoder-aapl:local"),
        _prediction_table([10.0], [0.5]),
    )

    assert decision is None


def test_default_rule_needs_a_positive_threshold() -> None:
    """Check that a threshold of zero gives no signal."""
    decision = default_signal_rule(
        _metadata("AAPL", "simple-autoencoder-aapl:local", 0.0),
        _prediction_table([10.0, 11.0], [0.5, 1.2]),
    )

    assert decision is None


def test_valid_item_is_accepted() -> None:
    """Check that a complete item passes the schema."""
    item = SignalItem(**_valid_item())

    assert item.symbol == "AAPL"
    assert item.confidence == 0.5


@pytest.mark.parametrize("action", ["long", "BUY", "", None])
def test_action_is_restricted_to_buy_sell_and_hold(action) -> None:
    """Check that any other action is refused."""
    with pytest.raises(ValidationError):
        SignalItem(**_valid_item(action=action))


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_must_be_between_0_and_1(confidence: float) -> None:
    """Check that confidence outside the range is refused."""
    with pytest.raises(ValidationError):
        SignalItem(**_valid_item(confidence=confidence))


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_confidence_limits_are_accepted(confidence: float) -> None:
    """Check that 0 and 1 are valid confidence values."""
    assert SignalItem(**_valid_item(confidence=confidence)).confidence == (
        confidence
    )


def test_symbol_is_stored_in_upper_case() -> None:
    """Check that a lower case ticker is converted to upper case."""
    assert SignalItem(**_valid_item(symbol="aapl")).symbol == "AAPL"


def test_generated_at_without_timezone_is_refused() -> None:
    """Check that a moment without a timezone is refused."""
    with pytest.raises(ValidationError):
        SignalItem(**_valid_item(generated_at=datetime(2026, 11, 2, 12)))


def test_generated_at_is_converted_to_utc() -> None:
    """Check that a moment in another timezone is stored in UTC."""
    local_moment = datetime(
        2026,
        11,
        2,
        9,
        0,
        tzinfo=timezone(timedelta(hours=-3)),
    )

    item = SignalItem(**_valid_item(generated_at=local_moment))

    assert item.generated_at == datetime(2026, 11, 2, 12, 0, tzinfo=UTC)


def test_reason_longer_than_280_characters_is_refused() -> None:
    """Check the maximum size of the reason."""
    with pytest.raises(ValidationError):
        SignalItem(**_valid_item(reason="x" * 281))


def test_model_and_reason_can_be_null() -> None:
    """Check that the optional fields accept null."""
    item = SignalItem(**_valid_item(model=None, reason=None))

    assert item.model is None
    assert item.reason is None


def test_history_gives_one_item_per_day_from_the_second_window() -> None:
    """Check that the history has one item per day from the second window."""
    table = _prediction_table(
        closes=[10.0, 11.0, 9.0, 12.0],
        scores=[0.5, 0.6, 2.0, 0.4],
    )
    items = build_signal_history(
        _metadata("AAPL", "simple-aapl"),
        table,
        default_signal_rule,
    )

    assert len(items) == 3
    assert [item.generated_at.date().isoformat() for item in items] == [
        "2026-10-02",
        "2026-10-05",
        "2026-10-06",
    ]
    assert [item.action for item in items] == ["hold", "buy", "hold"]


def test_history_uses_only_the_windows_known_each_day() -> None:
    """Check that each day is decided without the windows after it."""
    table = _prediction_table(
        closes=[10.0, 11.0, 9.0, 12.0],
        scores=[0.5, 0.6, 0.7, 2.0],
    )
    rule_calls = []

    def rule(_metadata, prediction_table):
        """Record the size of the table seen by the rule."""
        rule_calls.append(len(prediction_table))
        return SignalDecision(action="hold", confidence=0.5)

    build_signal_history(_metadata("AAPL", "simple-aapl"), table, rule)
    assert rule_calls == [2, 3, 4]


def test_history_without_a_rule_is_empty() -> None:
    """Check that the history is empty when there is no rule."""
    table = _prediction_table(closes=[10.0, 11.0], scores=[0.5, 0.6])
    assert not build_signal_history(
        _metadata("AAPL", "simple-aapl"),
        table,
        None,
    )
