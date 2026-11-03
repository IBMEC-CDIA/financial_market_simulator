"""Unit tests for the trading signals of the models and their schema."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from generative_models.serving.api_schemas import SignalItem
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
)
from generative_models.serving.signals import (
    SignalDecision,
    build_signals,
)

GENERATED_AT = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)


def _metadata(ticker: str, artifact_name: str) -> AutoencoderModelMetadata:
    """Build the metadata of a model for the tests."""
    return AutoencoderModelMetadata(
        ticker=ticker,
        window_size=10,
        return_train_mean=0.0,
        return_train_std=0.02,
        anomaly_threshold=1.3,
        artifact_name=artifact_name,
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
    metadatas = [_metadata("AAPL", "simple-autoencoder-aapl:local")]

    assert not build_signals(metadatas, None, GENERATED_AT)


def test_models_without_a_decision_are_left_out() -> None:
    """Check that only the models with a decision become items."""

    def rule(metadata: AutoencoderModelMetadata) -> SignalDecision | None:
        if metadata.ticker == "AAPL":
            return SignalDecision(action="sell", confidence=0.8)
        return None

    metadatas = [
        _metadata("AAPL", "simple-autoencoder-aapl:local"),
        _metadata("MSFT", "simple-autoencoder-msft:local"),
    ]

    items = build_signals(metadatas, rule, GENERATED_AT)

    assert [item.symbol for item in items] == ["AAPL"]
    assert items[0].action == "sell"


def test_build_signals_copies_the_model_and_the_time() -> None:
    """Check the model name and the generation time of each item."""

    def rule(_metadata_value) -> SignalDecision:
        return SignalDecision(action="buy", confidence=0.6, reason="Teste.")

    items = build_signals(
        [_metadata("AAPL", "simple-autoencoder-aapl:local")],
        rule,
        GENERATED_AT,
    )

    assert items[0].model == "simple-autoencoder-aapl:local"
    assert items[0].generated_at == GENERATED_AT
    assert items[0].reason == "Teste."


def test_model_name_longer_than_60_characters_is_not_reported() -> None:
    """Check that a long model name is left empty, not truncated."""

    def rule(_metadata_value) -> SignalDecision:
        return SignalDecision(action="hold", confidence=0.1)

    long_name = "simple-autoencoder-" + "x" * 60
    items = build_signals(
        [_metadata("AAPL", long_name)],
        rule,
        GENERATED_AT,
    )

    assert items[0].model is None


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
