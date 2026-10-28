"""Unit tests for the REST API schemas."""

import pytest
from pydantic import ValidationError

from generative_models.serving.api_schemas import (
    LogReturnPredictionRequest,
    ModelMetadataResponse,
    PricePredictionRequest,
    ReloadRequest,
    WindowPredictionResponse,
)


def test_log_returns_require_at_least_one_value() -> None:
    """CA-01: an empty log-return list is rejected, a single one is not."""
    with pytest.raises(ValidationError):
        LogReturnPredictionRequest(log_returns=[])

    request = LogReturnPredictionRequest(log_returns=[0.01])

    assert request.ticker is None


def test_prices_require_at_least_two_values() -> None:
    """CA-02: a single price cannot produce a log-return."""
    with pytest.raises(ValidationError):
        PricePredictionRequest(prices=[10.0])


@pytest.mark.parametrize(
    "prices",
    [[10.0, 0.0, 11.0], [10.0, -1.0]],
)
def test_prices_must_be_strictly_positive(prices: list[float]) -> None:
    """CA-03: zero and negative prices are rejected."""
    with pytest.raises(ValidationError):
        PricePredictionRequest(prices=prices)


def test_window_response_defaults_market_fields_to_none() -> None:
    """CA-04: the date and price of a window default to null."""
    window = WindowPredictionResponse(
        window_end_index=4,
        reconstruction_score=0.5,
        is_outlier=False,
    )

    serialized_window = window.model_dump()

    assert serialized_window["window_end_date"] is None
    assert serialized_window["close_price"] is None


def test_reload_request_accepts_an_empty_body() -> None:
    """CA-05: a reload without a ticker targets the default model."""
    request = ReloadRequest()

    assert request.ticker is None


def test_schemas_are_documented_for_openapi() -> None:
    """CA-06: bodies carry examples and metadata fields descriptions."""
    price_schema = PricePredictionRequest.model_json_schema()

    assert "examples" in price_schema

    metadata_schema = ModelMetadataResponse.model_json_schema()

    assert all(
        "description" in field_schema
        for field_schema in metadata_schema["properties"].values()
    )
