"""Unit tests for the request and response schemas of the API."""

import pytest
from pydantic import ValidationError

from generative_models.serving.api_schemas import (
    LogReturnPredictionRequest,
    ModelMetadataResponse,
    PricePredictionRequest,
    ReloadRequest,
    WindowPredictionResponse,
)


def test_empty_log_returns_are_rejected_and_one_value_is_accepted() -> None:
    """Check the minimum length of the log-returns list."""
    with pytest.raises(ValidationError):
        LogReturnPredictionRequest(log_returns=[])

    request = LogReturnPredictionRequest(log_returns=[0.01])

    assert request.ticker is None


def test_single_price_is_rejected() -> None:
    """Check that at least two prices are required."""
    with pytest.raises(ValidationError):
        PricePredictionRequest(prices=[10.0])


@pytest.mark.parametrize(
    "prices",
    [[10.0, 0.0, 11.0], [10.0, -1.0]],
)
def test_non_positive_prices_are_rejected(prices: list[float]) -> None:
    """Check that every price must be strictly positive."""
    with pytest.raises(ValidationError):
        PricePredictionRequest(prices=prices)


def test_window_without_market_fields_serializes_them_as_null() -> None:
    """Check that the market fields default to null."""
    window = WindowPredictionResponse(
        window_end_index=4,
        reconstruction_score=0.1,
        is_outlier=False,
    )

    serialized_window = window.model_dump()

    assert serialized_window["window_end_date"] is None
    assert serialized_window["close_price"] is None


def test_reload_request_accepts_an_empty_body() -> None:
    """Check that the ticker is optional in the reload request."""
    request = ReloadRequest()

    assert request.ticker is None


def test_json_schema_has_examples_and_field_descriptions() -> None:
    """Check the OpenAPI information of the request and response schemas."""
    price_schema = PricePredictionRequest.model_json_schema()
    metadata_schema = ModelMetadataResponse.model_json_schema()

    assert "examples" in price_schema
    for field_name, field_schema in metadata_schema["properties"].items():
        assert "description" in field_schema, field_name
