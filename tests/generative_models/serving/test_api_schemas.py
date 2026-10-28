"""Unit tests for the API schemas."""

import pytest
from pydantic import ValidationError

from generative_models.serving.api_schemas import (
    LogReturnPredictionRequest,
    ModelMetadataResponse,
    PricePredictionRequest,
    ReloadRequest,
    WindowPredictionResponse,
)


def test_log_return_request_requires_values() -> None:
    """CA-01: empty log-returns are rejected, one value is accepted."""
    with pytest.raises(ValidationError):
        LogReturnPredictionRequest(log_returns=[])

    prediction_request = LogReturnPredictionRequest(log_returns=[0.01])

    assert prediction_request.ticker is None


def test_price_request_requires_two_prices() -> None:
    """CA-02: fewer than two prices are rejected."""
    with pytest.raises(ValidationError):
        PricePredictionRequest(prices=[10.0])


@pytest.mark.parametrize("prices", [[10.0, 0.0, 11.0], [10.0, -1.0]])
def test_price_request_requires_positive_prices(prices: list[float]) -> None:
    """CA-03: zero or negative prices are rejected."""
    with pytest.raises(ValidationError):
        PricePredictionRequest(prices=prices)


def test_window_response_optional_fields_default_to_none() -> None:
    """CA-04: date and price are serialized as None when omitted."""
    window_response = WindowPredictionResponse(
        window_end_index=4,
        reconstruction_score=0.5,
        is_outlier=False,
    )

    serialized_window = window_response.model_dump()

    assert serialized_window["window_end_date"] is None
    assert serialized_window["close_price"] is None


def test_reload_request_without_ticker() -> None:
    """CA-05: ReloadRequest is valid without a ticker."""
    assert ReloadRequest().ticker is None


def test_schemas_document_examples_and_descriptions() -> None:
    """CA-06: request examples and metadata descriptions are present."""
    price_schema = PricePredictionRequest.model_json_schema()
    metadata_schema = ModelMetadataResponse.model_json_schema()

    assert "examples" in price_schema
    assert all(
        "description" in field_schema
        for field_schema in metadata_schema["properties"].values()
    )
