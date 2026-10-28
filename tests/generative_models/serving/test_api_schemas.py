"""Unit tests for the autoencoder REST API schemas."""

import pytest
from pydantic import ValidationError

from generative_models.serving.api_schemas import (
    LogReturnPredictionRequest,
    ModelMetadataResponse,
    PricePredictionRequest,
    ReloadRequest,
    WindowPredictionResponse,
)


def test_log_return_request_requires_one_value() -> None:
    """CA-01: check the minimum length and the optional ticker."""
    with pytest.raises(ValidationError):
        LogReturnPredictionRequest(log_returns=[])

    request = LogReturnPredictionRequest(log_returns=[0.01])

    assert request.ticker is None


def test_price_request_requires_two_prices() -> None:
    """CA-02: check that a single price is rejected."""
    with pytest.raises(ValidationError):
        PricePredictionRequest(prices=[10.0])


@pytest.mark.parametrize(
    "prices",
    [[10.0, 0.0, 11.0], [10.0, -1.0]],
)
def test_price_request_rejects_non_positive_prices(
    prices: list[float],
) -> None:
    """CA-03: check that zero and negative prices are rejected."""
    with pytest.raises(ValidationError):
        PricePredictionRequest(prices=prices)


def test_window_response_defaults_market_fields_to_none() -> None:
    """CA-04: check that date and price are serialized as None."""
    window_response = WindowPredictionResponse(
        window_end_index=4,
        reconstruction_score=0.5,
        is_outlier=False,
    )

    serialized_window = window_response.model_dump()

    assert serialized_window["window_end_date"] is None
    assert serialized_window["close_price"] is None


def test_reload_request_without_ticker_is_valid() -> None:
    """CA-05: check that the reload body may be empty."""
    assert ReloadRequest().ticker is None


def test_schemas_document_examples_and_descriptions() -> None:
    """CA-06: check the OpenAPI examples and field descriptions."""
    price_request_schema = PricePredictionRequest.model_json_schema()
    metadata_properties = ModelMetadataResponse.model_json_schema()[
        "properties"
    ]

    assert "examples" in price_request_schema
    assert all(
        "description" in field_schema
        for field_schema in metadata_properties.values()
    )
