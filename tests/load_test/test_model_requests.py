"""Tests for the request helpers of the Locust test."""

from datetime import date

import pytest

from load_test.model_requests import history_period, tickers_from_payload


def test_tickers_are_unique_upper_case_and_sorted():
    payload = [
        {"ticker": "googl"},
        {"ticker": "AAPL"},
        {"ticker": "aapl"},
    ]

    assert tickers_from_payload(payload) == ["AAPL", "GOOGL"]


def test_items_without_a_ticker_are_ignored():
    payload = [{"ticker": ""}, {"other": 1}, "not a dict", {"ticker": "MSFT"}]

    assert tickers_from_payload(payload) == ["MSFT"]


def test_an_empty_list_gives_no_tickers():
    assert tickers_from_payload([]) == []


def test_a_payload_that_is_not_a_list_is_refused():
    with pytest.raises(ValueError, match="must be a list"):
        tickers_from_payload({"ticker": "AAPL"})


def test_history_period_goes_back_the_given_days():
    assert history_period(date(2026, 11, 10), 90) == (
        "2026-08-12",
        "2026-11-10",
    )
