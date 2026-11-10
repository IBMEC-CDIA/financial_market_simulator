"""Tests for the pass or fail rule of the Locust test."""

from load_test.thresholds import failure_ratio_within_limit


def test_a_test_without_requests_does_not_pass():
    assert failure_ratio_within_limit(0, 0, 0.01) is False


def test_failures_at_the_limit_pass():
    assert failure_ratio_within_limit(1000, 10, 0.01) is True


def test_failures_above_the_limit_fail():
    assert failure_ratio_within_limit(1000, 11, 0.01) is False


def test_a_test_without_failures_passes():
    assert failure_ratio_within_limit(500, 0, 0.0) is True
