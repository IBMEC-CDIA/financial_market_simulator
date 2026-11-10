"""Tests for the pass or fail rule of the Locust test."""

from load_test.thresholds import failure_ratio_within_limit


def test_a_test_without_requests_does_not_pass():
    """Verify that a test without requests does not pass."""
    assert failure_ratio_within_limit(0, 0, 0.01) is False


def test_failures_at_the_limit_pass():
    """Verify that failures at the limit pass."""
    assert failure_ratio_within_limit(1000, 10, 0.01) is True


def test_failures_above_the_limit_fail():
    """Verify that failures above the limit fail."""
    assert failure_ratio_within_limit(1000, 11, 0.01) is False


def test_a_test_without_failures_passes():
    """Verify that a test without failures passes."""
    assert failure_ratio_within_limit(500, 0, 0.0) is True
