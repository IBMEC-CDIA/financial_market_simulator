"""Tests for the load profiles of the Locust test."""

import pytest

from load_test.load_profiles import (
    LoadSettings,
    settings_from_environment,
    stage_at,
)


def make_settings(
    profile: str,
    max_users: int = 100,
) -> LoadSettings:
    """Build settings with fixed durations for the profile tests.

    Parameters
    ----------
    profile : str
        Profile name to test.
    max_users : int, default 100
        Highest number of simulated users.

    Returns
    -------
    LoadSettings
        Settings with a 60 second stage, a 5 user step and a 600 second ramp.
    """
    return LoadSettings(
        profile=profile,
        max_users=max_users,
        ramp_seconds=600.0,
        step_seconds=60.0,
        step_users=5,
        spike_seconds=60.0,
        max_failure_ratio=0.01,
    )


def test_defaults_are_used_when_no_variable_is_set():
    settings = settings_from_environment({})

    assert settings.profile == "step"
    assert settings.max_users == 50
    assert settings.max_failure_ratio == 0.01


def test_values_are_read_from_the_environment():
    settings = settings_from_environment(
        {
            "LOAD_PROFILE": " Spike ",
            "LOAD_MAX_USERS": "200",
            "LOAD_SPIKE_SECONDS": "30",
        }
    )

    assert settings.profile == "spike"
    assert settings.max_users == 200
    assert settings.spike_seconds == 30.0


def test_unknown_profile_is_refused():
    with pytest.raises(ValueError, match="profile must be one of"):
        settings_from_environment({"LOAD_PROFILE": "soak"})


def test_a_value_that_is_not_a_number_is_refused():
    with pytest.raises(ValueError, match="LOAD_MAX_USERS must be an integer"):
        settings_from_environment({"LOAD_MAX_USERS": "many"})


def test_out_of_range_values_are_refused():
    with pytest.raises(ValueError, match="max_users must be at least 1"):
        settings_from_environment({"LOAD_MAX_USERS": "0"})
    with pytest.raises(ValueError, match="max_failure_ratio"):
        settings_from_environment({"LOAD_MAX_FAILURE_RATIO": "1.5"})


def test_ramp_starts_with_one_user_and_grows_to_the_maximum():
    settings = make_settings("ramp")

    counts = [stage_at(settings, second).users for second in range(0, 600, 60)]

    assert counts[0] == 1
    assert counts == sorted(counts)
    assert stage_at(settings, 599.0).users == 100


def test_ramp_is_over_after_its_duration():
    assert stage_at(make_settings("ramp"), 600.0) is None


def test_step_adds_users_at_each_stage_and_holds_the_last_one():
    settings = make_settings("step", max_users=20)

    assert stage_at(settings, 0.0).users == 5
    assert stage_at(settings, 60.0).users == 10
    assert stage_at(settings, 119.0).users == 10
    assert stage_at(settings, 180.0).users == 20
    assert stage_at(settings, 239.0).users == 20
    assert stage_at(settings, 240.0) is None


def test_step_never_goes_above_the_maximum():
    settings = make_settings("step", max_users=12)

    assert stage_at(settings, 120.0).users == 12


def test_spike_drops_a_burst_and_returns_to_the_baseline():
    settings = make_settings("spike", max_users=100)

    assert stage_at(settings, 0.0).users == 10
    assert stage_at(settings, 60.0).users == 100
    assert stage_at(settings, 60.0).spawn_rate == 100.0
    assert stage_at(settings, 120.0).users == 10
    assert stage_at(settings, 180.0) is None


def test_spike_baseline_is_at_least_one_user():
    settings = make_settings("spike", max_users=3)

    assert stage_at(settings, 0.0).users == 1
