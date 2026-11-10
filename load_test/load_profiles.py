"""Load profiles that set how many simulated users run at each moment.

Three profiles are available. The spike profile drops a burst of users on
the API and removes it again. The ramp profile raises the number of users
linearly. The step profile adds a fixed number of users every stage, so the
load grows progressively until the maximum is reached.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

PROFILE_RAMP = "ramp"
PROFILE_STEP = "step"
PROFILE_SPIKE = "spike"
PROFILE_NAMES = (PROFILE_RAMP, PROFILE_STEP, PROFILE_SPIKE)
SPIKE_PHASE_COUNT = 3
BASELINE_DIVISOR = 10
DEFAULT_MAX_USERS = 50
DEFAULT_RAMP_SECONDS = 600.0
DEFAULT_STEP_SECONDS = 60.0
DEFAULT_STEP_USERS = 5
DEFAULT_SPIKE_SECONDS = 60.0
DEFAULT_MAX_FAILURE_RATIO = 0.01


@dataclass(frozen=True)
class LoadSettings:
    """Parameters of a load test, read from the environment.

    Parameters
    ----------
    profile : str
        One of `ramp`, `step` or `spike`.
    max_users : int
        Highest number of simulated users.
    ramp_seconds : float
        Duration of the ramp profile, from one user to `max_users`.
    step_seconds : float
        Length of each stage of the step profile.
    step_users : int
        Users added at each stage of the step profile.
    spike_seconds : float
        Length of each of the three phases of the spike profile.
    max_failure_ratio : float
        Highest share of failed requests that still passes the test.

    Raises
    ------
    ValueError
        If a value is out of its valid range.
    """

    profile: str
    max_users: int
    ramp_seconds: float
    step_seconds: float
    step_users: int
    spike_seconds: float
    max_failure_ratio: float

    def __post_init__(self) -> None:
        if self.profile not in PROFILE_NAMES:
            raise ValueError(
                f"profile must be one of {', '.join(PROFILE_NAMES)}."
            )
        if self.max_users < 1:
            raise ValueError("max_users must be at least 1.")
        if self.step_users < 1:
            raise ValueError("step_users must be at least 1.")
        for name, value in (
            ("ramp_seconds", self.ramp_seconds),
            ("step_seconds", self.step_seconds),
            ("spike_seconds", self.spike_seconds),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")
        if not 0 <= self.max_failure_ratio <= 1:
            raise ValueError("max_failure_ratio must be between 0 and 1.")


@dataclass(frozen=True)
class LoadStage:
    """Number of users that should be running at one moment of the test.

    Parameters
    ----------
    users : int
        Users that should be running.
    spawn_rate : float
        Users started per second while the number of users grows.
    """

    users: int
    spawn_rate: float


def settings_from_environment(environment: Mapping[str, str]) -> LoadSettings:
    """Read the load test settings from environment variables.

    The variables are `LOAD_PROFILE`, `LOAD_MAX_USERS`,
    `LOAD_RAMP_SECONDS`, `LOAD_STEP_SECONDS`, `LOAD_STEP_USERS`,
    `LOAD_SPIKE_SECONDS` and `LOAD_MAX_FAILURE_RATIO`. Missing variables
    take their default values.

    Parameters
    ----------
    environment : Mapping[str, str]
        Environment variables, usually `os.environ`.

    Returns
    -------
    LoadSettings
        The validated settings of the load test.

    Raises
    ------
    ValueError
        If a value is not a number or is out of its valid range.

    Example
    -------
    >>> settings = settings_from_environment({"LOAD_PROFILE": "spike"})
    >>> settings.profile
    'spike'
    """
    return LoadSettings(
        profile=environment.get("LOAD_PROFILE", PROFILE_STEP)
        .strip()
        .lower(),
        max_users=_read_int(
            environment, "LOAD_MAX_USERS", DEFAULT_MAX_USERS
        ),
        ramp_seconds=_read_float(
            environment, "LOAD_RAMP_SECONDS", DEFAULT_RAMP_SECONDS
        ),
        step_seconds=_read_float(
            environment, "LOAD_STEP_SECONDS", DEFAULT_STEP_SECONDS
        ),
        step_users=_read_int(
            environment, "LOAD_STEP_USERS", DEFAULT_STEP_USERS
        ),
        spike_seconds=_read_float(
            environment, "LOAD_SPIKE_SECONDS", DEFAULT_SPIKE_SECONDS
        ),
        max_failure_ratio=_read_float(
            environment,
            "LOAD_MAX_FAILURE_RATIO",
            DEFAULT_MAX_FAILURE_RATIO,
        ),
    )


def _read_int(
    environment: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    """Read an integer environment variable.

    Parameters
    ----------
    environment : Mapping[str, str]
        Environment variables to read from.
    name : str
        Name of the variable.
    default : int
        Value used when the variable is not set.

    Returns
    -------
    int
        The integer value of the variable, or `default`.

    Raises
    ------
    ValueError
        If the variable is set but is not an integer.
    """
    raw_value = environment.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


def _read_float(
    environment: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    """Read a decimal environment variable.

    Parameters
    ----------
    environment : Mapping[str, str]
        Environment variables to read from.
    name : str
        Name of the variable.
    default : float
        Value used when the variable is not set.

    Returns
    -------
    float
        The decimal value of the variable, or `default`.

    Raises
    ------
    ValueError
        If the variable is set but is not a number.
    """
    raw_value = environment.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number.") from error


def stage_at(
    settings: LoadSettings,
    elapsed_seconds: float,
) -> LoadStage | None:
    """Return the load that should be running after some seconds.

    Parameters
    ----------
    settings : LoadSettings
        Settings of the load test, including the profile.
    elapsed_seconds : float
        Seconds since the test started.

    Returns
    -------
    LoadStage or None
        The users and spawn rate for this moment, or `None` when the
        profile is over and the test must stop.

    Example
    -------
    >>> settings = settings_from_environment(
    ...     {"LOAD_PROFILE": "step", "LOAD_MAX_USERS": "10"}
    ... )
    >>> stage_at(settings, 0.0).users
    5
    """
    if settings.profile == PROFILE_RAMP:
        return _ramp_stage(settings, elapsed_seconds)
    if settings.profile == PROFILE_STEP:
        return _step_stage(settings, elapsed_seconds)
    return _spike_stage(settings, elapsed_seconds)


def _ramp_stage(
    settings: LoadSettings,
    elapsed_seconds: float,
) -> LoadStage | None:
    """Return the linear ramp load at a moment.

    Parameters
    ----------
    settings : LoadSettings
        Settings with the ramp duration and the maximum users.
    elapsed_seconds : float
        Seconds since the test started.

    Returns
    -------
    LoadStage or None
        The users growing from one to the maximum, or `None` after the
        ramp duration.
    """
    if elapsed_seconds >= settings.ramp_seconds:
        return None
    growth = (settings.max_users - 1) * elapsed_seconds
    users = 1 + round(growth / settings.ramp_seconds)
    spawn_rate = max(1.0, (settings.max_users - 1) / settings.ramp_seconds)
    return LoadStage(users=users, spawn_rate=spawn_rate)


def _step_stage(
    settings: LoadSettings,
    elapsed_seconds: float,
) -> LoadStage | None:
    """Return the progressive step load at a moment.

    Parameters
    ----------
    settings : LoadSettings
        Settings with the stage length, the users added per stage and the
        maximum users.
    elapsed_seconds : float
        Seconds since the test started.

    Returns
    -------
    LoadStage or None
        The users of the current stage, or `None` after the last stage.
    """
    stage_count = math.ceil(settings.max_users / settings.step_users)
    stage_index = int(elapsed_seconds // settings.step_seconds)
    if stage_index >= stage_count:
        return None
    users = min(
        settings.max_users,
        settings.step_users * (stage_index + 1),
    )
    return LoadStage(users=users, spawn_rate=float(settings.step_users))


def _spike_stage(
    settings: LoadSettings,
    elapsed_seconds: float,
) -> LoadStage | None:
    """Return the spike load at a moment.

    The three phases are a baseline, a burst to the maximum users, and the
    baseline again.

    Parameters
    ----------
    settings : LoadSettings
        Settings with the phase length and the maximum users.
    elapsed_seconds : float
        Seconds since the test started.

    Returns
    -------
    LoadStage or None
        The users of the current phase, or `None` after the last phase.
    """
    phase_index = int(elapsed_seconds // settings.spike_seconds)
    if phase_index >= SPIKE_PHASE_COUNT:
        return None
    baseline = max(1, settings.max_users // BASELINE_DIVISOR)
    if phase_index == 1:
        return LoadStage(
            users=settings.max_users,
            spawn_rate=float(settings.max_users),
        )
    return LoadStage(users=baseline, spawn_rate=float(baseline))
