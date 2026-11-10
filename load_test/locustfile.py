"""Locust test for the autoencoder API.

Each simulated user reads the models loaded by the API with
`GET /models/loaded` and then sends requests for those models: the
suggestion of the day, the suggestion history of a window, the market
prediction and the health check. The load shape follows the profile set
in `LOAD_PROFILE` (`spike`, `ramp` or `step`). See `load_test/README.md`.
"""

import os
import random
from datetime import date

from locust import HttpUser, LoadTestShape, between, events, task
from locust.env import Environment
from locust.exception import StopUser

from load_profiles import settings_from_environment, stage_at
from model_requests import history_period, tickers_from_payload
from thresholds import failure_ratio_within_limit

SETTINGS = settings_from_environment(os.environ)
HISTORY_WINDOW_DAYS = 90


class ModelUser(HttpUser):
    """Simulated client that sends requests for the models of the API.

    Attributes
    ----------
    tickers : list[str]
        Tickers of the models loaded by the API, read in `on_start`.
    """

    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        """Read the models loaded by the API before the user starts.

        Raises
        ------
        StopUser
            If the API reports no model, so the user has nothing to call.
        """
        self.tickers: list[str] = []
        with self.client.get(
            "/models/loaded",
            name="/models/loaded",
            catch_response=True,
        ) as response:
            if response.ok:
                try:
                    self.tickers = tickers_from_payload(response.json())
                except ValueError as error:
                    response.failure(str(error))
        if not self.tickers:
            raise StopUser()

    @task(6)
    def get_signal(self) -> None:
        """Request the suggestion of the day of a random model."""
        ticker = random.choice(self.tickers)
        self.client.get(
            "/signals",
            params={"ticker": ticker},
            name="/signals?ticker=[ticker]",
        )

    @task(3)
    def get_signal_history(self) -> None:
        """Request the suggestions of a 90 day window of a random model."""
        ticker = random.choice(self.tickers)
        start_date, end_date = history_period(
            date.today(),
            HISTORY_WINDOW_DAYS,
        )
        self.client.get(
            "/signals/history",
            params={
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
            },
            name="/signals/history?ticker=[ticker]",
        )

    @task(1)
    def get_market_prediction(self) -> None:
        """Request the outlier prediction of a random model on market data.

        The API downloads the prices of the ticker when it answers, so this
        request also depends on the price source.
        """
        ticker = random.choice(self.tickers)
        self.client.get(
            "/predictions/market",
            params={"ticker": ticker},
            name="/predictions/market?ticker=[ticker]",
        )

    @task(1)
    def get_health(self) -> None:
        """Request the health check of the API."""
        self.client.get("/health", name="/health")


class ProfileLoadShape(LoadTestShape):
    """Load shape that follows the profile chosen in `LOAD_PROFILE`.

    The number of users and the spawn rate are read every second, and the
    test stops when the profile ends.
    """

    def tick(self) -> tuple[int, float] | None:
        """Return the users and the spawn rate for the current second.

        Returns
        -------
        tuple[int, float] or None
            Users and spawn rate, or `None` when the test must stop.
        """
        stage = stage_at(SETTINGS, self.get_run_time())
        if stage is None:
            return None
        return stage.users, stage.spawn_rate


@events.test_stop.add_listener
def check_failure_ratio(environment: Environment, **_kwargs: object) -> None:
    """Fail the process when the test breaks the failure limit.

    Parameters
    ----------
    environment : Environment
        Locust environment that holds the statistics of the test.
    **_kwargs : object
        Other arguments sent by Locust, not used here.
    """
    total = environment.stats.total
    passed = failure_ratio_within_limit(
        total.num_requests,
        total.num_failures,
        SETTINGS.max_failure_ratio,
    )
    print(
        f"Profile {SETTINGS.profile}: {total.num_requests} requests, "
        f"{total.num_failures} failures, p95 "
        f"{total.get_response_time_percentile(0.95)} ms."
    )
    if not passed:
        print(
            "FAILED: the failure ratio is above "
            f"{SETTINGS.max_failure_ratio:.2%} or no request was sent."
        )
        environment.process_exit_code = 1
