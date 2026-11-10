"""Tests of the browser origins allowed to call the autoencoder API."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from generative_models.serving.api import (
    CORS_ORIGINS_ENVIRONMENT_VARIABLE,
    add_cors_middleware,
    cors_origins_from_environment,
)


ALLOW_ORIGIN = "access-control-allow-origin"


def ping_application(origins: list[str]) -> TestClient:
    """Build a small application with the CORS middleware under test.

    Parameters
    ----------
    origins : list[str]
        Origins passed to `add_cors_middleware`.

    Returns
    -------
    TestClient
        Client of the small application.
    """
    application = FastAPI()

    @application.get("/ping")
    def ping() -> dict[str, str]:
        """Answer a fixed body."""
        return {"status": "ok"}

    add_cors_middleware(application, origins)
    return TestClient(application)


def test_origins_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    """Check that origins are read from the environment, trimmed."""
    monkeypatch.setenv(
        CORS_ORIGINS_ENVIRONMENT_VARIABLE,
        " http://localhost:5500 , http://127.0.0.1:5500 ,",
    )
    assert cors_origins_from_environment() == [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ]


def test_no_origin_is_allowed_when_the_variable_is_not_set(
    monkeypatch: pytest.MonkeyPatch,
):
    """Check that no origin is allowed when the variable is not set."""
    monkeypatch.delenv(CORS_ORIGINS_ENVIRONMENT_VARIABLE, raising=False)
    assert cors_origins_from_environment() == []


def test_a_listed_origin_gets_the_cors_header():
    """Check that a listed origin receives the CORS header."""
    client = ping_application(["http://localhost:5500"])
    response = client.get(
        "/ping",
        headers={"Origin": "http://localhost:5500"},
    )
    assert response.status_code == 200
    assert response.headers[ALLOW_ORIGIN] == "http://localhost:5500"


def test_an_unlisted_origin_gets_no_cors_header():
    """Check that an unlisted origin receives no CORS header."""
    client = ping_application(["http://localhost:5500"])
    response = client.get("/ping", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_the_preflight_request_is_answered_for_listed_origins():
    """Check that the preflight request is answered for listed origins."""
    client = ping_application(["http://localhost:5500"])
    response = client.options(
        "/ping",
        headers={
            "Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers[ALLOW_ORIGIN] == "http://localhost:5500"


def test_no_middleware_is_added_without_origins():
    """Check that no middleware is added when there are no origins."""
    application = FastAPI()
    add_cors_middleware(application, [])
    assert not application.user_middleware
