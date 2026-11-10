"""Pass or fail rules applied when a load test ends."""


def failure_ratio_within_limit(
    total_requests: int,
    failed_requests: int,
    limit: float,
) -> bool:
    """Tell whether the share of failed requests is within the limit.

    A test that sent no request does not pass, because it proves nothing
    about the API.

    Parameters
    ----------
    total_requests : int
        Number of requests sent during the test.
    failed_requests : int
        Number of requests that failed.
    limit : float
        Highest share of failed requests allowed, between 0 and 1.

    Returns
    -------
    bool
        `True` when at least one request was sent and the share of failures
        is less than or equal to `limit`.

    Example
    -------
    >>> failure_ratio_within_limit(1000, 10, 0.01)
    True
    """
    if total_requests <= 0:
        return False
    return failed_requests / total_requests <= limit
