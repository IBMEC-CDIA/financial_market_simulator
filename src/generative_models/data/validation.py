"""Quality checks for collected price series, using Great Expectations."""

import great_expectations as gx
import great_expectations.expectations as gxe
import numpy as np
import pandas as pd

MINIMUM_ROW_COUNT = 252
MAXIMUM_ABS_LOG_RETURN = 0.5
DATA_SOURCE_NAME = "market_prices"
ASSET_NAME = "daily_prices"
BATCH_DEFINITION_NAME = "whole_price_table"
SUITE_NAME = "price_quality_suite"
VALIDATION_NAME = "price_quality_validation"


class PriceValidationError(ValueError):
    """Raised when a price series does not pass the quality checks.

    The message lists every failed expectation, so the problem can be
    found in the logs of the pipeline.
    """


def build_price_expectations() -> list[gxe.Expectation]:
    """Create the quality expectations that every price series must meet.

    The checks cover the number of rows, the closing prices (no missing
    values, always positive), the dates (unique and strictly increasing)
    and the daily log-returns (no move larger than
    `MAXIMUM_ABS_LOG_RETURN`, which would indicate a data error).

    Returns
    -------
    list[gxe.Expectation]
        Expectations to add to the suite, in the order they are checked.

    Example
    -------
    >>> expectations = build_price_expectations()
    >>> len(expectations)
    6
    """
    return [
        gxe.ExpectTableRowCountToBeBetween(min_value=MINIMUM_ROW_COUNT),
        gxe.ExpectColumnValuesToNotBeNull(column="close_price"),
        gxe.ExpectColumnValuesToBeBetween(
            column="close_price",
            min_value=0,
            strict_min=True,
        ),
        gxe.ExpectColumnValuesToBeUnique(column="date"),
        gxe.ExpectColumnValuesToBeIncreasing(column="date", strictly=True),
        gxe.ExpectColumnValuesToBeBetween(
            column="log_return",
            min_value=-MAXIMUM_ABS_LOG_RETURN,
            max_value=MAXIMUM_ABS_LOG_RETURN,
        ),
    ]


def build_price_frame(price_data: pd.DataFrame) -> pd.DataFrame:
    """Turn a price series into the table that the expectations check.

    Parameters
    ----------
    price_data : pd.DataFrame
        Data frame indexed by date, with a `"close_price"` column.

    Returns
    -------
    pd.DataFrame
        Table with the `date`, `close_price` and `log_return` columns. The
        first `log_return` is `NaN`, because it has no previous price.

    Example
    -------
    >>> frame = build_price_frame(price_data)
    >>> list(frame.columns)
    ['date', 'close_price', 'log_return']
    """
    frame = price_data.rename_axis("date").reset_index()
    frame["log_return"] = np.log(frame["close_price"]).diff()

    return frame


def _describe_failed_result(result) -> str:
    """Return a readable line for an expectation result that failed.

    Parameters
    ----------
    result : ExpectationValidationResult
        Result of one expectation.

    Returns
    -------
    str
        Expectation type and the column it checked, when there is one.
    """
    configuration = result.expectation_config
    column = configuration.kwargs.get("column", "table")

    return f"{configuration.type} on '{column}'"


def validate_price_dataset(price_data: pd.DataFrame) -> None:
    """Check a price series with Great Expectations and fail if it is invalid.

    The check runs in an ephemeral context, so nothing is written to disk.

    Parameters
    ----------
    price_data : pd.DataFrame
        Data frame indexed by date, with a `"close_price"` column.

    Raises
    ------
    PriceValidationError
        If at least one expectation fails. The message lists them.

    Example
    -------
    >>> validate_price_dataset(price_data)  # returns None when valid
    """
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(DATA_SOURCE_NAME)
    asset = data_source.add_dataframe_asset(ASSET_NAME)
    batch_definition = asset.add_batch_definition_whole_dataframe(
        BATCH_DEFINITION_NAME,
    )
    suite = context.suites.add(gx.ExpectationSuite(name=SUITE_NAME))
    for expectation in build_price_expectations():
        suite.add_expectation(expectation)
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name=VALIDATION_NAME,
            data=batch_definition,
            suite=suite,
        )
    )
    validation_result = validation_definition.run(
        batch_parameters={"dataframe": build_price_frame(price_data)},
    )
    if not validation_result.success:
        failed_checks = [
            _describe_failed_result(result)
            for result in validation_result.results
            if not result.success
        ]
        raise PriceValidationError(
            "The price data failed the quality checks: "
            + "; ".join(failed_checks)
        )
