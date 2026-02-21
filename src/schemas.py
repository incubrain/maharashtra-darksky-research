"""
Pandera DataFrame schemas for pipeline validation gates.

Replaces the manual validate_*() functions in pipeline_runner.py with
declarative, reusable schemas that check both structure AND data quality
(physical value ranges, statistical sanity).

Usage:
    from src.schemas import YearlyRadianceSchema
    YearlyRadianceSchema.validate(df)  # raises pa.errors.SchemaError on failure
"""

import pandera as pa
from pandera import Column, Check, DataFrameSchema


# ── District yearly radiance ────────────────────────────────────────────

YearlyRadianceSchema = DataFrameSchema(
    columns={
        "district": Column(str, nullable=False),
        "year": Column(int, Check.in_range(2012, 2050), nullable=False),
        "mean_radiance": Column(float, Check.in_range(0.0, 500.0), nullable=False),
        "median_radiance": Column(float, Check.in_range(0.0, 500.0), nullable=False),
        "pixel_count": Column(int, Check.greater_than(0), nullable=False, coerce=True),
    },
    # Allow extra columns (min_radiance, max_radiance, std_radiance, etc.)
    strict=False,
    coerce=False,
    name="YearlyRadianceSchema",
)


# ── District trends ─────────────────────────────────────────────────────

TrendsSchema = DataFrameSchema(
    columns={
        "district": Column(str, nullable=False),
        "annual_pct_change": Column(float, Check.in_range(-50.0, 100.0), nullable=True),
        "r_squared": Column(float, Check.in_range(0.0, 1.0), nullable=True),
    },
    strict=False,
    coerce=False,
    name="TrendsSchema",
)


# ── Site yearly radiance ────────────────────────────────────────────────

SiteYearlySchema = DataFrameSchema(
    columns={
        "name": Column(str, nullable=False),
        "year": Column(int, Check.in_range(2012, 2050), nullable=False),
        "median_radiance": Column(float, Check.in_range(0.0, 500.0), nullable=False),
        "type": Column(str, Check.isin(["city", "site"]), nullable=False),
    },
    strict=False,
    coerce=False,
    name="SiteYearlySchema",
)


# ── Site trends ─────────────────────────────────────────────────────────

SiteTrendsSchema = DataFrameSchema(
    columns={
        "name": Column(str, nullable=False),
        "annual_pct_change": Column(float, Check.in_range(-50.0, 100.0), nullable=True),
        "type": Column(str, Check.isin(["city", "site"]), nullable=False),
    },
    strict=False,
    coerce=False,
    name="SiteTrendsSchema",
)


# ── Stability metrics ──────────────────────────────────────────────────

StabilitySchema = DataFrameSchema(
    columns={
        "district": Column(str, nullable=False),
        "cv": Column(float, Check.greater_than_or_equal_to(0.0), nullable=True),
    },
    strict=False,
    coerce=False,
    name="StabilitySchema",
)


# ── Convenience validation function ─────────────────────────────────────

def validate_schema(df, schema, step_name, strict=False):
    """Validate a DataFrame against a Pandera schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    schema : pa.DataFrameSchema
        Schema to validate against.
    step_name : str
        Pipeline step name for error messages.
    strict : bool
        If True, raise on failure. If False, return warnings list.

    Returns
    -------
    list[str]
        Validation warning messages (empty if all pass).

    Raises
    ------
    pa.errors.SchemaError
        Only if strict=True and validation fails.
    """
    warnings_list = []

    if df is None:
        msg = f"[{step_name}] DataFrame is None"
        if strict:
            raise ValueError(msg)
        return [msg]

    if len(df) == 0:
        msg = f"[{step_name}] DataFrame is empty (0 rows)"
        if strict:
            raise ValueError(msg)
        return [msg]

    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        for failure in exc.failure_cases.itertuples():
            msg = (
                f"[{step_name}] Schema violation: "
                f"column='{failure.column}' check='{failure.check}' "
                f"failure_case={failure.failure_case}"
            )
            warnings_list.append(msg)

        if strict:
            raise ValueError(
                f"[{step_name}] Schema validation failed with "
                f"{len(warnings_list)} errors"
            ) from exc

    return warnings_list
