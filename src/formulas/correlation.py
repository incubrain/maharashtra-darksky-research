"""
Correlation and regression functions for cross-dataset analysis.

Pure functions, no I/O, fully testable. Used by cross_dataset_steps.py
to compute correlations between VNL radiance metrics and external
dataset metrics (Census, AQI, etc.).
"""

import numpy as np
import pandas as pd
from scipy import stats


def pearson_correlation(x, y):
    """Compute Pearson correlation with confidence interval.

    Parameters
    ----------
    x, y : array-like
        Input arrays (NaN values are dropped pairwise).

    Returns
    -------
    dict
        Keys: r, p_value, n, ci_low, ci_high.
    """
    x, y = _clean_pair(x, y)
    n = len(x)

    if n < 3:
        return {"r": np.nan, "p_value": np.nan, "n": n, "ci_low": np.nan, "ci_high": np.nan}

    r, p = stats.pearsonr(x, y)

    # Fisher z-transform for CI (clamp to avoid arctanh(±1) = ±inf)
    r_clamped = np.clip(r, -0.9999999, 0.9999999)
    z = np.arctanh(r_clamped)
    se = 1.0 / np.sqrt(n - 3)
    ci_low = np.tanh(z - 1.96 * se)
    ci_high = np.tanh(z + 1.96 * se)

    return {"r": r, "p_value": p, "n": n, "ci_low": ci_low, "ci_high": ci_high}


def spearman_correlation(x, y):
    """Compute Spearman rank correlation.

    Parameters
    ----------
    x, y : array-like
        Input arrays (NaN values are dropped pairwise).

    Returns
    -------
    dict
        Keys: rho, p_value, n.
    """
    x, y = _clean_pair(x, y)
    n = len(x)

    if n < 3:
        return {"rho": np.nan, "p_value": np.nan, "n": n}

    rho, p = stats.spearmanr(x, y)
    return {"rho": rho, "p_value": p, "n": n}


def partial_correlation(x, y, covariates):
    """Pearson correlation between x and y after removing linear effect of covariates.

    Used to control for population when correlating radiance with other
    census metrics.

    Parameters
    ----------
    x, y : array-like
        Variables to correlate.
    covariates : array-like or pd.DataFrame
        Covariates to control for. If 1-D, treated as single covariate.

    Returns
    -------
    dict
        Keys: r, p_value, n.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if isinstance(covariates, pd.DataFrame):
        cov = covariates.values.astype(float)
    else:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)

    # Drop rows with any NaN
    mask = ~(np.isnan(x) | np.isnan(y) | np.any(np.isnan(cov), axis=1))
    x, y, cov = x[mask], y[mask], cov[mask]
    n = len(x)

    if n < cov.shape[1] + 3:
        return {"r": np.nan, "p_value": np.nan, "n": n}

    # Residualize x and y against covariates using OLS
    x_resid = _residualize(x, cov)
    y_resid = _residualize(y, cov)

    r, p = stats.pearsonr(x_resid, y_resid)
    return {"r": r, "p_value": p, "n": n}


def ols_regression(y, X, feature_names=None):
    """Simple OLS regression: y ~ X.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    X : array-like or pd.DataFrame
        Independent variables (columns).
    feature_names : list[str], optional
        Names for the features in X.

    Returns
    -------
    dict
        Keys: coefficients, r_squared, p_values, residuals, feature_names.
    """
    y = np.asarray(y, dtype=float)
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = list(X.columns)
        X = X.values.astype(float)
    else:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    # Drop rows with NaN
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y, X = y[mask], X[mask]
    n = len(y)

    if n < X.shape[1] + 2:
        return {
            "coefficients": {},
            "r_squared": np.nan,
            "p_values": {},
            "residuals": np.array([]),
            "feature_names": feature_names,
        }

    # Add intercept
    X_with_const = np.column_stack([np.ones(n), X])
    names_with_const = ["intercept"] + list(feature_names)

    # OLS: beta = (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {
            "coefficients": {},
            "r_squared": np.nan,
            "p_values": {},
            "residuals": np.array([]),
            "feature_names": feature_names,
        }

    y_hat = X_with_const @ beta
    residuals = y - y_hat

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # Standard errors and p-values
    dof = n - X_with_const.shape[1]
    if dof > 0:
        mse = ss_res / dof
        try:
            var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
            se = np.sqrt(np.maximum(var_beta, 0))
            t_stats = beta / np.where(se > 0, se, np.nan)
            p_vals = 2 * stats.t.sf(np.abs(t_stats), dof)
        except np.linalg.LinAlgError:
            p_vals = np.full(len(beta), np.nan)
    else:
        p_vals = np.full(len(beta), np.nan)

    coefficients = dict(zip(names_with_const, beta))
    p_values = dict(zip(names_with_const, p_vals))

    return {
        "coefficients": coefficients,
        "r_squared": r_squared,
        "p_values": p_values,
        "residuals": residuals,
        "feature_names": feature_names,
    }


def compute_correlation_matrix(df, x_cols, y_cols, method="both"):
    """Compute pairwise correlations between two column sets.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing both x_cols and y_cols.
    x_cols : list[str]
        First set of columns (e.g., VNL metrics).
    y_cols : list[str]
        Second set of columns (e.g., Census metrics).
    method : str
        "pearson", "spearman", or "both".

    Returns
    -------
    pd.DataFrame
        Columns: [x_col, y_col, pearson_r, pearson_p, spearman_r, spearman_p, n]
    """
    rows = []

    for x_col in x_cols:
        if x_col not in df.columns:
            continue
        for y_col in y_cols:
            if y_col not in df.columns:
                continue

            row = {"x_col": x_col, "y_col": y_col}

            if method in ("pearson", "both"):
                p = pearson_correlation(df[x_col], df[y_col])
                row["pearson_r"] = p["r"]
                row["pearson_p"] = p["p_value"]
                row["n"] = p["n"]

            if method in ("spearman", "both"):
                s = spearman_correlation(df[x_col], df[y_col])
                row["spearman_r"] = s["rho"]
                row["spearman_p"] = s["p_value"]
                if "n" not in row:
                    row["n"] = s["n"]

            rows.append(row)

    return pd.DataFrame(rows)


def _clean_pair(x, y):
    """Drop NaN pairwise from two arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    return x[mask], y[mask]


def _residualize(v, cov):
    """Residualize v against covariates using OLS."""
    cov_with_const = np.column_stack([np.ones(len(v)), cov])
    beta = np.linalg.lstsq(cov_with_const, v, rcond=None)[0]
    return v - cov_with_const @ beta
