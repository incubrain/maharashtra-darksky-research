"""
Log-linear trend fitting with bootstrap confidence intervals.

Shared implementation used by both district and site pipelines.
All functions are pure (no I/O, no side effects).

METHODOLOGY:
Log-linear regression log(radiance + ε) ~ year models approximately
exponential ALAN growth in developing regions. The slope β converts to
annual % change via (exp(β) - 1) × 100.
Citation: Elvidge et al. (2021), Section 3; Li et al. (2020).

IMPORTANT CAVEATS — VIIRS radiance trends have known limitations:
1. **LED spectral bias (F4, KY3):** VIIRS DNB (500-900 nm) is blind to
   blue LED emissions. As India transitions from HPS to LED under the UJALA
   programme (2015+), VIIRS underestimates true radiance by ~30%.
   Ref: Kyba et al. (2017), Science Advances, 3(11), e1701528.
2. **Ground-truth divergence (K1):** Ground-based measurements show sky
   brightness increasing ~9.6%/yr vs VIIRS-detected ~2%/yr — a ~5x gap.
   Ref: Kyba et al. (2023), Science, 379(6629), 265-268.
3. **Electrification confound (M1):** Maharashtra VIIRS trends for
   2012-2024 conflate ALAN growth with electricity reliability improvements
   from DDUGJY (2014), UJALA (2015), and Saubhagya (2017) programmes.
   Ref: Min et al. (2017), Papers in Regional Science, 96(4), 811-832.
4. **Bortle drift (K2):** Cumulative LED spectral bias causes VIIRS-derived
   Bortle classifications to drift 1-2 classes over the study period.

These caveats apply to ALL trend outputs and Bortle classifications
produced by this pipeline.
"""

import numpy as np
import statsmodels.api as sm

from src import config


def fit_log_linear_trend(
    years,
    radiance,
    log_epsilon=None,
    n_bootstrap=None,
    ci_level=None,
    seed=None,
    min_years=None,
):
    """Fit log-linear OLS trend with bootstrap confidence intervals.

    Parameters
    ----------
    years : array-like
        Year values (e.g. [2012, 2013, ..., 2024]).
    radiance : array-like
        Median radiance values in nW/cm²/sr.
    log_epsilon : float, optional
        Small constant to prevent log(0). Default: config.LOG_EPSILON (1e-6).
    n_bootstrap : int, optional
        Number of bootstrap resamples. Default: config.BOOTSTRAP_RESAMPLES (1000).
    ci_level : tuple, optional
        (low_percentile, high_percentile). Default: config.BOOTSTRAP_CI_LEVEL (2.5, 97.5).
    seed : int, optional
        Random seed. Default: config.BOOTSTRAP_SEED (42).
    min_years : int, optional
        Minimum data points required. Default: config.MIN_YEARS_FOR_TREND (2).

    Returns
    -------
    dict
        Keys: annual_pct_change, ci_low, ci_high, r_squared, p_value,
        n_years, beta, residuals.
    """
    if log_epsilon is None:
        log_epsilon = config.LOG_EPSILON
    if n_bootstrap is None:
        n_bootstrap = config.BOOTSTRAP_RESAMPLES
    if ci_level is None:
        ci_level = config.BOOTSTRAP_CI_LEVEL
    if seed is None:
        seed = config.BOOTSTRAP_SEED
    if min_years is None:
        min_years = config.MIN_YEARS_FOR_TREND

    years = np.asarray(years, dtype=float)
    radiance = np.asarray(radiance, dtype=float)

    # Clip negative radiance to zero before log transform.
    # VIIRS DNB can produce small negatives from background subtraction.
    # log(negative + epsilon) → NaN, which silently corrupts the OLS fit.
    # Ref: Elvidge et al. (2017) — negative radiances are sensor artefacts.
    radiance = np.clip(radiance, 0, None)

    nan_result = {
        "annual_pct_change": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "r_squared": np.nan,
        "p_value": np.nan,
        "n_years": len(years),
        "beta": np.nan,
        "residuals": np.array([]),
    }

    if len(years) < min_years:
        return nan_result

    log_rad = np.log(radiance + log_epsilon)

    X = sm.add_constant(years)
    model = sm.OLS(log_rad, X).fit()
    beta = model.params[1]
    annual_pct = (np.exp(beta) - 1) * 100

    # Bootstrap CI
    boot_pcts = []
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(years), size=len(years), replace=True)
        X_b = sm.add_constant(years[idx])
        y_b = log_rad[idx]
        try:
            m_b = sm.OLS(y_b, X_b).fit()
            boot_pcts.append((np.exp(m_b.params[1]) - 1) * 100)
        except (np.linalg.LinAlgError, ValueError, IndexError):
            continue

    boot_pcts = np.array(boot_pcts)
    ci_lo, ci_hi = ci_level
    ci_low = np.percentile(boot_pcts, ci_lo) if len(boot_pcts) > 0 else np.nan
    ci_high = np.percentile(boot_pcts, ci_hi) if len(boot_pcts) > 0 else np.nan

    return {
        "annual_pct_change": annual_pct,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "r_squared": model.rsquared,
        "p_value": model.pvalues[1] if len(model.pvalues) > 1 else np.nan,
        "n_years": len(years),
        "beta": beta,
        "residuals": model.resid,
    }
