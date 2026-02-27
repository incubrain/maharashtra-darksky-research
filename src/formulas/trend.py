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

5. **Airglow confound (L1):** At very dark sites, VIIRS radiance includes
   natural airglow emissions that vary with the ~11-year solar cycle.
   This can bias dark-site trend estimates by up to 0.1-0.2 nW/cm²/sr.
   Ref: Levin, N. et al. (2020). Remote Sensing of Night Lights: A
   Review. Remote Sensing of Environment, 237, 111443.

6. **VIIRS product version transition (M7):** VNL v2.1 (2012-2013) uses
   vcmcfg (no stray-light correction); v2.2 (2014+) uses vcmslcfg
   (stray-light corrected). This introduces a systematic radiometric
   shift that can confound trend detection if not modelled.
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


def block_bootstrap_ci(
    years, radiance, block_size=3, n_bootstrap=None, ci_level=None,
    seed=None, log_epsilon=None,
):
    """Compute bootstrap CIs using block resampling for temporal autocorrelation.

    Standard i.i.d. bootstrap underestimates CI width when observations are
    temporally correlated (finding E5). Block bootstrap preserves local
    autocorrelation structure by resampling contiguous blocks.

    Ref: Elvidge, C.D. et al. (2021). Remote Sensing, 13(5), 922 — notes
    temporal autocorrelation in annual composites.
    Ref: Kunsch, H.R. (1989). The Jackknife and the Bootstrap for General
    Stationary Observations. Annals of Statistics, 17(3), 1217-1241.

    Parameters
    ----------
    years : array-like
        Year values.
    radiance : array-like
        Median radiance values in nW/cm²/sr.
    block_size : int
        Size of contiguous blocks to resample (default 3 years).
    n_bootstrap : int, optional
        Number of bootstrap resamples. Default: config.BOOTSTRAP_RESAMPLES.
    ci_level : tuple, optional
        CI percentiles. Default: config.BOOTSTRAP_CI_LEVEL.
    seed : int, optional
        Random seed. Default: config.BOOTSTRAP_SEED.
    log_epsilon : float, optional
        Small constant for log(0). Default: config.LOG_EPSILON.

    Returns
    -------
    dict
        Keys: ci_low_block, ci_high_block, n_valid_resamples.
    """
    if n_bootstrap is None:
        n_bootstrap = config.BOOTSTRAP_RESAMPLES
    if ci_level is None:
        ci_level = config.BOOTSTRAP_CI_LEVEL
    if seed is None:
        seed = config.BOOTSTRAP_SEED
    if log_epsilon is None:
        log_epsilon = config.LOG_EPSILON

    years = np.asarray(years, dtype=float)
    radiance = np.asarray(radiance, dtype=float)
    radiance = np.clip(radiance, 0, None)
    n = len(years)

    if n < 4:
        return {"ci_low_block": np.nan, "ci_high_block": np.nan,
                "n_valid_resamples": 0}

    log_rad = np.log(radiance + log_epsilon)
    n_blocks = max(1, n - block_size + 1)

    rng = np.random.default_rng(seed)
    boot_pcts = []

    for _ in range(n_bootstrap):
        # Sample block starting indices with replacement
        starts = rng.choice(n_blocks, size=(n // block_size) + 1, replace=True)
        idx = []
        for s in starts:
            idx.extend(range(s, min(s + block_size, n)))
            if len(idx) >= n:
                break
        idx = np.array(idx[:n])

        X_b = sm.add_constant(years[idx])
        y_b = log_rad[idx]
        try:
            m_b = sm.OLS(y_b, X_b).fit()
            boot_pcts.append((np.exp(m_b.params[1]) - 1) * 100)
        except (np.linalg.LinAlgError, ValueError, IndexError):
            continue

    boot_pcts = np.array(boot_pcts)
    ci_lo, ci_hi = ci_level
    return {
        "ci_low_block": np.percentile(boot_pcts, ci_lo) if len(boot_pcts) > 0 else np.nan,
        "ci_high_block": np.percentile(boot_pcts, ci_hi) if len(boot_pcts) > 0 else np.nan,
        "n_valid_resamples": len(boot_pcts),
    }


def fit_trend_with_version_covariate(
    years, radiance, version_boundary=2014, log_epsilon=None, min_years=None,
):
    """Fit log-linear trend with a VIIRS product version dummy variable.

    Adds a binary covariate for the VNL v2.1 → v2.2 transition (finding M7).
    VNL v2.1 (2012-2013) uses vcmcfg; v2.2 (2014+) uses vcmslcfg with
    stray-light correction, introducing a systematic radiometric shift.
    Including this covariate absorbs the version-change step, yielding a
    cleaner estimate of the underlying ALAN trend.

    Parameters
    ----------
    years : array-like
        Year values.
    radiance : array-like
        Median radiance values.
    version_boundary : int
        First year of v2.2 (default 2014).
    log_epsilon : float, optional
        Default: config.LOG_EPSILON.
    min_years : int, optional
        Default: config.MIN_YEARS_FOR_TREND.

    Returns
    -------
    dict
        Keys: annual_pct_change, annual_pct_change_adjusted, version_effect,
        r_squared, r_squared_adjusted, p_value_trend, p_value_version.
    """
    if log_epsilon is None:
        log_epsilon = config.LOG_EPSILON
    if min_years is None:
        min_years = config.MIN_YEARS_FOR_TREND

    years = np.asarray(years, dtype=float)
    radiance = np.asarray(radiance, dtype=float)
    radiance = np.clip(radiance, 0, None)

    nan_result = {
        "annual_pct_change": np.nan,
        "annual_pct_change_adjusted": np.nan,
        "version_effect": np.nan,
        "r_squared": np.nan,
        "r_squared_adjusted": np.nan,
        "p_value_trend": np.nan,
        "p_value_version": np.nan,
    }

    if len(years) < max(min_years, 4):
        return nan_result

    log_rad = np.log(radiance + log_epsilon)

    # Simple model (no version covariate)
    X_simple = sm.add_constant(years)
    model_simple = sm.OLS(log_rad, X_simple).fit()

    # Model with version dummy
    version_dummy = (years >= version_boundary).astype(float)
    X_version = np.column_stack([np.ones(len(years)), years, version_dummy])
    model_version = sm.OLS(log_rad, X_version).fit()

    beta_simple = model_simple.params[1]
    beta_adjusted = model_version.params[1]
    version_effect = model_version.params[2]

    return {
        "annual_pct_change": (np.exp(beta_simple) - 1) * 100,
        "annual_pct_change_adjusted": (np.exp(beta_adjusted) - 1) * 100,
        "version_effect": version_effect,
        "r_squared": model_simple.rsquared,
        "r_squared_adjusted": model_version.rsquared,
        "p_value_trend": model_version.pvalues[1] if len(model_version.pvalues) > 1 else np.nan,
        "p_value_version": model_version.pvalues[2] if len(model_version.pvalues) > 2 else np.nan,
    }
