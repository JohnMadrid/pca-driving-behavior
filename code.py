import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import os
import glob
import math
from scipy.stats import chi2, norm
import re


def compute_event_pc_bo_ao_stats(
        df: pd.DataFrame,
        event_col: str = 'event',
        pc_col: str = 'PC',
        value_col: str = 'value',
        timepoint_col: str = 'Timepoint',
        relative_time_col: Optional[str] = 'relative_time'
) -> pd.DataFrame:
    """
    Aggregate BO/AO statistics for each event × PC pair:
      means, SDs, counts, SEMs, and standardized differences.
    Works directly on the long-format variance dataframe.
    """

    data = df.copy()
    # If a relative time column exists, drop onset samples at exactly 0
    if relative_time_col is not None and relative_time_col in data.columns:
        mask0 = np.isclose(data[relative_time_col], 0.0)
        data = data.loc[~mask0].copy()

    required_cols = {event_col, pc_col, timepoint_col, value_col}
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    grouped = (
        data
        .groupby([event_col, pc_col, timepoint_col], observed=True)[value_col]
        .agg(count='count', mean='mean', std='std')
        .reset_index()
    )

    wide = grouped.pivot(index=[event_col, pc_col], columns=timepoint_col)
    wide.columns = [f"{stat}_{tp}" for stat, tp in wide.columns]
    wide = wide.reset_index()

    # Ensure columns exist
    for col in ['count_BO', 'count_AO', 'std_BO', 'std_AO', 'mean_BO', 'mean_AO']:
        if col not in wide.columns:
            wide[col] = np.nan

    n_bo = wide['count_BO'].astype(float)
    n_ao = wide['count_AO'].astype(float)
    wide['sem_BO'] = np.where(n_bo > 1.0, wide['std_BO'] / np.sqrt(n_bo), 0.0)
    wide['sem_AO'] = np.where(n_ao > 1.0, wide['std_AO'] / np.sqrt(n_ao), 0.0)

    wide['delta'] = wide['mean_AO'] - wide['mean_BO']
    denom = np.sqrt(np.square(wide['sem_AO']) + np.square(wide['sem_BO']))
    eps = np.finfo(float).eps
    denom = denom.replace(0.0, np.sqrt(eps))
    wide['denom'] = denom
    wide['z'] = wide['delta'] / wide['denom']
    wide['z2'] = np.square(wide['z'])

    cols = [
        event_col, pc_col,
        'mean_BO', 'mean_AO', 'std_BO', 'std_AO', 'count_BO', 'count_AO',
        'sem_BO', 'sem_AO', 'delta', 'denom', 'z', 'z2'
    ]
    for c in cols:
        if c not in wide.columns:
            wide[c] = np.nan
    return wide[cols]


def variance_weighted_chi_square_global(
        df: pd.DataFrame,
        event_col: str = 'event',
        pc_col: str = 'PC',
        value_col: str = 'value',
        timepoint_col: str = 'Timepoint',
        relative_time_col: Optional[str] = 'relative_time',
        min_n: int = 1
) -> Dict[str, object]:
    """
    Single omnibus chi-square across all event×PC pairs using the
    inverse-variance standardized differences between AO and BO variances.

    Returns a dictionary with a formatted p-value string and Δ̂ ± CI.
    """

    per_pair = compute_event_pc_bo_ao_stats(
        df=df,
        event_col=event_col,
        pc_col=pc_col,
        value_col=value_col,
        timepoint_col=timepoint_col,
        relative_time_col=relative_time_col
    )

    # Optionally drop pairs with too few samples
    mask_min = (per_pair['count_BO'] >= min_n) & (per_pair['count_AO'] >= min_n)
    valid = per_pair.loc[mask_min].dropna(subset=['denom', 'delta'])
    if valid.empty:
        return {
            'num_pairs': 0,
            'chi2': np.nan,
            'df': 0,
            'p_value': 'NA',
            'weighted_mean_diff': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'per_pair': per_pair
        }

    z2_sum = float(valid['z2'].sum())
    df_pairs = int(valid.shape[0])

    # Robust p-value computation with fallbacks
    log_p = chi2.logsf(z2_sum, df_pairs)
    p_source = 'scipy-chi2'
    if not np.isfinite(log_p):
        try:
            import mpmath as mp
            mp.mp.dps = 80
            a = df_pairs / 2.0
            x = z2_sum / 2.0
            Q = mp.gammainc(a, x, mp.inf) / mp.gamma(a)
            log_p = float(mp.log(Q))
            p_val = float(Q) if Q > 0 else 0.0
            neg_log10_p = float(-mp.log(Q) / mp.log(10))
            p_value_str = f"0.000 (<1e-{int(np.floor(neg_log10_p))})" if (p_val == 0.0 or neg_log10_p >= 3) else f"{p_val:.6f}"
            p_source = 'mpmath'
        except Exception:
            z_wh = ((z2_sum / df_pairs) ** (1.0 / 3.0) - (1 - 2.0 / (9.0 * df_pairs))) / np.sqrt(2.0 / (9.0 * df_pairs))
            log_p = norm.logsf(z_wh)
            tiny_log = np.log(np.finfo(float).tiny)
            p_val = float(np.exp(log_p)) if log_p > tiny_log else 0.0
            neg_log10_p = float(-log_p / np.log(10))
            p_value_str = f"0.000 (<1e-{int(np.floor(neg_log10_p))})" if p_val == 0.0 else f"{p_val:.6f}"
            p_source = 'wilson-hilferty'
    else:
        tiny_log = np.log(np.finfo(float).tiny)
        p_val = float(np.exp(log_p)) if log_p > tiny_log else 0.0
        neg_log10_p = float(-log_p / np.log(10))
        p_value_str = f"0.000 (<1e-{int(np.floor(neg_log10_p))})" if (p_val == 0.0 or neg_log10_p >= 3) else f"{p_val:.6f}"

    # Inverse-variance weighted mean difference and CI
    weights = 1.0 / np.square(valid['denom'])
    weighted_delta = float(np.sum(weights * valid['delta']) / np.sum(weights))
    se_weighted = float(1.0 / np.sqrt(np.sum(weights)))
    ci_low = weighted_delta - 1.96 * se_weighted
    ci_high = weighted_delta + 1.96 * se_weighted

    return {
        'num_pairs': df_pairs,
        'chi2': z2_sum,
        'df': df_pairs,
        'p_value': p_value_str,
        'weighted_mean_diff': weighted_delta,
        'ci_lower': float(ci_low),
        'ci_upper': float(ci_high),
        'per_pair': valid.reset_index(drop=True)
    }


def variance_redistribution_across_pcs(
        df: pd.DataFrame,
        event_col: str = 'event',
        pc_col: str = 'PC',
        value_col: str = 'value',
        timepoint_col: str = 'Timepoint',
        relative_time_col: Optional[str] = 'relative_time'
) -> Dict[str, object]:
    """
    Test whether BO→AO changes differ across PCs (redistribution across PCs)
    using a single omnibus statistic:

      Q_pc = Σ_p w_p (Δ̂_p − Δ̂)^2,  df = P − 1

    where Δ̂_p is the inverse-variance weighted mean change for PC p across
    events, and w_p = Σ_i w_ip with w_ip = 1/(SEM_AO^2 + SEM_BO^2) for pair (i,p).

    Returns a dict with Q_pc, df, formatted p_value, I2_pc, and the global
    pooled Δ̂ ± CI. Also returns a per-PC table (Δ̂_p, CI) for reference.
    """

    # Build per-pair stats
    per_pair = compute_event_pc_bo_ao_stats(
        df=df,
        event_col=event_col,
        pc_col=pc_col,
        value_col=value_col,
        timepoint_col=timepoint_col,
        relative_time_col=relative_time_col
    )

    valid = per_pair.dropna(subset=['denom', 'delta']).copy()
    if valid.empty:
        return {
            'Q_pc': np.nan,
            'df': 0,
            'p_value': 'NA',
            'I2_pc': np.nan,
            'global_delta': np.nan,
            'global_ci_lower': np.nan,
            'global_ci_upper': np.nan,
            'per_pc': pd.DataFrame(columns=[pc_col, 'delta_hat', 'ci_lower', 'ci_upper', 'weight_sum'])
        }

    # weights per pair
    valid['w'] = 1.0 / np.square(valid['denom'])

    # Aggregate within each PC
    per_pc = (
        valid
        .groupby(pc_col, observed=True)
        .apply(lambda g: pd.Series({
            'weight_sum': g['w'].sum(),
            'delta_hat': float(np.sum(g['w'] * g['delta']) / np.sum(g['w']))
        }))
        .reset_index()
    )

    # Variance of delta_hat per PC and CI
    per_pc['se'] = (1.0 / np.sqrt(per_pc['weight_sum'].replace(0.0, np.finfo(float).eps))).astype(float)
    per_pc['ci_lower'] = per_pc['delta_hat'] - 1.96 * per_pc['se']
    per_pc['ci_upper'] = per_pc['delta_hat'] + 1.96 * per_pc['se']

    # Global pooled effect across PCs (same as across all pairs)
    W_total = per_pc['weight_sum'].sum()
    delta_global = float(np.sum(per_pc['weight_sum'] * per_pc['delta_hat']) / W_total)
    se_global = float(1.0 / np.sqrt(W_total))
    ci_gl_low = delta_global - 1.96 * se_global
    ci_gl_high = delta_global + 1.96 * se_global

    # Q statistic across PCs
    Q_pc = float(np.sum(per_pc['weight_sum'] * np.square(per_pc['delta_hat'] - delta_global)))
    P = per_pc.shape[0]
    df_pc = max(0, P - 1)

    # p-value formatting with robust fallbacks
    log_p = chi2.logsf(Q_pc, df_pc) if df_pc > 0 else np.nan
    p_val = np.nan
    if not np.isfinite(log_p):
        try:
            import mpmath as mp
            mp.mp.dps = 80
            a = df_pc / 2.0
            x = Q_pc / 2.0
            Q = mp.gammainc(a, x, mp.inf) / mp.gamma(a)
            log_p = float(mp.log(Q))
            p_val = float(Q) if Q > 0 else 0.0
            neg_log10_p = float(-mp.log(Q) / mp.log(10))
            p_value_str = f"0.000 (<1e-{int(np.floor(neg_log10_p))})" if (p_val == 0.0 or neg_log10_p >= 3) else f"{p_val:.6f}"
        except Exception:
            # Wilson–Hilferty approximation
            z_wh = ((Q_pc / df_pc) ** (1.0 / 3.0) - (1 - 2.0 / (9.0 * df_pc))) / np.sqrt(2.0 / (9.0 * df_pc))
            log_p = norm.logsf(z_wh)
            tiny_log = np.log(np.finfo(float).tiny)
            p_val = float(np.exp(log_p)) if log_p > tiny_log else 0.0
            neg_log10_p = float(-log_p / np.log(10))
            p_value_str = f"0.000 (<1e-{int(np.floor(neg_log10_p))})" if p_val == 0.0 else f"{p_val:.6f}"
    else:
        tiny_log = np.log(np.finfo(float).tiny)
        p_val = float(np.exp(log_p)) if log_p > tiny_log else 0.0
        neg_log10_p = float(-log_p / np.log(10))
        p_value_str = f"0.000 (<1e-{int(np.floor(neg_log10_p))})" if (p_val == 0.0 or neg_log10_p >= 3) else f"{p_val:.6f}"

    # I^2 across PCs
    I2_pc = float(max(0.0, (Q_pc - df_pc) / Q_pc) * 100.0) if (Q_pc > 0 and df_pc > 0) else 0.0

    # Order columns for per_pc output
    per_pc_out = per_pc[[pc_col, 'delta_hat', 'ci_lower', 'ci_upper', 'weight_sum']].copy()

    return {
        'Q_pc': Q_pc,
        'df': df_pc,
        'p_value': p_value_str,
        'p_value_float': float(p_val) if np.isfinite(p_val) else np.nan,
        'I2_pc': I2_pc,
        'global_delta': delta_global,
        'global_ci_lower': float(ci_gl_low),
        'global_ci_upper': float(ci_gl_high),
        'per_pc': per_pc_out
    }