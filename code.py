import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import os
import glob



def dimdesc_pca_quant(
    variables_df: pd.DataFrame,
    pc_scores: "np.ndarray | pd.DataFrame",
    axes: Optional[Iterable[int]] = None,
    alpha: float = 0.05,
    dropna: str = "pairwise",
) -> Dict[str, pd.DataFrame]:
    """
    Compute per-axis correlations and two-sided p-values between quantitative variables
    and PCA dimensions (individual scores), similar to FactoMineR's dimdesc for PCA.

    This function mirrors the quantitative part of FactoMineR::dimdesc(PCA(...)) by:
      - Computing Pearson correlation r between each variable column and each requested PC axis
      - Computing the two-sided p-value for H0: r = 0 using a t-test with n-2 df
      - Returning only variables significant at level alpha for each axis

    Notes
    -----
    - Provide the individuals' PC scores (e.g., from sklearn: pca.transform(X)).
    - Correlations are invariant to linear scaling and centering of variables, so you can
      pass either raw variables or the standardized variables used to fit PCA.
    - Missing values are handled with pairwise deletion by default.

    Parameters
    ----------
    variables_df : pd.DataFrame
        DataFrame of quantitative variables used in PCA (rows: individuals, cols: variables).
    pc_scores : np.ndarray | pd.DataFrame
        Individuals-by-components matrix of PC scores (e.g., res.pca$ind$coord analogue).
    axes : Iterable[int] | None
        1-based indices of dimensions to describe. If None, all PCs in pc_scores are used.
    alpha : float
        Significance level for filtering variables per axis (two-sided p-value).
    dropna : {"pairwise", "complete"}
        - "pairwise": for each (variable, axis), drop rows where either side is NaN/infinite
        - "complete": restrict to rows having no NaNs across the selected variable and axis

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping from axis label (e.g., "Dim 1") to a DataFrame with columns:
          - variable: variable name
          - correlation: Pearson r with that axis
          - p_value: two-sided test p-value
        Only rows with p_value < alpha are included, sorted by ascending p_value then |r| desc.
    """
    from math import isfinite
    from scipy import stats

    if isinstance(pc_scores, pd.DataFrame):
        scores = pc_scores.to_numpy()
        num_components = pc_scores.shape[1]
    else:
        scores = np.asarray(pc_scores)
        num_components = scores.shape[1]

    if axes is None:
        axes_idx_1_based = list(range(1, num_components + 1))
    else:
        axes_idx_1_based = list(axes)

    # Ensure we only use numeric columns
    numeric_df = variables_df.select_dtypes(include=[np.number]).copy()
    variable_names: List[str] = list(numeric_df.columns)

    if numeric_df.shape[0] != scores.shape[0]:
        raise ValueError(
            "variables_df and pc_scores must have the same number of rows (individuals)."
        )

    results: Dict[str, pd.DataFrame] = {}

    # Pre-fetch columns as arrays for speed
    variable_arrays: Dict[str, np.ndarray] = {
        name: numeric_df[name].to_numpy() for name in variable_names
    }

    for axis_1b in axes_idx_1_based:
        axis_idx = axis_1b - 1
        if axis_idx < 0 or axis_idx >= num_components:
            raise IndexError(f"Axis {axis_1b} is out of bounds for scores with {num_components} components.")

        axis_scores = scores[:, axis_idx]

        per_axis_rows: List[Dict[str, float]] = []
        for var_name in variable_names:
            x = variable_arrays[var_name]

            if dropna == "pairwise":
                mask = (
                    np.isfinite(x) & np.isfinite(axis_scores)
                )
            elif dropna == "complete":
                # same as pairwise when comparing only two vectors
                mask = (
                    np.isfinite(x) & np.isfinite(axis_scores)
                )
            else:
                raise ValueError("dropna must be 'pairwise' or 'complete'.")

            x_masked = x[mask]
            y_masked = axis_scores[mask]

            # Need at least 3 observations to compute correlation and p-value
            if x_masked.size < 3:
                r = np.nan
                p_val = np.nan
            else:
                # If variable is constant after masking, correlation is undefined
                if np.allclose(np.nanstd(x_masked, ddof=1), 0.0):
                    r = np.nan
                    p_val = np.nan
                else:
                    r, p_val = stats.pearsonr(x_masked, y_masked)

            per_axis_rows.append({
                "variable": var_name,
                "correlation": r,
                "p_value": p_val,
            })

        df_axis = pd.DataFrame(per_axis_rows)
        # Optional significance filtering when p-values exist
        if alpha is not None and df_axis["p_value"].notna().any():
            df_axis = df_axis[df_axis["p_value"].notna()]
            df_axis = df_axis[df_axis["p_value"] < alpha]
        # Order by correlation value descending (most positive to most negative)
        df_axis = df_axis.sort_values(["correlation"], ascending=[False])
        results[f"Dim {axis_1b}"] = df_axis.reset_index(drop=True)

    return results


def dimdesc_from_saved_components(
    components_csv_path: str,
    stddev_csv_path: Optional[str] = None,
    eigenvalues_csv_path: Optional[str] = None,
    cleaned_csv_path: Optional[str] = None,
    n_samples: Optional[int] = None,
    time_value: Optional[float] = None,
    event_label: Optional[str] = None,
    axes: Optional[Iterable[int]] = None,
    alpha: Optional[float] = 0.05,
    components_scale: str = "auto",  # {"auto", "eigenvectors", "loadings"}
    time_tolerance: float = 1e-6,
) -> Dict[str, pd.DataFrame]:
    """
    Describe PCA dimensions using saved components and eigenvalues/stddev files, mimicking
    FactoMineR's dimdesc() for quantitative variables.

    This function computes, for a specific time and event label:
      - Correlation of each variable with each requested PC axis
      - Two-sided p-values (H0: correlation = 0) using n-2 degrees of freedom
      - Returns only variables with p < alpha for each axis

    It supports components either as eigenvectors or loadings (correlations). If set to
    components_scale="auto", the function will attempt to detect which one by checking
    whether sum_k (comp_{jk}^2) is close to 1 (loadings) or sum_k ((comp_{jk} * std_k)^2)
    is close to 1 (eigenvectors) for standardized PCA.

    Parameters
    ----------
    components_csv_path : str
        Path to components file with columns `PC1..PCp`, `Features`, `Time`, `Event`.
    stddev_csv_path : str | None
        Path to per-time PC stddev file with columns `PC1_std..PCp_std`, `Time`, `Event`.
        Provide this or `eigenvalues_csv_path` when components are eigenvectors or when
        components_scale="auto".
    eigenvalues_csv_path : str | None
        Path to per-time eigenvalues with columns like `eigen_val1..eigen_valp`, `Time`, `Event`.
        Used only if `stddev_csv_path` is None.
    cleaned_csv_path : str | None
        Deprecated. No longer used. Kept for backward compatibility.
    n_samples : int | None
        Optional number of individuals used to compute the PCA at the selected time/event.
        When not provided, p-values are set to NaN and variables are not filtered by alpha.
    time_value : float | None
        Time to select in the CSVs. If None, the most frequent/first time present is used.
    event_label : str | None
        Event label to select (e.g., "one", "all_events", or "NoEv"). If None, no event
        filtering is applied.
    axes : Iterable[int] | None
        1-based axis indices to describe. Defaults to all PCs present in components.
    alpha : float
        Significance threshold to filter variables per axis.
    components_scale : {"auto", "eigenvectors", "loadings"}
        - "loadings": components are already correlations var↔axis
        - "eigenvectors": components are eigenvectors; multiply by stddev to get correlations
        - "auto": attempt detection using the per-variable energy check
    time_tolerance : float
        Absolute tolerance when matching `time_value` against the `Time` column.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping axis label (e.g., "Dim 1") -> DataFrame with columns:
        ["variable", "correlation", "p_value"], filtered to p < alpha and sorted by
        p ascending then |correlation| descending.
    """
    import re
    from scipy import stats

    # Load components and filter by event/time
    comp_df = pd.read_csv(components_csv_path)

    # Identify PC columns and metadata columns
    pc_cols = [c for c in comp_df.columns if re.fullmatch(r"PC\d+", c)]
    if not pc_cols:
        raise ValueError("No PC columns found in components CSV (expected columns like 'PC1').")

    if axes is None:
        axes_idx_1b = list(range(1, len(pc_cols) + 1))
    else:
        axes_idx_1b = list(axes)

    # Narrow to event if provided
    df = comp_df.copy()
    if event_label is not None and "Event" in df.columns:
        df = df[df["Event"] == event_label]
    
    # Narrow to time if provided, using tolerance
    if time_value is not None and "Time" in df.columns:
        is_close = (df["Time"] - float(time_value)).abs() <= time_tolerance
        if not is_close.any():
            # Try exact comparison for string-like times
            is_close = df["Time"].astype(str) == str(time_value)
        df = df[is_close]

    if df.empty:
        raise ValueError("No component rows remain after filtering by event/time.")

    # For a given time/event, we expect one row per feature
    # Gather stddev per axis for that time/event, if available
    std_by_axis: Dict[int, float] = {}
    if stddev_csv_path is not None or eigenvalues_csv_path is not None:
        if stddev_csv_path is not None:
            sdev_df = pd.read_csv(stddev_csv_path)
            if event_label is not None and "Event" in sdev_df.columns:
                sdev_df = sdev_df[sdev_df["Event"] == event_label]
            if time_value is not None and "Time" in sdev_df.columns:
                is_close = (sdev_df["Time"] - float(time_value)).abs() <= time_tolerance
                if not is_close.any():
                    is_close = sdev_df["Time"].astype(str) == str(time_value)
                sdev_df = sdev_df[is_close]
            if sdev_df.empty:
                raise ValueError("No stddev rows found for the selected event/time.")
            sdev_row = sdev_df.iloc[0].to_dict()
            # Fill stddev for ALL PCs present in components, not only requested axes
            for c in pc_cols:
                a = int(c[2:])
                key = f"PC{a}_std"
                if key in sdev_row and pd.notna(sdev_row[key]):
                    std_by_axis[a] = float(sdev_row[key])
        else:
            eval_df = pd.read_csv(eigenvalues_csv_path)
            if event_label is not None and "Event" in eval_df.columns:
                eval_df = eval_df[eval_df["Event"] == event_label]
            if time_value is not None and "Time" in eval_df.columns:
                is_close = (eval_df["Time"] - float(time_value)).abs() <= time_tolerance
                if not is_close.any():
                    is_close = eval_df["Time"].astype(str) == str(time_value)
                eval_df = eval_df[is_close]
            if eval_df.empty:
                raise ValueError("No eigenvalue rows found for the selected event/time.")
            eval_row = eval_df.iloc[0].to_dict()
            # Fill stddev (sqrt eigenvalue) for ALL PCs present in components
            for c in pc_cols:
                a = int(c[2:])
                key = f"eigen_val{a}"
                if key in eval_row and pd.notna(eval_row[key]):
                    std_by_axis[a] = float(eval_row[key]) ** 0.5

    # Prepare correlations per axis
    # df rows: one per feature; columns: pc_cols + [Features, Time, Event]
    feature_col = "Features" if "Features" in df.columns else None
    if feature_col is None:
        raise ValueError("Components CSV must contain a 'Features' column with variable names.")

    # Auto-detect whether components are eigenvectors or loadings (correlations)
    def detect_components_scale(df_features: pd.DataFrame) -> str:
        # Use all PCs present, compute per-feature energy
        comps = df_features[pc_cols].to_numpy(dtype=float)
        # Build std vector only for PCs we have std for, and mask columns accordingly
        std_values: List[float] = []
        keep_indices: List[int] = []
        for idx, c in enumerate(pc_cols):
            a = int(c[2:])
            if a in std_by_axis:
                keep_indices.append(idx)
                std_values.append(std_by_axis[a])
        if not keep_indices:
            # Cannot scale; default to treat as loadings
            return "loadings"
        comps_sub = comps[:, keep_indices]
        std_vec = np.array(std_values, dtype=float)
        # Compare energies on the same subset of PCs
        energy_as_is = np.mean(np.sum(comps_sub ** 2, axis=1))
        comps_scaled = comps_sub * std_vec[None, :]
        energy_scaled = np.mean(np.sum(comps_scaled ** 2, axis=1))
        # For standardized PCA, per-feature total squared correlation with all PCs ~ 1
        if abs(energy_as_is - 1.0) < abs(energy_scaled - 1.0):
            return "loadings"
        return "eigenvectors"

    scale_mode = components_scale
    if components_scale == "auto":
        scale_mode = detect_components_scale(df)

    # Compute correlation matrix variables x axes
    # Keep only requested axes
    wanted_pc_cols = [f"PC{a}" for a in axes_idx_1b]
    for c in wanted_pc_cols:
        if c not in pc_cols:
            raise ValueError(f"Requested axis column '{c}' not found in components CSV.")

    comps_sel = df[[feature_col] + wanted_pc_cols].copy()
    comps_mat = comps_sel[wanted_pc_cols].to_numpy(dtype=float)

    if scale_mode == "eigenvectors":
        if not std_by_axis:
            raise ValueError(
                "stddev or eigenvalues CSV required when components_scale='eigenvectors' or 'auto'."
            )
        std_vec = np.array([std_by_axis[a] for a in axes_idx_1b], dtype=float)
        corr_mat = comps_mat * std_vec[None, :]
    elif scale_mode == "loadings":
        corr_mat = comps_mat
    else:
        raise ValueError("components_scale must be one of {'auto','eigenvectors','loadings'}")

    # Compute p-values for correlations using t-test with n-2 df
    results: Dict[str, pd.DataFrame] = {}
    for idx, axis_1b in enumerate(axes_idx_1b):
        r_vals = corr_mat[:, idx]
        if n_samples is None or n_samples < 3:
            p_vals = np.full_like(r_vals, fill_value=np.nan, dtype=float)
        else:
            dfree = n_samples - 2
            # Avoid divide-by-zero for |r|==1
            denom = np.maximum(1.0 - (r_vals ** 2), 1e-15)
            t_stat = r_vals * np.sqrt(dfree / denom)
            p_vals = 2.0 * stats.t.sf(np.abs(t_stat), df=dfree)
        axis_df = pd.DataFrame({
            "variable": comps_sel[feature_col].tolist(),
            "correlation": r_vals,
            "p_value": p_vals,
        })
        # Optional significance filtering (only if p-values are available)
        if (alpha is not None) and axis_df["p_value"].notna().any():
            axis_df = axis_df[axis_df["p_value"].notna()]
            axis_df = axis_df[axis_df["p_value"] < alpha]
        # Order by correlation value descending (most positive to most negative)
        axis_df = axis_df.sort_values(["correlation"], ascending=[False])
        results[f"Dim {axis_1b}"] = axis_df.reset_index(drop=True)

    return results

