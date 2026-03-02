"""Data processing: loading, cleaning, and transforming CSV data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CONSTANT_FREQUENCY_CUTOFF,
    DEFAULT_CONSTANT_FREQUENCY_HZ,
    DROPNA_DEFAULT,
    DUPLICATE_HANDLING,
    SWEEP_PATTERN,
    SweepSpec,
    TIME_COLUMN_CANDIDATES,
    TIME_PARSE_ERROR_HANDLING,
)


def guess_time_column(df: pd.DataFrame) -> str | None:
    """Guess the time column from common naming patterns."""
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def guess_signal_column(df: pd.DataFrame, signal_type: str = "flow") -> str | None:
    """Guess a single signal column by type (flow, pressure, temperature, etc.).
    
    Returns the first matching column from heuristics, or first numeric column as fallback.
    """
    from .config import SIGNAL_COLUMN_HEURISTICS
    
    if signal_type not in SIGNAL_COLUMN_HEURISTICS:
        # Fallback: return first numeric column
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        return numeric_cols[0] if numeric_cols else None
    
    for candidate in SIGNAL_COLUMN_HEURISTICS[signal_type]:
        if candidate in df.columns:
            return candidate
    
    # Fallback: return first numeric column
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return numeric_cols[0] if numeric_cols else None


def get_signal_columns(df: pd.DataFrame, signal_type: str = "flow") -> list[str]:
    """Return all candidate signal columns for a given type (flow, pressure, etc).
    
    Returns matching heuristic columns + all numeric columns for selection.
    For Flowboard format, also matches columns containing "flow" that aren't setpoints/events.
    """
    from .config import SIGNAL_COLUMN_HEURISTICS
    
    candidates = []
    
    # Add matching heuristic columns first (exact matches)
    if signal_type in SIGNAL_COLUMN_HEURISTICS:
        for cand in SIGNAL_COLUMN_HEURISTICS[signal_type]:
            if cand in df.columns and cand not in candidates:
                candidates.append(cand)
    
    # For flow type, also try partial matching for Flowboard format
    # Look for columns that contain "flow" but exclude setpoint/air/event columns
    if signal_type == "flow":
        for col in df.columns:
            col_lower = col.lower()
            # Match columns with "flow" in name, excluding setpoints and events
            if "flow" in col_lower and "setpoint" not in col_lower and "air" not in col_lower:
                if col not in candidates:
                    candidates.append(col)
    
    # Add all numeric columns (not already in candidates)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if col not in candidates:
            candidates.append(col)
    
    return candidates


def parse_sweep_spec_from_name(run_name: str) -> SweepSpec | None:
    """Extract sweep specification from run folder name (e.g., '10Hz_500Hz_60s')."""
    match = SWEEP_PATTERN.search(run_name)
    if not match:
        return None
    
    return SweepSpec(
        start_hz=float(match.group("start")),
        end_hz=float(match.group("end")),
        duration_s=float(match.group("dur")),
    )


def detect_time_format(df: pd.DataFrame, time_col: str) -> str:
    """Detect whether time column is elapsed seconds or absolute timestamp.
    
    Returns:
        'elapsed_seconds': Time is measured in seconds from test start (merged.csv format)
        'absolute_timestamp': Time is an absolute timestamp (Flowboard format)
    """
    # If column is named t_s or similar, it's elapsed seconds
    if time_col.lower() in ['t_s', 't']:
        return 'elapsed_seconds'
    
    # Try to detect from first non-null value
    sample = df[time_col].dropna().iloc[0] if not df[time_col].dropna().empty else None
    
    if sample is None:
        return 'elapsed_seconds'  # Default
    
    # If it's a numeric type and small (< 1000000), likely elapsed seconds
    try:
        val = float(sample)
        if val < 1000000:  # Likely elapsed seconds (max ~11 days)
            return 'elapsed_seconds'
        else:
            return 'absolute_timestamp'  # Unix timestamp or large number
    except (ValueError, TypeError):
        # It's a string or datetime-like, treat as absolute timestamp
        return 'absolute_timestamp'


def prepare_time_series_data(
    df: pd.DataFrame,
    time_col: str,
    signal_col: str,
    parse_time: bool = True,
    drop_na: bool = DROPNA_DEFAULT,
) -> pd.DataFrame:
    """Prepare data for time series plotting.
    
    Handles:
    - Selecting columns
    - Parsing time to datetime (only for absolute timestamps, not elapsed seconds)
    - Removing duplicates (by time)
    - Dropping NaNs
    
    Time format detection:
    - merged.csv: t_s column is elapsed seconds (do NOT parse as datetime)
    - Flowboard: Time is absolute timestamp (DO parse as datetime)
    """
    # Select relevant columns
    plot_df = df[[time_col, signal_col]].copy()
    plot_df = plot_df.reset_index(drop=True)
    
    # Detect time format
    time_format = detect_time_format(df, time_col)
    
    # Parse time ONLY if it's an absolute timestamp
    if parse_time and time_format == 'absolute_timestamp':
        plot_df[time_col] = pd.to_datetime(
            plot_df[time_col], 
            errors=TIME_PARSE_ERROR_HANDLING
        )
    
    # Remove duplicates (keep first occurrence of duplicate timestamps)
    plot_df = plot_df.drop_duplicates(subset=[time_col], keep=DUPLICATE_HANDLING)
    
    # Drop NaNs
    if drop_na:
        plot_df = plot_df.dropna(subset=[time_col, signal_col])
    
    # Reset index after filtering
    plot_df = plot_df.reset_index(drop=True)
    
    return plot_df


def prepare_sweep_data(
    df: pd.DataFrame,
    time_col: str,
    signal_col: str,
    spec: SweepSpec,
    parse_time: bool = True,
    drop_na: bool = DROPNA_DEFAULT,
) -> pd.DataFrame:
    """Prepare data for sweep analysis.
    
    Adds:
    - Elapsed_s: elapsed time from start
    - Sweep: sweep cycle index
    - Frequency: calculated frequency at each point
    """
    # Start with time series preparation
    sweep_df = prepare_time_series_data(df, time_col, signal_col, parse_time, drop_na)
    
    if not parse_time or not np.issubdtype(sweep_df[time_col].dtype, np.datetime64):
        raise ValueError("Time column must be datetime for sweep analysis")
    
    # Calculate elapsed seconds
    sweep_df = sweep_df.copy()
    sweep_df["Elapsed_s"] = (sweep_df[time_col] - sweep_df[time_col].iloc[0]).dt.total_seconds()
    
    # Calculate sweep cycle and frequency
    sweep_s = float(spec.duration_s)
    sweep_df["Sweep"] = (sweep_df["Elapsed_s"] // sweep_s).astype(int)
    
    phase = (sweep_df["Elapsed_s"] % sweep_s) / sweep_s
    sweep_df["Frequency"] = spec.start_hz + (spec.end_hz - spec.start_hz) * phase
    
    return sweep_df


def bin_by_frequency(
    df: pd.DataFrame,
    value_col: str,
    freq_col: str = "Frequency",
    bin_hz: float = 5.0,
) -> pd.DataFrame:
    """Bin frequency sweep data and compute mean ± std per bin."""
    out = df[[freq_col, value_col]].dropna().copy()
    
    f = out[freq_col].astype(float).to_numpy()
    v = out[value_col].astype(float).to_numpy()
    
    # Validate frequency range
    fmin = float(np.nanmin(f))
    fmax = float(np.nanmax(f))
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
        raise ValueError("Invalid frequency range for binning.")
    
    # Create bins
    edges = np.arange(fmin, fmax + bin_hz, bin_hz)
    idx = np.digitize(f, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    centers = (edges[:-1] + edges[1:]) / 2.0
    
    # Aggregate by bin
    binned = (
        pd.DataFrame({"bin": idx, "value": v})
        .groupby("bin")
        .agg(mean=("value", "mean"), std=("value", "std"), count=("value", "count"))
        .reset_index()
    )
    
    binned["freq_center"] = binned["bin"].map(lambda i: float(centers[int(i)]))
    return binned.sort_values("freq_center")


def is_constant_frequency_test(run_name: str) -> bool:
    """Check if a test folder name indicates constant frequency (not a sweep).
    
    Tests are sweep tests if they match pattern like '10Hz_500Hz_60s'.
    Otherwise, they're constant frequency tests.
    """
    return parse_sweep_spec_from_name(run_name) is None


def prepare_constant_frequency_data(
    df: pd.DataFrame,
    time_col: str,
    signal_col: str,
    frequency_hz: float,
    parse_time: bool = True,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Prepare data for constant frequency visualization.
    
    Creates time-based windows to show distribution of flow across the test duration.
    Handles both:
    - Elapsed seconds (merged.csv): numeric time values in seconds
    - Absolute timestamps (Flowboard): datetime values
    
    Returns DataFrame with columns: [time_col, signal_col, "Time_Window"]
    where Time_Window groups data into 1-second intervals for statistics.
    """
    out = df[[time_col, signal_col]].copy()
    
    # Detect time format
    time_format = detect_time_format(df, time_col)
    
    # Parse time ONLY if absolute timestamp
    if parse_time and time_format == 'absolute_timestamp':
        out[time_col] = pd.to_datetime(out[time_col], errors=TIME_PARSE_ERROR_HANDLING)
    
    if drop_na:
        out = out.dropna(subset=[time_col, signal_col])
    
    out = out.reset_index(drop=True)
    
    if out.empty:
        raise ValueError("No valid data after filtering.")
    
    # Create time windows based on format
    if time_format == 'elapsed_seconds':
        # For elapsed seconds: directly use numeric values and bin them
        out['Time_Window'] = (out[time_col] // 1.0).astype(int)
    else:
        # For absolute timestamps: convert to elapsed seconds first, then bin
        t_numeric = (out[time_col] - out[time_col].iloc[0]).dt.total_seconds()
        out['Time_Window'] = (t_numeric // 1.0).astype(int)
    
    return out
