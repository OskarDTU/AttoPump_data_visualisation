"""Data processing: loading, cleaning, and transforming CSV data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CONSTANT_FREQUENCY_CUTOFF,
    DEFAULT_CONSTANT_FREQUENCY_HZ,
    DROPNA_DEFAULT,
    DUPLICATE_HANDLING,
    SWEEP_PATTERN,
    SWEEP_PATTERNS,
    SweepSpec,
    TIME_COLUMN_CANDIDATES,
    TIME_PARSE_ERROR_HANDLING,
)

# ============================================================================
# TEST METADATA (loaded once, cached)
# ============================================================================
_METADATA_PATH = Path(__file__).parent / "test_metadata.json"
_USER_PATTERNS_PATH = Path(__file__).parent / "user_patterns.json"

_metadata_cache: dict | None = None


def load_test_metadata() -> dict:
    """Load the test metadata JSON file. Cached after first load."""
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache

    if _METADATA_PATH.exists():
        with open(_METADATA_PATH, "r") as f:
            raw = json.load(f)
        _metadata_cache = raw.get("tests", {})
    else:
        _metadata_cache = {}
    return _metadata_cache


def save_metadata_entry(folder_name: str, entry: dict) -> None:
    """Add or update a single entry in test_metadata.json."""
    global _metadata_cache

    if _METADATA_PATH.exists():
        with open(_METADATA_PATH, "r") as f:
            raw = json.load(f)
    else:
        raw = {"_description": "Auto-generated metadata.", "tests": {}}

    raw.setdefault("tests", {})[folder_name] = entry

    with open(_METADATA_PATH, "w") as f:
        json.dump(raw, f, indent=2)

    # Invalidate cache
    _metadata_cache = None


def load_user_patterns() -> list[str]:
    """Load user-defined sweep regex patterns (raw strings)."""
    if _USER_PATTERNS_PATH.exists():
        with open(_USER_PATTERNS_PATH, "r") as f:
            return json.load(f)
    return []


def save_user_patterns(patterns: list[str]) -> None:
    """Save user-defined sweep regex patterns."""
    with open(_USER_PATTERNS_PATH, "w") as f:
        json.dump(patterns, f, indent=2)


def detect_test_type(
    run_name: str,
    df: pd.DataFrame | None = None,
) -> tuple[str, str, dict | None]:
    """Determine test type using data-first detection hierarchy.

    Priority:
      1. Check ``freq_set_hz`` column – 1 unique value ⇒ constant, >1 ⇒ sweep
      2. Look up folder name in ``test_metadata.json``
      3. Regex pattern matching on folder name (built-in + user patterns)
      4. Default to "unknown"

    Returns
    -------
    (test_type, detection_method, metadata_entry)
        test_type:        "constant" | "sweep" | "unknown"
        detection_method: "freq_set_hz_column" | "metadata" | "regex" | "unknown"
        metadata_entry:   dict from metadata file (or None)
    """
    # ── Priority 1: freq_set_hz column in the data ──────────────────────
    if df is not None and "freq_set_hz" in df.columns:
        unique_freqs = df["freq_set_hz"].dropna().nunique()
        if unique_freqs <= 1:
            return ("constant", "freq_set_hz_column", None)
        else:
            return ("sweep", "freq_set_hz_column", None)

    # ── Priority 2: metadata file ──────────────────────────────────────
    metadata = load_test_metadata()
    entry = metadata.get(run_name)
    if entry:
        test_type = entry.get("type", "unknown")
        if test_type in ("constant", "sweep"):
            return (test_type, "metadata", entry)

    # ── Priority 3: regex patterns (built-in + user) ──────────────────
    import re

    # Try built-in patterns first
    spec = parse_sweep_spec_from_name(run_name)
    if spec is not None:
        return ("sweep", "regex", None)

    # Try user-defined patterns
    for pattern_str in load_user_patterns():
        try:
            if re.search(pattern_str, run_name, re.IGNORECASE):
                return ("sweep", "user_regex", None)
        except re.error:
            continue

    # ── Priority 4: unknown ────────────────────────────────────────────
    return ("unknown", "unknown", None)


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
    """Extract sweep specification from run folder name.
    
    Handles multiple naming conventions:
      - 10Hz_500Hz_60s             → 10 to 500 Hz, 60 s
      - 1Hz_1500H_Hz_500_seconds   → 1 to 1500 Hz, 500 s
      - 1Hz-1kHz                   → 1 to 1000 Hz (kHz multiplier), duration unknown
    """
    from .config import SWEEP_PATTERNS
    
    for pattern in SWEEP_PATTERNS:
        match = pattern.search(run_name)
        if match:
            groups = match.groupdict()
            start_hz = float(groups["start"])
            
            # Handle kHz: check if the pattern matched a kHz variant
            end_hz = float(groups["end"])
            matched_text = match.group(0)
            if "khz" in matched_text.lower():
                end_hz *= 1000.0
            
            # Duration may not be present (e.g. kHz pattern)
            duration_s = float(groups["dur"]) if "dur" in groups and groups["dur"] else 0.0
            
            return SweepSpec(
                start_hz=start_hz,
                end_hz=end_hz,
                duration_s=duration_s,
            )
    
    return None


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
    spec: SweepSpec | None = None,
    parse_time: bool = True,
    drop_na: bool = DROPNA_DEFAULT,
    full_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Prepare data for sweep analysis.

    Adds:
    - Elapsed_s: elapsed time from start
    - Sweep: sweep cycle index (if spec provided and duration > 0)
    - Frequency: from ``freq_set_hz`` column if available, else computed

    Handles both elapsed seconds (merged.csv) and absolute timestamps (Flowboard).

    Parameters
    ----------
    full_df : pd.DataFrame | None
        The *original* unprocessed DataFrame (before column selection).
        When provided and it contains a ``freq_set_hz`` column, that
        column is used directly for frequency mapping – much more
        accurate than computing it from elapsed time.
    """
    # Start with time series preparation
    sweep_df = prepare_time_series_data(df, time_col, signal_col, parse_time, drop_na)
    sweep_df = sweep_df.copy()

    # Calculate elapsed seconds depending on time format
    time_format = detect_time_format(df, time_col)

    if time_format == 'elapsed_seconds':
        sweep_df["Elapsed_s"] = pd.to_numeric(sweep_df[time_col], errors="coerce")
    elif np.issubdtype(sweep_df[time_col].dtype, np.datetime64):
        sweep_df["Elapsed_s"] = (
            sweep_df[time_col] - sweep_df[time_col].iloc[0]
        ).dt.total_seconds()
    else:
        sweep_df["Elapsed_s"] = pd.to_numeric(sweep_df[time_col], errors="coerce")

    # Drop rows where elapsed time is NaN / inf
    sweep_df = sweep_df[np.isfinite(sweep_df["Elapsed_s"])].copy()
    if sweep_df.empty:
        raise ValueError("No finite elapsed-time values after cleaning.")

    # ── Frequency column ────────────────────────────────────────────
    # Priority 1: use the real freq_set_hz column from the original CSV
    source_df = full_df if full_df is not None else df
    if "freq_set_hz" in source_df.columns:
        # Align by index length (the time-series prep may have dropped rows)
        freq_series = pd.to_numeric(
            source_df["freq_set_hz"], errors="coerce"
        ).reindex(sweep_df.index)
        sweep_df["Frequency"] = freq_series.values
        # Drop rows with missing frequency
        sweep_df = sweep_df.dropna(subset=["Frequency"]).copy()
    elif spec is not None and spec.duration_s > 0:
        # Priority 2: compute from sweep spec
        sweep_s = float(spec.duration_s)
        phase = (sweep_df["Elapsed_s"] % sweep_s) / sweep_s
        sweep_df["Frequency"] = spec.start_hz + (
            spec.end_hz - spec.start_hz
        ) * phase
    else:
        # Priority 3: elapsed time as proxy
        sweep_df["Frequency"] = sweep_df["Elapsed_s"]

    # ── Sweep cycle index ───────────────────────────────────────────
    if spec is not None and spec.duration_s > 0:
        sweep_s = float(spec.duration_s)
        raw_sweep = sweep_df["Elapsed_s"] / sweep_s
        raw_sweep = raw_sweep.where(np.isfinite(raw_sweep), 0)
        sweep_df["Sweep"] = raw_sweep.astype(int)
    else:
        sweep_df["Sweep"] = 0

    return sweep_df


def bin_by_frequency(
    df: pd.DataFrame,
    value_col: str,
    freq_col: str = "Frequency",
    bin_hz: float = 5.0,
    max_bins: int = 10000,
) -> pd.DataFrame:
    """Bin frequency sweep data and compute mean ± std per bin."""
    out = df[[freq_col, value_col]].dropna().copy()

    f = out[freq_col].astype(float).to_numpy()
    v = out[value_col].astype(float).to_numpy()

    # Remove non-finite values
    mask = np.isfinite(f) & np.isfinite(v)
    f, v = f[mask], v[mask]
    if len(f) == 0:
        raise ValueError("No finite data remaining after cleaning.")

    fmin, fmax = float(np.min(f)), float(np.max(f))
    if fmax <= fmin:
        raise ValueError(
            f"Frequency range is flat ({fmin}). Cannot create bins."
        )

    # Clamp bin width so we don't create too many bins
    bin_hz = max(bin_hz, (fmax - fmin) / max_bins)

    edges = np.arange(fmin, fmax + bin_hz, bin_hz)
    if len(edges) < 2:
        edges = np.array([fmin, fmax])

    idx = np.digitize(f, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    centers = (edges[:-1] + edges[1:]) / 2.0

    binned = (
        pd.DataFrame({"bin": idx, "value": v})
        .groupby("bin")
        .agg(mean=("value", "mean"), std=("value", "std"), count=("value", "count"))
        .reset_index()
    )

    binned["freq_center"] = binned["bin"].map(
        lambda i: float(centers[int(i)]) if int(i) < len(centers) else float(fmax)
    )
    return binned.sort_values("freq_center")


def is_constant_frequency_test(run_name: str, df: pd.DataFrame | None = None) -> bool:
    """Check if a test is constant frequency (not a sweep).

    Uses the full detection hierarchy from ``detect_test_type``:
      1. ``freq_set_hz`` column → 1 unique value = constant
      2. Metadata file lookup
      3. Regex pattern matching on folder name
      4. Default to constant when unknown

    Parameters
    ----------
    run_name : str
        Name of the test folder.
    df : pd.DataFrame | None
        Loaded CSV data (if available). Enables freq_set_hz auto-detection.
    """
    test_type, _method, _entry = detect_test_type(run_name, df)
    # "unknown" defaults to constant (safest: shows boxplot/histogram)
    return test_type != "sweep"


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
