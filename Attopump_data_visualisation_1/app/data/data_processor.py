"""Data processing: loading, cleaning, and transforming CSV data.

This is the core data-engineering module.  It sits between the raw CSV
files (loaded by ``io_local``) and the plotting layer (``app.plots.*``).
Every function here takes a ``pd.DataFrame`` (or folder name) and returns
either a transformed ``pd.DataFrame`` or metadata.

Responsibilities
----------------
1. **Test-type detection** — determine whether a test folder is a
   *constant-frequency* or *frequency-sweep* test using a four-level
   priority system (``freq_set_hz`` column → metadata JSON → regex →
   default).
2. **Column guessing** — auto-detect the time column and signal
   (flow / pressure / temperature) columns from a DataFrame.
3. **Time-format detection** — distinguish elapsed-seconds (merged.csv)
   from absolute timestamps (Flowboard raw data).
4. **Data preparation** — produce clean, plot-ready DataFrames for
   time-series, frequency-sweep, and constant-frequency views.
5. **Frequency binning** — aggregate sweep data into equal-width
   frequency bins with mean ± std statistics.
6. **Metadata / pattern persistence** — read and write
   ``test_metadata.json`` (per-folder overrides) and
   ``user_patterns.json`` (custom sweep-detection regexes).

Inputs
------
- ``pd.DataFrame`` loaded from CSV (lowercase columns, via ``io_local``).
- Folder names (``str``) for regex / metadata lookups.

Outputs
-------
- Cleaned ``pd.DataFrame`` objects ready for plotting.
- ``(test_type, detection_method, metadata_entry)`` tuples.
- ``SweepSpec`` dataclass instances.
"""

from __future__ import annotations

import json
import re
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
from .experiment_log import lookup_experiment_log_entry

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
    data_root: str | Path | None = None,
) -> tuple[str, str, dict | None]:
    """Determine test type using data-first detection hierarchy.

    Priority:
      0. Check saved test configuration (``test_configs.json``)
      1. Check ``freq_set_hz`` column – 1 unique value ⇒ constant, >1 ⇒ sweep
      2. Look up the shared experiment log
      3. Look up folder name in ``test_metadata.json``
      4. Regex pattern matching on folder name (built-in + user patterns)
      5. Default to "unknown"

    Returns
    -------
    (test_type, detection_method, metadata_entry)
        test_type:        "constant" | "sweep" | "unknown"
        detection_method: "freq_set_hz_column" | "experiment_log" |
                          "metadata" | "regex" | "saved_config" | "unknown"
        metadata_entry:   dict from metadata file (or None)
    """
    # ── Priority 0: saved test configuration ────────────────────────────
    from .test_configs import get_test_config

    saved_cfg = get_test_config(run_name)
    if saved_cfg is not None:
        return (saved_cfg.test_type, "saved_config", None)

    # ── Priority 1: freq_set_hz column in the data ──────────────────────
    if df is not None and "freq_set_hz" in df.columns:
        unique_freqs = df["freq_set_hz"].dropna().nunique()
        if unique_freqs <= 1:
            return ("constant", "freq_set_hz_column", None)
        else:
            return ("sweep", "freq_set_hz_column", None)

    # ── Priority 2: experiment log ─────────────────────────────────────
    log_entry = lookup_experiment_log_entry(data_root, run_name)
    if log_entry is not None and log_entry.inferred_test_type in ("constant", "sweep"):
        return (log_entry.inferred_test_type, "experiment_log", None)

    # ── Priority 3: metadata file ──────────────────────────────────────
    metadata = load_test_metadata()
    entry = metadata.get(run_name)
    if entry:
        test_type = entry.get("type", "unknown")
        if test_type in ("constant", "sweep"):
            return (test_type, "metadata", entry)

    # ── Priority 4: regex patterns (built-in + user) ──────────────────
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

    # ── Priority 5: unknown ────────────────────────────────────────────
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

    Also parses a repetition count if present, e.g. ``10x_Sweep`` → 10.
    """
    from .config import SWEEP_PATTERNS
    
    # ── Parse repetition count (e.g. "10x", "5X") ──
    rep_match = re.search(r"(\d+)\s*[xX]", run_name)
    num_repeats = int(rep_match.group(1)) if rep_match else 1
    
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
                num_repeats=max(1, num_repeats),
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

    Handles both elapsed seconds (merged.csv) and absolute timestamps
    (Flowboard).

    Frequency assignment priority
    -----------------------------
    1. ``freq_set_hz`` column present in *full_df* → use it directly.
    2. ``spec.duration_s > 0`` → compute from elapsed time + spec.
    3. ``spec`` has valid start/end but no duration → estimate sweep
       period from the total elapsed time and ``spec.num_repeats``.
    4. Fall-back: elapsed time as a proxy (labels the axis correctly
       but values are seconds, not Hz).

    Parameters
    ----------
    full_df : pd.DataFrame | None
        The *original* unprocessed DataFrame (before column selection).
        When provided and it contains a ``freq_set_hz`` column, that
        column is used directly for frequency mapping – much more
        accurate than computing it from elapsed time.
    """
    # ── Determine data source and whether freq_set_hz exists ────────
    source_df = full_df if full_df is not None else df
    has_freq_col = "freq_set_hz" in source_df.columns

    # ── Build working DataFrame keeping freq_set_hz aligned ─────────
    # We select columns from *source_df* (not df) so that freq_set_hz
    # stays row-aligned through dedup / dropna filtering.
    cols_to_keep = [time_col, signal_col]
    if has_freq_col:
        cols_to_keep.append("freq_set_hz")

    sweep_df = source_df[cols_to_keep].copy()
    sweep_df = sweep_df.reset_index(drop=True)

    # ── Time parsing (absolute timestamps only) ────────────────────
    time_format = detect_time_format(source_df, time_col)
    if parse_time and time_format == "absolute_timestamp":
        sweep_df[time_col] = pd.to_datetime(
            sweep_df[time_col], errors=TIME_PARSE_ERROR_HANDLING
        )

    # ── De-duplicate & drop NaN ────────────────────────────────────
    sweep_df = sweep_df.drop_duplicates(subset=[time_col], keep=DUPLICATE_HANDLING)
    if drop_na:
        sweep_df = sweep_df.dropna(subset=[time_col, signal_col])
    sweep_df = sweep_df.reset_index(drop=True)

    if sweep_df.empty:
        raise ValueError("No data remaining after time-series preparation.")

    # ── Elapsed seconds ────────────────────────────────────────────
    if time_format == "elapsed_seconds":
        sweep_df["Elapsed_s"] = pd.to_numeric(sweep_df[time_col], errors="coerce")
    elif not sweep_df.empty and np.issubdtype(sweep_df[time_col].dtype, np.datetime64):
        sweep_df["Elapsed_s"] = (
            sweep_df[time_col] - sweep_df[time_col].iloc[0]
        ).dt.total_seconds()
    else:
        sweep_df["Elapsed_s"] = pd.to_numeric(sweep_df[time_col], errors="coerce")

    sweep_df = sweep_df[np.isfinite(sweep_df["Elapsed_s"])].copy()
    if sweep_df.empty:
        raise ValueError("No finite elapsed-time values after cleaning.")

    # ── Frequency column ────────────────────────────────────────────
    # Priority 1: real freq_set_hz column (properly aligned above)
    if has_freq_col and "freq_set_hz" in sweep_df.columns:
        sweep_df["Frequency"] = pd.to_numeric(
            sweep_df["freq_set_hz"], errors="coerce"
        )
        sweep_df = sweep_df.drop(columns=["freq_set_hz"])
        sweep_df = sweep_df.dropna(subset=["Frequency"]).copy()

    elif spec is not None and spec.duration_s > 0:
        # Priority 2: compute from sweep spec (known duration)
        sweep_s = float(spec.duration_s)
        phase = (sweep_df["Elapsed_s"] % sweep_s) / sweep_s
        sweep_df["Frequency"] = spec.start_hz + (
            spec.end_hz - spec.start_hz
        ) * phase

    elif spec is not None and spec.start_hz != spec.end_hz:
        # Priority 3: spec with start/end but unknown duration.
        # Estimate one sweep period from total elapsed time and the
        # repetition count parsed from the folder name.
        total_elapsed = float(
            sweep_df["Elapsed_s"].max() - sweep_df["Elapsed_s"].min()
        )
        if total_elapsed > 0:
            n_reps = max(1, spec.num_repeats)
            est_dur = total_elapsed / n_reps
            elapsed_from_start = sweep_df["Elapsed_s"] - sweep_df["Elapsed_s"].min()
            phase = (elapsed_from_start % est_dur) / est_dur
            sweep_df["Frequency"] = spec.start_hz + (
                spec.end_hz - spec.start_hz
            ) * phase
        else:
            sweep_df["Frequency"] = sweep_df["Elapsed_s"]

    else:
        # Priority 4: elapsed time as proxy (last resort)
        sweep_df["Frequency"] = sweep_df["Elapsed_s"]

    # Clean up freq_set_hz if still lingering
    sweep_df = sweep_df.drop(columns=["freq_set_hz"], errors="ignore")

    # ── Sweep cycle index ───────────────────────────────────────────
    if spec is not None and spec.duration_s > 0:
        sweep_s = float(spec.duration_s)
        raw_sweep = sweep_df["Elapsed_s"] / sweep_s
        raw_sweep = raw_sweep.where(np.isfinite(raw_sweep), 0)
        sweep_df["Sweep"] = raw_sweep.astype(int)
    elif spec is not None and spec.start_hz != spec.end_hz:
        total_elapsed = float(
            sweep_df["Elapsed_s"].max() - sweep_df["Elapsed_s"].min()
        )
        n_reps = max(1, spec.num_repeats)
        if total_elapsed > 0 and n_reps > 1:
            est_dur = total_elapsed / n_reps
            raw_sweep = (
                sweep_df["Elapsed_s"] - sweep_df["Elapsed_s"].min()
            ) / est_dur
            raw_sweep = raw_sweep.where(np.isfinite(raw_sweep), 0)
            sweep_df["Sweep"] = raw_sweep.astype(int)
        else:
            sweep_df["Sweep"] = 0
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

    # Single-point bins produce NaN std (ddof=1 with n=1). Fill with 0.
    binned["std"] = binned["std"].fillna(0)

    binned["freq_center"] = binned["bin"].map(
        lambda i: float(centers[int(i)]) if int(i) < len(centers) else float(fmax)
    )
    return binned.sort_values("freq_center")


def compute_frequency_average(
    df: pd.DataFrame,
    value_col: str,
    freq_col: str = "Frequency",
) -> pd.DataFrame:
    """Compute mean and std of signal values at each unique frequency.

    Groups the data by the frequency column and calculates per-frequency
    statistics.  Handles single-point frequencies safely (std → 0 when
    only one data point exists for that frequency).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least ``freq_col`` and ``value_col``.
    value_col : str
        Name of the signal / flow column.
    freq_col : str
        Name of the frequency column.

    Returns
    -------
    pd.DataFrame
        Columns: ``freq``, ``mean``, ``std``, ``count``.
        Sorted by ascending frequency.
    """
    out = df[[freq_col, value_col]].dropna().copy()
    grouped = (
        out.groupby(freq_col)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # Single-point frequencies: std is NaN (ddof=1), fill with 0.
    grouped["std"] = grouped["std"].fillna(0)
    grouped = grouped.rename(columns={freq_col: "freq"})
    return grouped.sort_values("freq").reset_index(drop=True)


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
    
    # Always parse absolute timestamps so elapsed-second arithmetic works.
    # The `parse_time` flag is a UI hint; for internal math we MUST have
    # numeric or datetime values.
    if time_format == 'absolute_timestamp':
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


# ============================================================================
# FREQUENCY SLICE EXTRACTION (for mixed-type comparison)
# ============================================================================

def extract_frequency_slice(
    sweep_df: pd.DataFrame,
    target_freq_hz: float,
    tolerance_hz: float = 5.0,
    signal_col: str = "flow",
) -> pd.DataFrame:
    """Extract data points from a frequency sweep near a target frequency.

    Used to compare sweep-test results with constant-frequency tests by
    isolating the sweep readings around the constant frequency.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Sweep DataFrame (must contain a ``Frequency`` column).
    target_freq_hz : float
        Centre frequency to extract (e.g. 500.0).
    tolerance_hz : float
        Half-width of the extraction window (default ±5 Hz).
    signal_col : str
        Signal column name (kept for downstream compatibility).

    Returns
    -------
    pd.DataFrame
        Subset of ``sweep_df`` where Frequency is within
        ``target_freq_hz ± tolerance_hz``.  Empty DataFrame if no match.
    """
    if "Frequency" not in sweep_df.columns:
        return pd.DataFrame()
    mask = (
        (sweep_df["Frequency"] >= target_freq_hz - tolerance_hz)
        & (sweep_df["Frequency"] <= target_freq_hz + tolerance_hz)
    )
    return sweep_df.loc[mask].copy()


def detect_constant_frequency(
    df: pd.DataFrame,
    run_name: str = "",
    data_root: str | Path | None = None,
) -> float | None:
    """Auto-detect the operating frequency of a constant-frequency test.

    Priority:
      1. ``freq_set_hz`` column with a single unique value.
      2. ``frequency_hz`` field in ``test_metadata.json``.
      3. ``DEFAULT_CONSTANT_FREQUENCY_HZ`` from config.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded CSV data.
    run_name : str
        Folder name (for metadata lookup).

    Returns
    -------
    float or None
        Detected frequency in Hz.
    """
    # Priority 1: freq_set_hz column with single unique value
    if "freq_set_hz" in df.columns:
        unique = df["freq_set_hz"].dropna().unique()
        if len(unique) == 1:
            return float(unique[0])

    # Priority 2: experiment log
    log_entry = lookup_experiment_log_entry(data_root, run_name)
    if log_entry is not None and log_entry.frequency_hz is not None:
        return float(log_entry.frequency_hz)

    # Priority 3: metadata file
    meta = load_test_metadata()
    entry = meta.get(run_name, {})
    if entry.get("frequency_hz"):
        return float(entry["frequency_hz"])

    # Priority 4: config default
    return DEFAULT_CONSTANT_FREQUENCY_HZ
