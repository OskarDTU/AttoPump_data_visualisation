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

_FREQUENCY_HOLD_RUN_MULTIPLIER = 4.0
_MIN_FREQUENCY_HOLD_POINTS = 50
_RECOMMENDATION_ROLLING_WINDOW = 5
_RECOMMENDATION_ROLLING_STD_TARGET_FRACTION = 0.05
_RECOMMENDATION_ISOLATED_JUMP_TARGET_FRACTION = 0.025
_RECOMMENDATION_ISOLATED_JUMP_RATIO = 1.75
_RECOMMENDATION_BIN_WIDTH_PENALTY = 0.35
_RECOMMENDATION_ROUGHNESS_PERCENTILE = 90.0
_RECOMMENDATION_RESERVED_COLUMNS = {
    "elapsed_s",
    "frequency",
    "freq_center",
    "freq_set_hz",
    "isfrequencyhold",
    "frequencyrunid",
    "frequencyrunpoints",
    "sweep",
    "t_s",
    "time",
}

# ============================================================================
# TEST METADATA (loaded once, cached)
# ============================================================================
_METADATA_PATH = Path(__file__).parent / "test_metadata.json"
_USER_PATTERNS_PATH = Path(__file__).parent / "user_patterns.json"

_metadata_cache: dict | None = None
_metadata_cache_signature: tuple[str, int, int] | None = None
_user_patterns_cache: list[str] | None = None
_user_patterns_cache_signature: tuple[str, int, int] | None = None


def _path_signature(path: Path) -> tuple[str, int, int]:
    """Return a cheap file signature for cache invalidation."""
    if not path.exists():
        return (str(path), 0, 0)
    stat = path.stat()
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


def test_metadata_storage_signature() -> tuple[str, int, int]:
    """Return a file signature for ``test_metadata.json``."""
    return _path_signature(_METADATA_PATH)


def user_patterns_storage_signature() -> tuple[str, int, int]:
    """Return a file signature for ``user_patterns.json``."""
    return _path_signature(_USER_PATTERNS_PATH)


def load_test_metadata() -> dict:
    """Load the test metadata JSON file. Cached after first load."""
    global _metadata_cache
    global _metadata_cache_signature

    signature = test_metadata_storage_signature()
    if _metadata_cache is not None and _metadata_cache_signature == signature:
        return _metadata_cache

    if _METADATA_PATH.exists():
        with open(_METADATA_PATH, "r") as f:
            raw = json.load(f)
        _metadata_cache = raw.get("tests", {})
    else:
        _metadata_cache = {}
    _metadata_cache_signature = signature
    return _metadata_cache


def save_metadata_entry(folder_name: str, entry: dict) -> None:
    """Add or update a single entry in test_metadata.json."""
    global _metadata_cache
    global _metadata_cache_signature

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
    _metadata_cache_signature = None


def load_user_patterns() -> list[str]:
    """Load user-defined sweep regex patterns (raw strings)."""
    global _user_patterns_cache
    global _user_patterns_cache_signature

    signature = user_patterns_storage_signature()
    if _user_patterns_cache is not None and _user_patterns_cache_signature == signature:
        return _user_patterns_cache

    if _USER_PATTERNS_PATH.exists():
        with open(_USER_PATTERNS_PATH, "r") as f:
            _user_patterns_cache = json.load(f)
    else:
        _user_patterns_cache = []
    _user_patterns_cache_signature = signature
    return _user_patterns_cache


def save_user_patterns(patterns: list[str]) -> None:
    """Save user-defined sweep regex patterns."""
    global _user_patterns_cache
    global _user_patterns_cache_signature

    with open(_USER_PATTERNS_PATH, "w") as f:
        json.dump(patterns, f, indent=2)
    _user_patterns_cache = None
    _user_patterns_cache_signature = None


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


def _annotate_frequency_holds(
    sweep_df: pd.DataFrame,
    *,
    freq_col: str = "Frequency",
) -> pd.DataFrame:
    """Flag unusually long constant-frequency runs as hold segments."""
    out = sweep_df.copy()
    out["FrequencyRunId"] = np.arange(len(out), dtype=int)
    out["FrequencyRunPoints"] = 1
    out["IsFrequencyHold"] = False

    if out.empty or freq_col not in out.columns:
        return out

    freq = pd.to_numeric(out[freq_col], errors="coerce").round(6)
    if freq.notna().sum() < 2:
        return out

    run_id = freq.ne(freq.shift()).cumsum()
    run_sizes = run_id.value_counts().sort_index()
    repeated_sizes = run_sizes[run_sizes > 1]
    typical_run_points = (
        float(repeated_sizes.median())
        if not repeated_sizes.empty
        else float(run_sizes.median())
    )
    if not np.isfinite(typical_run_points) or typical_run_points <= 0:
        return out

    hold_threshold = max(
        float(_MIN_FREQUENCY_HOLD_POINTS),
        typical_run_points * _FREQUENCY_HOLD_RUN_MULTIPLIER,
    )
    hold_ids = set(run_sizes[run_sizes >= hold_threshold].index.tolist())

    out["FrequencyRunId"] = run_id.astype(int)
    out["FrequencyRunPoints"] = run_id.map(run_sizes).astype(int)
    if hold_ids:
        out["IsFrequencyHold"] = out["FrequencyRunId"].isin(hold_ids)

    return out


def prepare_time_series_data(
    df: pd.DataFrame,
    time_col: str,
    signal_col: str,
    parse_time: bool = True,
    drop_na: bool = DROPNA_DEFAULT,
    time_format: str | None = None,
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
    resolved_time_format = time_format or detect_time_format(df, time_col)

    # Parse time ONLY if it's an absolute timestamp
    if parse_time and resolved_time_format == "absolute_timestamp":
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
    time_format: str | None = None,
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
    resolved_time_format = time_format or detect_time_format(source_df, time_col)
    if parse_time and resolved_time_format == "absolute_timestamp":
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
    if resolved_time_format == "elapsed_seconds":
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

    return _annotate_frequency_holds(sweep_df)


def bin_by_frequency(
    df: pd.DataFrame,
    value_col: str,
    freq_col: str = "Frequency",
    bin_hz: float = 5.0,
    max_bins: int = 10000,
    exclude_frequency_holds: bool = True,
) -> pd.DataFrame:
    """Bin frequency sweep data and compute mean ± std per bin."""
    cols = [freq_col, value_col]
    if exclude_frequency_holds and "IsFrequencyHold" in df.columns:
        cols.append("IsFrequencyHold")

    out = df[cols].copy()
    if exclude_frequency_holds and "IsFrequencyHold" in out.columns:
        out = out.loc[~out["IsFrequencyHold"]].drop(columns=["IsFrequencyHold"])
    out = out.dropna().copy()

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


def _infer_recommendation_value_col(
    frames: list[pd.DataFrame],
    value_col: str | None,
) -> str | None:
    """Resolve the signal column used for automatic bin recommendations."""
    if value_col:
        return value_col

    for frame in frames:
        if frame.empty:
            continue
        numeric_cols = frame.select_dtypes(include=["number"]).columns.tolist()
        for col in numeric_cols:
            if col.lower() not in _RECOMMENDATION_RESERVED_COLUMNS:
                return col
    return None


def _extract_overlay_series(
    frames: list[pd.DataFrame],
    *,
    value_col: str,
    freq_col: str,
    sweep_col: str,
    split_by_sweep: bool,
) -> list[pd.DataFrame]:
    """Build the overlaid series used when scoring candidate bin widths."""
    overlay_series: list[pd.DataFrame] = []

    for frame in frames:
        if frame.empty or freq_col not in frame.columns or value_col not in frame.columns:
            continue

        working = frame.copy()
        if "IsFrequencyHold" in working.columns:
            working = working.loc[~working["IsFrequencyHold"]].copy()
        if working.empty:
            continue

        groups: list[pd.DataFrame]
        if split_by_sweep and sweep_col in working.columns and working[sweep_col].nunique() > 1:
            groups = [sub.copy() for _, sub in working.groupby(sweep_col, sort=True)]
        else:
            groups = [working]

        for group in groups:
            series = group[[freq_col, value_col]].copy()
            series[freq_col] = pd.to_numeric(series[freq_col], errors="coerce")
            series[value_col] = pd.to_numeric(series[value_col], errors="coerce")
            series = series.loc[
                np.isfinite(series[freq_col]) & np.isfinite(series[value_col])
            ].copy()
            if series.empty or series[freq_col].nunique() < _RECOMMENDATION_ROLLING_WINDOW:
                continue

            series = (
                series.groupby(freq_col, as_index=False)[value_col]
                .mean()
                .sort_values(freq_col)
                .reset_index(drop=True)
            )
            if len(series) >= _RECOMMENDATION_ROLLING_WINDOW:
                overlay_series.append(series)

    return overlay_series


def _average_overlay_series(
    overlay_series: list[pd.DataFrame],
    *,
    value_col: str,
    freq_col: str,
    bin_hz: float,
) -> pd.DataFrame:
    """Average several overlaid series onto one common frequency grid."""
    if len(overlay_series) < 2:
        return pd.DataFrame()

    binned_series: list[pd.DataFrame] = []
    all_freqs: list[float] = []
    for series in overlay_series:
        try:
            binned = bin_by_frequency(
                series,
                value_col=value_col,
                freq_col=freq_col,
                bin_hz=bin_hz,
                exclude_frequency_holds=False,
            )
        except ValueError:
            continue
        if binned.empty:
            continue
        binned_series.append(binned)
        all_freqs.extend(binned["freq_center"].tolist())

    if len(binned_series) < 2 or not all_freqs:
        return pd.DataFrame()

    fmin, fmax = min(all_freqs), max(all_freqs)
    if fmax <= fmin:
        return pd.DataFrame()

    edges = np.arange(fmin, fmax + bin_hz, bin_hz)
    if len(edges) < 2:
        return pd.DataFrame()
    centers = (edges[:-1] + edges[1:]) / 2.0

    avg_vals: list[float] = []
    std_vals: list[float] = []
    half = bin_hz / 2.0
    for center in centers:
        vals: list[float] = []
        for binned in binned_series:
            mask = (
                (binned["freq_center"] >= center - half)
                & (binned["freq_center"] < center + half)
            )
            matched = binned.loc[mask, "mean"]
            if not matched.empty:
                vals.append(float(matched.mean()))
        if vals:
            avg_vals.append(float(np.mean(vals)))
            std_vals.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        else:
            avg_vals.append(np.nan)
            std_vals.append(np.nan)

    return (
        pd.DataFrame({"freq_center": centers, "mean": avg_vals, "std": std_vals})
        .dropna()
        .reset_index(drop=True)
    )


def _recommend_bin_for_series(
    overlay_series: list[pd.DataFrame],
    *,
    value_col: str,
    freq_col: str,
    min_bin_hz: float,
    max_bin_hz: float,
    step_hz: float,
) -> dict[str, float] | None:
    """Pick the smallest bin width that removes isolated mismatch jumps."""
    if len(overlay_series) < 2:
        return None

    pooled_values = pd.concat(
        [series[value_col] for series in overlay_series],
        ignore_index=True,
    )
    if pooled_values.empty:
        return None

    signal_span = float(
        np.nanpercentile(pooled_values, 95) - np.nanpercentile(pooled_values, 5)
    )
    target_rolling_std = max(
        1.0,
        signal_span * _RECOMMENDATION_ROLLING_STD_TARGET_FRACTION,
    )
    target_isolated_jump = max(
        1.0,
        signal_span * _RECOMMENDATION_ISOLATED_JUMP_TARGET_FRACTION,
    )

    best_candidate: dict[str, float] | None = None
    for candidate in np.arange(min_bin_hz, max_bin_hz + (step_hz / 2.0), step_hz):
        candidate = float(round(candidate, 6))
        avg_df = _average_overlay_series(
            overlay_series,
            value_col=value_col,
            freq_col=freq_col,
            bin_hz=candidate,
        )
        if len(avg_df) < _RECOMMENDATION_ROLLING_WINDOW:
            continue

        rolling_std = (
            avg_df["mean"]
            .rolling(
                window=_RECOMMENDATION_ROLLING_WINDOW,
                min_periods=_RECOMMENDATION_ROLLING_WINDOW,
            )
            .std()
            .dropna()
        )
        if rolling_std.empty:
            continue

        roughness = float(
            np.nanpercentile(rolling_std, _RECOMMENDATION_ROUGHNESS_PERCENTILE)
        )
        mean_values = avg_df["mean"].to_numpy(dtype=float)
        centered_residual = np.abs(
            mean_values[1:-1]
            - ((mean_values[:-2] + mean_values[2:]) / 2.0)
        )
        left_step = mean_values[1:-1] - mean_values[:-2]
        right_step = mean_values[2:] - mean_values[1:-1]
        reversal_mask = (left_step * right_step) < 0
        residual_baseline = (
            float(np.nanmedian(centered_residual))
            if centered_residual.size
            else 0.0
        )
        isolated_jump_cutoff = max(
            target_isolated_jump,
            residual_baseline * _RECOMMENDATION_ISOLATED_JUMP_RATIO,
        )
        isolated_jump_mask = reversal_mask & (centered_residual > isolated_jump_cutoff)
        isolated_jump_count = int(np.sum(isolated_jump_mask))
        isolated_jump_residual = (
            float(np.max(centered_residual[isolated_jump_mask]))
            if isolated_jump_count
            else (
                float(np.max(centered_residual))
                if centered_residual.size
                else 0.0
            )
        )
        isolated_jump_score = (
            float(isolated_jump_count)
            + float(
                np.maximum(
                    (centered_residual[isolated_jump_mask] / isolated_jump_cutoff) - 1.0,
                    0.0,
                ).sum()
            )
            if isolated_jump_count
            else 0.0
        )
        score = (
            isolated_jump_score
            + max(0.0, (roughness / target_rolling_std) - 1.0)
            + _RECOMMENDATION_BIN_WIDTH_PENALTY * (candidate / max_bin_hz)
        )
        candidate_result = {
            "bin_hz": candidate,
            "rolling_std": roughness,
            "target_rolling_std": float(target_rolling_std),
            "isolated_jump_residual": float(isolated_jump_residual),
            "target_isolated_jump_residual": float(target_isolated_jump),
            "isolated_jump_count": float(isolated_jump_count),
            "series_count": float(len(overlay_series)),
            "score": float(score),
        }
        if best_candidate is None or candidate_result["score"] < best_candidate["score"]:
            best_candidate = candidate_result
        if (
            isolated_jump_count == 0
            and roughness <= target_rolling_std
        ):
            return candidate_result

    if best_candidate is None:
        return None
    return best_candidate


def recommend_frequency_bin_widths(
    sweep_frames: list[pd.DataFrame] | dict[str, pd.DataFrame],
    *,
    value_col: str | None = None,
    freq_col: str = "Frequency",
    sweep_col: str = "Sweep",
    min_bin_hz: float = 0.5,
    max_bin_hz: float = 100.0,
    step_hz: float = 0.5,
) -> dict[str, float]:
    """Recommend bin widths for overlaid sweep/test averages.

    The scoring rule prefers the smallest bin width that removes isolated
    one-bin jump artifacts on the averaged curve while also keeping the
    5-point rolling standard deviation below a target tied to the signal
    range. Wider bins are explicitly penalized to avoid unnecessary loss
    of frequency detail.
    """
    if isinstance(sweep_frames, dict):
        frames = list(sweep_frames.values())
    else:
        frames = list(sweep_frames)

    resolved_value_col = _infer_recommendation_value_col(frames, value_col)

    def _snap(value: float) -> float:
        return round(max(min_bin_hz, min(max_bin_hz, value)) / step_hz) * step_hz

    positive_steps: list[float] = []
    start_freqs: list[float] = []

    for frame in frames:
        if frame.empty or freq_col not in frame.columns:
            continue

        working = frame.copy()
        if "IsFrequencyHold" in working.columns:
            working = working.loc[~working["IsFrequencyHold"]].copy()
        if working.empty:
            continue

        freqs = pd.to_numeric(working[freq_col], errors="coerce").to_numpy(dtype=float)
        freqs = freqs[np.isfinite(freqs)]
        if freqs.size == 0:
            continue

        unique_freqs = np.unique(np.round(freqs, 6))
        if unique_freqs.size > 1:
            diffs = np.diff(np.sort(unique_freqs))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                positive_steps.extend(diffs.tolist())

        if sweep_col in working.columns:
            starts = (
                working.groupby(sweep_col, sort=True)[freq_col]
                .first()
                .pipe(pd.to_numeric, errors="coerce")
                .dropna()
                .astype(float)
                .tolist()
            )
            start_freqs.extend(starts)
        else:
            start_freqs.append(float(freqs[0]))

    typical_step_hz = (
        float(np.median(positive_steps))
        if positive_steps
        else max(min_bin_hz, step_hz)
    )
    sorted_starts = np.sort(np.asarray(start_freqs, dtype=float)) if start_freqs else np.array([])
    start_gaps = (
        np.diff(sorted_starts)
        if sorted_starts.size >= 2
        else np.array([])
    )
    start_alignment_gap_hz = (
        float(np.max(start_gaps))
        if start_gaps.size
        else 0.0
    )
    start_spread_hz = (
        float(np.percentile(start_freqs, 90) - np.percentile(start_freqs, 10))
        if len(start_freqs) >= 2
        else 0.0
    )

    fallback_bin_hz = _snap(max(min_bin_hz, step_hz))
    test_series = (
        _extract_overlay_series(
            frames,
            value_col=resolved_value_col,
            freq_col=freq_col,
            sweep_col=sweep_col,
            split_by_sweep=True,
        )
        if resolved_value_col
        else []
    )
    average_series = (
        _extract_overlay_series(
            frames,
            value_col=resolved_value_col,
            freq_col=freq_col,
            sweep_col=sweep_col,
            split_by_sweep=False,
        )
        if resolved_value_col
        else []
    )

    test_reco = (
        _recommend_bin_for_series(
            test_series,
            value_col=resolved_value_col,
            freq_col=freq_col,
            min_bin_hz=min_bin_hz,
            max_bin_hz=max_bin_hz,
            step_hz=step_hz,
        )
        if resolved_value_col
        else None
    )
    average_reco = (
        _recommend_bin_for_series(
            average_series,
            value_col=resolved_value_col,
            freq_col=freq_col,
            min_bin_hz=min_bin_hz,
            max_bin_hz=max_bin_hz,
            step_hz=step_hz,
        )
        if resolved_value_col
        else None
    )

    test_bin_hz = (
        float(_snap(test_reco["bin_hz"]))
        if test_reco is not None
        else float(fallback_bin_hz)
    )
    average_bin_hz = (
        float(_snap(average_reco["bin_hz"]))
        if average_reco is not None
        else float(test_bin_hz)
    )
    average_bin_hz = float(max(test_bin_hz, average_bin_hz))

    return {
        "test_bin_hz": float(max(min_bin_hz, min(max_bin_hz, test_bin_hz))),
        "average_bin_hz": float(max(min_bin_hz, min(max_bin_hz, average_bin_hz))),
        "typical_step_hz": float(typical_step_hz),
        "start_alignment_gap_hz": float(start_alignment_gap_hz),
        "start_spread_hz": float(start_spread_hz),
        "test_series_count": float(len(test_series)),
        "average_series_count": float(len(average_series)),
        "test_target_rolling_std": float(
            test_reco["target_rolling_std"] if test_reco is not None else 0.0
        ),
        "average_target_rolling_std": float(
            average_reco["target_rolling_std"] if average_reco is not None else 0.0
        ),
        "test_rolling_std": float(
            test_reco["rolling_std"] if test_reco is not None else 0.0
        ),
        "average_rolling_std": float(
            average_reco["rolling_std"] if average_reco is not None else 0.0
        ),
        "test_isolated_jump_residual": float(
            test_reco["isolated_jump_residual"] if test_reco is not None else 0.0
        ),
        "average_isolated_jump_residual": float(
            average_reco["isolated_jump_residual"] if average_reco is not None else 0.0
        ),
        "test_target_isolated_jump_residual": float(
            test_reco["target_isolated_jump_residual"] if test_reco is not None else 0.0
        ),
        "average_target_isolated_jump_residual": float(
            average_reco["target_isolated_jump_residual"] if average_reco is not None else 0.0
        ),
        "test_isolated_jump_count": float(
            test_reco["isolated_jump_count"] if test_reco is not None else 0.0
        ),
        "average_isolated_jump_count": float(
            average_reco["isolated_jump_count"] if average_reco is not None else 0.0
        ),
    }


def explain_frequency_bin_recommendation(
    recommendation: dict[str, float],
    *,
    include_average_bin: bool = True,
) -> str:
    """Format a human-readable explanation of the bin recommendation."""
    parts = [f"typical frequency step = {recommendation['typical_step_hz']:.1f} Hz"]

    test_series_count = int(recommendation.get("test_series_count", 0))
    if test_series_count >= 2:
        parts.append(
            f"{test_series_count} overlaid sweep/test series; "
            f"isolated jump count = {recommendation.get('test_isolated_jump_count', 0.0):.0f}; "
            f"5-point roughness = {recommendation.get('test_rolling_std', 0.0):.1f} "
            f"target {recommendation.get('test_target_rolling_std', 0.0):.1f} µL/min; "
            f"recommended plot bin = {recommendation['test_bin_hz']:.1f} Hz"
        )
    else:
        parts.append(f"recommended plot bin = {recommendation['test_bin_hz']:.1f} Hz")

    if include_average_bin:
        average_series_count = int(recommendation.get("average_series_count", 0))
        if average_series_count >= 2:
            parts.append(
                f"{average_series_count} overlaid test series for averaging; "
                f"isolated jump count = {recommendation.get('average_isolated_jump_count', 0.0):.0f}; "
                f"5-point roughness = {recommendation.get('average_rolling_std', 0.0):.1f} "
                f"target {recommendation.get('average_target_rolling_std', 0.0):.1f} µL/min; "
                f"recommended average bin = {recommendation['average_bin_hz']:.1f} Hz"
            )
        else:
            parts.append(
                f"recommended average bin = {recommendation['average_bin_hz']:.1f} Hz"
            )

    return "; ".join(parts) + "."


def format_bin_choice_label(
    selected_bin_hz: float,
    recommendation: dict[str, float] | None = None,
    *,
    use_average_bin: bool = False,
) -> str:
    """Format a short label showing selected vs recommended bin width."""
    label = f"selected Δf = {selected_bin_hz:g} Hz"
    if recommendation:
        reco_key = "average_bin_hz" if use_average_bin else "test_bin_hz"
        recommended = float(recommendation[reco_key])
        if abs(recommended - float(selected_bin_hz)) < 1e-9:
            label += " · matches recommended"
        else:
            label += f" · recommended Δf = {recommended:g} Hz"
    return label


def summarize_frequency_holds(
    sweep_df: pd.DataFrame,
    *,
    signal_col: str,
) -> pd.DataFrame:
    """Summarize constant-frequency hold segments removed from sweep plots."""
    if (
        sweep_df.empty
        or "IsFrequencyHold" not in sweep_df.columns
        or not sweep_df["IsFrequencyHold"].any()
        or "FrequencyRunId" not in sweep_df.columns
        or signal_col not in sweep_df.columns
    ):
        return pd.DataFrame()

    hold_df = sweep_df.loc[sweep_df["IsFrequencyHold"]].copy()
    summary = (
        hold_df.groupby("FrequencyRunId", sort=True)
        .agg(
            sweep=("Sweep", "first"),
            frequency_hz=("Frequency", "first"),
            points=("FrequencyRunPoints", "first"),
            start_s=("Elapsed_s", "min"),
            end_s=("Elapsed_s", "max"),
            mean_signal=(signal_col, "mean"),
            std_signal=(signal_col, "std"),
        )
        .reset_index(drop=True)
    )
    summary["duration_s"] = summary["end_s"] - summary["start_s"]
    summary["std_signal"] = summary["std_signal"].fillna(0)
    return summary.rename(
        columns={
            "sweep": "Sweep",
            "frequency_hz": "Frequency (Hz)",
            "points": "Points",
            "start_s": "Start (s)",
            "end_s": "End (s)",
            "duration_s": "Duration (s)",
            "mean_signal": f"Mean {signal_col}",
            "std_signal": f"Std {signal_col}",
        }
    )


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
