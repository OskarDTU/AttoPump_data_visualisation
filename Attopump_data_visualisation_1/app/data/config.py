"""Configuration and constants for AttoPump data visualization.

This module is the single source of truth for all tuneable parameters,
regex patterns, column-name heuristics, and dataclasses shared across the
application.  Nothing in this file performs I/O or computation — it only
*defines* values consumed by other modules.

Sections
--------
- **Paths & defaults** — export directory, app root.
- **Regex patterns** — compiled patterns to detect frequency-sweep test
  folders from their naming convention.
- **Data column guessing** — ordered candidate lists used by
  ``data_processor.guess_time_column`` / ``guess_signal_column``.
- **Plotting defaults** — chart height, bin width, point caps.
- **Data processing** — NaN handling, duplicate strategy.
- **Constant-frequency tests** — default frequency for older tests.
- **SweepSpec dataclass** — lightweight container for parsed sweep
  parameters (start Hz, end Hz, duration).

No inputs / outputs — this module is imported for its constant values.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# PATHS & DEFAULTS
# ============================================================================
APP_DIR = Path(__file__).parent
DATA_EXPORT_DIR = APP_DIR / "exports"
DATA_EXPORT_DIR.mkdir(exist_ok=True)

# ============================================================================
# REGEX PATTERNS
# ============================================================================
# Multiple patterns tried in order to handle naming variations:
#   20260302-1037-1Hz_1500H_Hz_500_seconds  → 1 Hz to 1500 Hz, 500 s
#   20260206_162936_10x_Sweep_1Hz-1kHz_...  → 1 Hz to 1000 Hz (kHz)
#   10Hz_500Hz_60s                          → 10 Hz to 500 Hz, 60 s
SWEEP_PATTERNS = [
    # Standard: 10Hz_500Hz_60s
    re.compile(
        r"(?P<start>\d+(?:\.\d+)?)\s*Hz[_-]+(?P<end>\d+(?:\.\d+)?)\s*Hz[_-]+(?P<dur>\d+(?:\.\d+)?)\s*s\b",
        re.IGNORECASE,
    ),
    # With H_Hz typo and _seconds: 1Hz_1500H_Hz_500_seconds
    re.compile(
        r"(?P<start>\d+(?:\.\d+)?)\s*Hz[_-]+(?P<end>\d+(?:\.\d+)?)\s*H_Hz[_-]+(?P<dur>\d+(?:\.\d+)?)_?(?:seconds?|s)\b",
        re.IGNORECASE,
    ),
    # With kHz: 1Hz-1kHz (no duration in name)
    re.compile(
        r"(?P<start>\d+(?:\.\d+)?)\s*Hz[_-]+(?P<end>\d+(?:\.\d+)?)\s*kHz",
        re.IGNORECASE,
    ),
]
# Keep single SWEEP_PATTERN for backward compat (uses first pattern)
SWEEP_PATTERN = SWEEP_PATTERNS[0]

# ============================================================================
# DATA COLUMN GUESSING
# ============================================================================
TIME_COLUMN_CANDIDATES = [
    "time", "t_s", "timestamp", "datetime",  # Lowercase for standardized columns
    "Time", "time", "timestamp", "Timestamp", "DateTime", "datetime",
    "elapsed_time", "Elapsed_Time", "t", "T"
]

SIGNAL_COLUMN_HEURISTICS = {
    "flow": [
        # Format 1 (merged.csv): perfectly labeled
        "flow",
        # Format 2 (Flowboard raw data): match columns that contain "flow" and measurement data
        # Common patterns - check contains instead of exact match
        "flow unit #1 [flowboard (1384)]",
        "flow unit #1",
        # Fallback: any column with "flow" in name and containing measurement (not setpoint/air)
        # This will be handled by partial matching in the guess function
    ],
    "pressure": [
        "pressure", "press", "p",  # Standardized
        "Pressure", "pressure", "Press", "press", "P"
    ],
    "temperature": [
        "temperature", "temp", "t",  # Standardized
        "Temperature", "temperature", "Temp", "temp", "T"
    ],
}

# ============================================================================
# PLOTTING DEFAULTS
# ============================================================================
PLOT_HEIGHT = 600
PLOT_BIN_WIDTH_HZ = 5.0
MAX_POINTS_DEFAULT = 200000
DOWNSAMPLE_THRESHOLD = 50000

# ============================================================================
# DATA PROCESSING
# ============================================================================
TIME_PARSE_ERROR_HANDLING = "coerce"  # Drop unparseable timestamps
DUPLICATE_HANDLING = "first"  # Keep first occurrence of duplicates
DROPNA_DEFAULT = True

# ============================================================================
# CONSTANT FREQUENCY TESTS (before 2026-03-01)
# ============================================================================
CONSTANT_FREQUENCY_CUTOFF = "2026-03-01"
DEFAULT_CONSTANT_FREQUENCY_HZ = 500.0  # All tests before cutoff are 500Hz

# ============================================================================
# SWEEP SPECS DATACLASS
# ============================================================================
@dataclass(frozen=True)
class SweepSpec:
    """Parsed frequency sweep specification.

    Extracted from a test-folder name by :func:`data_processor.parse_sweep_spec_from_name`.

    Attributes
    ----------
    start_hz : float
        Lower bound of the sweep range (e.g. 1.0).
    end_hz : float
        Upper bound of the sweep range (e.g. 1500.0).
    duration_s : float
        Duration of one complete sweep cycle in seconds.
        May be 0.0 if the naming convention omits this value.
    num_repeats : int
        Number of sweep repetitions parsed from the folder name
        (e.g. ``10x`` → 10).  Defaults to 1 when not specified.
    """
    start_hz: float
    end_hz: float
    duration_s: float
    num_repeats: int = 1

    def __str__(self) -> str:
        reps = f", {self.num_repeats}×" if self.num_repeats > 1 else ""
        dur = f", {self.duration_s:g}s" if self.duration_s > 0 else ""
        return f"{self.start_hz:g}→{self.end_hz:g} Hz{dur}{reps}"
