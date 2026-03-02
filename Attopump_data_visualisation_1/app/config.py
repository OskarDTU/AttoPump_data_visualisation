"""Configuration and constants for AttoPump data visualization."""

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
SWEEP_PATTERN = re.compile(
    r"(?P<start>\d+(?:\.\d+)?)Hz_(?P<end>\d+(?:\.\d+)?)Hz_(?P<dur>\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)

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
    """Parsed frequency sweep specification."""
    start_hz: float
    end_hz: float
    duration_s: float

    def __str__(self) -> str:
        return f"{self.start_hz:g}→{self.end_hz:g} Hz, {self.duration_s:g}s"
