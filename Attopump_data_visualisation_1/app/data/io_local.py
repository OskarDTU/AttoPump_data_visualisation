"""
Local OneDrive (synced folder) access utilities.

This module provides low-level file-system helpers for discovering and
reading test-data CSV files stored in a locally-synced OneDrive folder.
It is intentionally **UI-agnostic** so it can be reused by Streamlit
pages, tests, and CLI scripts without importing Streamlit.

Strategy
--------
OneDrive is treated as a regular local directory tree.  Each immediate
subfolder under the user-supplied root represents one *test run*.  Inside
each test-run folder the module locates the best CSV to analyse.

Key functions
-------------
- ``normalize_root(path)`` → validated ``Path`` (handles macOS escaped spaces).
- ``list_run_dirs(root)``  → sorted list of test-run ``Path`` objects.
- ``find_csvs(run_dir)``   → all non-empty CSVs in a run folder.
- ``pick_best_csv(run_dir)`` → single ``CsvPick`` (prefers merged.csv > trimmed_*.csv).
- ``read_csv_preview(csv, nrows)`` → small ``pd.DataFrame`` for UI preview.
- ``read_csv_full(csv)``   → complete ``pd.DataFrame`` with auto-delimiter detection.

Inputs
------
- A local folder path (``str`` or ``Path``) pointing at the OneDrive root.
- CSV files in either **comma-delimited** (merged.csv) or **semicolon-
  delimited** (Flowboard raw export) format.

Outputs
-------
- ``Path`` objects for directories / files.
- ``pd.DataFrame`` with **lowercase column names** (always normalised).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CsvPick:
    """A resolved CSV selection inside a run folder."""

    run_dir: Path
    csv_path: Path


def _is_nonempty_file(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def normalize_root(root: str | Path) -> Path:
    """Normalize and validate the OneDrive-synced root path.
    
    Handles escaped spaces from macOS terminal drag-and-drop (e.g., "path\\ with\\ spaces").
    """
    # Convert to string if Path object
    root_str = str(root)
    
    # Handle escaped spaces from macOS terminal drag-and-drop
    root_str = root_str.replace("\\ ", " ")
    
    p = Path(root_str).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Path does not exist: {p}\n\n"
            f"Please check the path. Note: paths with spaces should work, but verify the folder exists."
        )
    if not p.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {p}")
    return p


def list_run_dirs(
    root: str | Path,
    relative: str | Path = ".",
    *,
    include_hidden: bool = False,
) -> list[Path]:
    """List immediate subdirectories that *exist on disk* under (root / relative).

    This simply reads the OneDrive-synced local folder structure. It does not depend on,
    or reuse, any list produced by the legacy analysis script.
    """
    base = normalize_root(root) / Path(relative)
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")

    dirs: list[Path] = []
    for p in sorted(base.iterdir()):
        if not include_hidden and p.name.startswith("."):
            continue
        if p.is_dir():
            dirs.append(p)
    return dirs


def find_csvs(run_dir: str | Path) -> list[Path]:
    """Return non-empty CSV files that *exist on disk* in a run directory (non-recursive)."""
    d = Path(run_dir)
    if not d.exists():
        raise FileNotFoundError(f"Run directory does not exist: {d}")
    if not d.is_dir():
        raise NotADirectoryError(f"Run directory is not a directory: {d}")

    return [p for p in sorted(d.glob("*.csv")) if _is_nonempty_file(p)]


def pick_best_csv(run_dir: str | Path) -> CsvPick:
    """Pick a single best CSV from a run folder.

    Format detection:
    1) If 'merged.csv' exists → use it (Format 1: merged data)
    2) Else prefer 'trimmed_*.csv' files (Format 2: raw Flowboard data)
    3) Else first non-empty '*.csv' in name-sorted order
    """
    d = Path(run_dir)

    # Check for merged.csv first (Format 1)
    merged_csv = d / "merged.csv"
    if _is_nonempty_file(merged_csv):
        return CsvPick(run_dir=d, csv_path=merged_csv)

    # Try trimmed_*.csv files (Format 2: raw Flowboard data)
    trimmed = [p for p in sorted(d.glob("trimmed_*.csv")) if _is_nonempty_file(p)]
    if trimmed:
        return CsvPick(run_dir=d, csv_path=trimmed[0])

    # Fall back to first CSV found
    csvs = find_csvs(d)
    if not csvs:
        raise FileNotFoundError(f"No non-empty CSV files found in: {d}")
    return CsvPick(run_dir=d, csv_path=csvs[0])


def read_csv_preview(csv_path: str | Path, nrows: int = 200) -> "pd.DataFrame":
    """Read a small preview of a CSV (keeps UI snappy).
    
    Handles both comma and semicolon delimited formats.
    """
    import pandas as pd

    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV does not exist: {p}")
    
    # Try to detect the delimiter by reading first line
    with open(p, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    
    # Determine delimiter: if more semicolons than commas in first line, use semicolon
    delimiter = ';' if first_line.count(';') > first_line.count(',') else ','
    
    df = pd.read_csv(p, delimiter=delimiter, nrows=nrows)
    
    # Standardize column names to lowercase and strip whitespace
    df.columns = df.columns.str.strip().str.lower()
    
    return df


def read_csv_full(csv_path: str | Path) -> "pd.DataFrame":
    """Read the full CSV.
    
    Handles both comma and semicolon delimited formats:
    - Format 1: t_s,freq_set_hz,duty_set,flow (comma-separated, "merged" files)
    - Format 2: Time;Flow Unit #1 [...];... (semicolon-separated with timestamps)
    """
    import pandas as pd

    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV does not exist: {p}")
    
    # Try to detect the delimiter by reading first line
    with open(p, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    
    # Determine delimiter: if more semicolons than commas in first line, use semicolon
    delimiter = ';' if first_line.count(';') > first_line.count(',') else ','
    
    df = pd.read_csv(p, delimiter=delimiter)
    
    # Standardize column names to lowercase and strip whitespace
    df.columns = df.columns.str.strip().str.lower()
    
    return df


__all__ = [
    "CsvPick",
    "normalize_root",
    "list_run_dirs",
    "find_csvs",
    "pick_best_csv",
    "read_csv_preview",
    "read_csv_full",
]