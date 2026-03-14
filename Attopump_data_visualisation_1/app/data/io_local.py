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
import sys


_DARWIN_UF_DATALESS = 0x40000000
_AUXILIARY_CSV_NAMES = {"events.csv"}


@dataclass(frozen=True)
class CsvPick:
    """A resolved CSV selection inside a run folder."""

    run_dir: Path
    csv_path: Path


def _is_cloud_placeholder_file(p: Path) -> bool:
    """Return whether *p* is a macOS cloud placeholder with no local data."""
    if sys.platform != "darwin":
        return False
    try:
        return bool(p.stat().st_flags & _DARWIN_UF_DATALESS)
    except (AttributeError, OSError):
        return False


def _is_auxiliary_csv(p: Path) -> bool:
    """Return whether *p* is a non-analysis CSV such as events.csv."""
    return p.name.lower() in _AUXILIARY_CSV_NAMES


def _is_nonempty_file(p: Path, *, require_local: bool = False) -> bool:
    try:
        if not p.is_file():
            return False
        if require_local and _is_cloud_placeholder_file(p):
            return False
        return p.stat().st_size > 0
    except OSError:
        return False


def _format_cloud_placeholder_error(p: Path) -> str:
    """Return a user-facing message for cloud-only placeholder files."""
    return (
        f"CSV is not downloaded locally: {p}. "
        "This looks like a cloud-only OneDrive/iCloud placeholder. "
        "Mark the file or its parent folder as 'Always Keep on This Device' and retry."
    )


def _raise_if_unavailable_csv(p: Path) -> None:
    """Raise a clear error when a CSV cannot be read from local disk."""
    if not p.exists():
        raise FileNotFoundError(f"CSV does not exist: {p}")
    if _is_cloud_placeholder_file(p):
        raise OSError(_format_cloud_placeholder_error(p))


def _read_delimiter_from_header(p: Path) -> str:
    """Detect the CSV delimiter from the first line."""
    _raise_if_unavailable_csv(p)
    try:
        with open(p, "r", encoding="utf-8") as f:
            first_line = f.readline()
    except TimeoutError as exc:
        raise OSError(_format_cloud_placeholder_error(p)) from exc

    return ";" if first_line.count(";") > first_line.count(",") else ","


def _csv_discovery_error(run_dir: Path) -> FileNotFoundError:
    """Build a clear data-discovery error for a run directory."""
    skipped_placeholders: list[str] = []
    for p in sorted(run_dir.glob("*.csv")):
        if not p.is_file() or _is_auxiliary_csv(p):
            continue
        if _is_cloud_placeholder_file(p):
            skipped_placeholders.append(p.name)

    if skipped_placeholders:
        skipped_list = ", ".join(skipped_placeholders)
        return FileNotFoundError(
            f"No locally available data CSV files found in: {run_dir}. "
            f"Cloud-only placeholders skipped: {skipped_list}. "
            "Mark the folder or files as 'Always Keep on This Device' and retry."
        )

    return FileNotFoundError(f"No non-empty data CSV files found in: {run_dir}")


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
    """Return non-empty, locally available analysis CSVs in a run directory."""
    d = Path(run_dir)
    if not d.exists():
        raise FileNotFoundError(f"Run directory does not exist: {d}")
    if not d.is_dir():
        raise NotADirectoryError(f"Run directory is not a directory: {d}")

    csvs: list[Path] = []
    for p in sorted(d.glob("*.csv")):
        if _is_auxiliary_csv(p):
            continue
        if _is_nonempty_file(p, require_local=True):
            csvs.append(p)
    return csvs


def pick_best_csv(run_dir: str | Path) -> CsvPick:
    """Pick a single best CSV from a run folder.

    Format detection:
    1) If a local 'merged.csv' exists → use it (Format 1: merged data)
    2) Else prefer local 'trimmed_*.csv' files (Format 2: raw Flowboard data)
    3) Else prefer local 'flow.csv'
    4) Else first local non-empty data CSV in name-sorted order
    """
    d = Path(run_dir)

    # Check for merged.csv first (Format 1)
    merged_csv = d / "merged.csv"
    if _is_nonempty_file(merged_csv, require_local=True):
        return CsvPick(run_dir=d, csv_path=merged_csv)

    # Try trimmed_*.csv files (Format 2: raw Flowboard data)
    trimmed = [
        p for p in sorted(d.glob("trimmed_*.csv")) if _is_nonempty_file(p, require_local=True)
    ]
    if trimmed:
        return CsvPick(run_dir=d, csv_path=trimmed[0])

    # Some runs store the raw Flowboard export as flow.csv
    flow_csv = d / "flow.csv"
    if _is_nonempty_file(flow_csv, require_local=True):
        return CsvPick(run_dir=d, csv_path=flow_csv)

    # Fall back to first CSV found
    csvs = find_csvs(d)
    if not csvs:
        raise _csv_discovery_error(d)
    return CsvPick(run_dir=d, csv_path=csvs[0])


def read_csv_preview(csv_path: str | Path, nrows: int = 200) -> "pd.DataFrame":
    """Read a small preview of a CSV (keeps UI snappy).
    
    Handles both comma and semicolon delimited formats.
    """
    p = Path(csv_path)
    delimiter = _read_delimiter_from_header(p)

    import pandas as pd

    try:
        df = pd.read_csv(p, delimiter=delimiter, nrows=nrows)
    except TimeoutError as exc:
        raise OSError(_format_cloud_placeholder_error(p)) from exc

    # Standardize column names to lowercase and strip whitespace
    df.columns = df.columns.str.strip().str.lower()

    return df


def read_csv_full(csv_path: str | Path) -> "pd.DataFrame":
    """Read the full CSV.
    
    Handles both comma and semicolon delimited formats:
    - Format 1: t_s,freq_set_hz,duty_set,flow (comma-separated, "merged" files)
    - Format 2: Time;Flow Unit #1 [...];... (semicolon-separated with timestamps)
    """
    p = Path(csv_path)
    delimiter = _read_delimiter_from_header(p)

    import pandas as pd

    try:
        df = pd.read_csv(p, delimiter=delimiter)
    except TimeoutError as exc:
        raise OSError(_format_cloud_placeholder_error(p)) from exc

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
