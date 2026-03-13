"""Experiment log loading and test metadata extraction.

This module reads the shared experiment logbook that lives alongside the
``All_tests`` directory and exposes a small, cached lookup API for the
rest of the app.

The log file is treated as a supplementary metadata source that can
provide:

- BAR / pump ID
- raw test-type description
- inferred constant/sweep classification
- constant frequency or sweep parameters parsed from free text
- date/time/author/voltage/result/notes

Rows may contain multiple test IDs in one cell, so the parser splits
those into one entry per test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd


EXPERIMENT_LOG_FILENAME = "Experiment logs_new format.xlsx"
_LOG_RELATIVE_PARTS = ("Logs", "Experiment logs", EXPERIMENT_LOG_FILENAME)


@dataclass(frozen=True)
class ExperimentLogEntry:
    """Structured metadata for one test folder from the experiment log."""

    folder: str
    normalized_folder: str
    row_number: int
    pump_bar_id: str = ""
    raw_test_type: str = ""
    inferred_test_type: str = "unknown"
    frequency_hz: float | None = None
    sweep_start_hz: float | None = None
    sweep_end_hz: float | None = None
    duration_s: float | None = None
    date: str = ""
    time: str = ""
    author: str = ""
    voltage: str = ""
    success: str = ""
    data_note: str = ""
    note: str = ""
    explanation: str = ""

    @property
    def combined_notes(self) -> str:
        """Return non-empty note fields as a single string."""
        parts = [self.data_note, self.note, self.explanation]
        return " | ".join(part for part in parts if part)


def normalize_test_identifier(value: str) -> str:
    """Normalize a test identifier for matching."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def infer_test_type_from_text(text: str) -> str:
    """Infer the app's coarse test type from free text."""
    low = text.lower().strip()
    if not low:
        return "unknown"
    if "sweep" in low or "various freq" in low or "various frequency" in low:
        return "sweep"
    if "constant" in low:
        return "constant"
    if "manual" in low and "pressure" not in low:
        return "constant"
    return "unknown"


def extract_duration_seconds(text: str) -> float | None:
    """Extract a duration in seconds from free text."""
    low = text.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|sec|s)\b", low)
    if match:
        return float(match.group(1))

    match = re.search(r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|min)\b", low)
    if match:
        return float(match.group(1)) * 60.0

    return None


def extract_constant_frequency_hz(text: str) -> float | None:
    """Extract a constant frequency from free text."""
    if infer_test_type_from_text(text) == "sweep":
        return None

    match = re.search(r"(\d+(?:\.\d+)?)\s*hz\b", text.lower())
    if match:
        return float(match.group(1))
    return None


def extract_sweep_parameters(text: str) -> tuple[float | None, float | None]:
    """Extract a sweep range from free text."""
    low = text.lower()
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*hz\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*hz\b",
        low,
    )
    if match:
        return (float(match.group(1)), float(match.group(2)))
    return (None, None)


def find_experiment_log(data_root: str | Path) -> Path | None:
    """Find the experiment log relative to a test-data root."""
    root = Path(data_root).expanduser().resolve()
    candidates: list[Path] = []

    for current in (root, *root.parents):
        candidates.append(current / Path(*_LOG_RELATIVE_PARTS))
        candidates.append(current.parent / Path(*_LOG_RELATIVE_PARTS))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def experiment_log_storage_signature(data_root: str | Path | None) -> tuple[str, int, int]:
    """Return a file signature for the experiment log near *data_root*."""
    if data_root is None:
        return ("", 0, 0)

    log_path = find_experiment_log(data_root)
    if log_path is None:
        return ("", 0, 0)

    stat = log_path.stat()
    return (str(log_path), int(stat.st_mtime_ns), int(stat.st_size))


def _clean_scalar(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ").strip()
    if text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text)


def _format_date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except Exception:
            pass
    return _clean_scalar(value)


def _format_time(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%H:%M:%S")
        except Exception:
            pass
    return _clean_scalar(value)


def _extract_test_ids(raw_value: object) -> list[str]:
    """Split one spreadsheet cell into zero or more candidate test IDs."""
    if raw_value is None or pd.isna(raw_value):
        return []

    raw_text = str(raw_value).replace("\r\n", "\n")
    parts = re.split(r"[\n;]+", raw_text)
    test_ids: list[str] = []

    for part in parts:
        cleaned = part.strip().strip(".,;:")
        if not cleaned:
            continue

        token = cleaned.split()[0].strip(".,;:")
        if token and token.lower() != "nan":
            test_ids.append(token)

    return test_ids


def _map_columns(df: pd.DataFrame) -> dict[str, str]:
    col_map: dict[str, str] = {}
    lower_cols = {col: col.lower().strip() for col in df.columns}

    for orig, low in lower_cols.items():
        if any(kw in low for kw in ["pump", "bar", "device"]):
            col_map.setdefault("pump_bar_id", orig)
        elif "test" in low and ("id" in low or "folder" in low or "name" in low):
            col_map.setdefault("test_id", orig)
        elif "test" in low and "type" in low:
            col_map.setdefault("test_type", orig)
        elif low in ("date", "dato"):
            col_map.setdefault("date", orig)
        elif low in ("time", "tid"):
            col_map.setdefault("time", orig)
        elif "author" in low or "person" in low or "operator" in low:
            col_map.setdefault("author", orig)
        elif "volt" in low:
            col_map.setdefault("voltage", orig)
        elif "success" in low or "fail" in low or "pass" in low:
            col_map.setdefault("success", orig)
        elif low == "data note":
            col_map.setdefault("data_note", orig)
        elif low == "note":
            col_map.setdefault("note", orig)
        elif "explanation" in low:
            col_map.setdefault("explanation", orig)

    return col_map


def parse_experiment_log_dataframe(df: pd.DataFrame) -> list[ExperimentLogEntry]:
    """Parse the experiment log into one entry per test ID."""
    col_map = _map_columns(df)
    if "test_id" not in col_map:
        return []

    entries: dict[str, ExperimentLogEntry] = {}

    for row_number, (_, row) in enumerate(df.iterrows(), start=2):
        test_ids = _extract_test_ids(row.get(col_map["test_id"]))
        if not test_ids:
            continue

        raw_test_type = _clean_scalar(row.get(col_map.get("test_type", "")))
        notes_text = " ".join(
            part
            for part in [
                raw_test_type,
                _clean_scalar(row.get(col_map.get("data_note", ""))),
                _clean_scalar(row.get(col_map.get("note", ""))),
                _clean_scalar(row.get(col_map.get("explanation", ""))),
            ]
            if part
        )

        inferred_type = infer_test_type_from_text(raw_test_type or notes_text)
        sweep_start_hz, sweep_end_hz = extract_sweep_parameters(raw_test_type or notes_text)

        entry_template = dict(
            row_number=row_number,
            pump_bar_id=_clean_scalar(row.get(col_map.get("pump_bar_id", ""))),
            raw_test_type=raw_test_type,
            inferred_test_type=inferred_type,
            frequency_hz=extract_constant_frequency_hz(raw_test_type or notes_text),
            sweep_start_hz=sweep_start_hz,
            sweep_end_hz=sweep_end_hz,
            duration_s=extract_duration_seconds(raw_test_type or notes_text),
            date=_format_date(row.get(col_map.get("date", ""))),
            time=_format_time(row.get(col_map.get("time", ""))),
            author=_clean_scalar(row.get(col_map.get("author", ""))),
            voltage=_clean_scalar(row.get(col_map.get("voltage", ""))),
            success=_clean_scalar(row.get(col_map.get("success", ""))),
            data_note=_clean_scalar(row.get(col_map.get("data_note", ""))),
            note=_clean_scalar(row.get(col_map.get("note", ""))),
            explanation=_clean_scalar(row.get(col_map.get("explanation", ""))),
        )

        for folder in test_ids:
            normalized_folder = normalize_test_identifier(folder)
            if not normalized_folder:
                continue

            entry = ExperimentLogEntry(
                folder=folder,
                normalized_folder=normalized_folder,
                **entry_template,
            )

            existing = entries.get(normalized_folder)
            if existing is None or _entry_score(entry) >= _entry_score(existing):
                entries[normalized_folder] = entry

    return list(entries.values())


def _entry_score(entry: ExperimentLogEntry) -> tuple[int, int]:
    info_score = sum(
        int(bool(value))
        for value in [
            entry.pump_bar_id,
            entry.raw_test_type,
            entry.frequency_hz,
            entry.sweep_start_hz,
            entry.sweep_end_hz,
            entry.duration_s,
            entry.date,
            entry.author,
            entry.voltage,
            entry.success,
            entry.combined_notes,
        ]
    )
    return (info_score, entry.row_number)


@lru_cache(maxsize=8)
def _load_entries_for_root(root_str: str) -> tuple[ExperimentLogEntry, ...]:
    log_path = find_experiment_log(root_str)
    if log_path is None:
        return ()

    df = pd.read_excel(log_path)
    df = df.dropna(how="all")
    return tuple(parse_experiment_log_dataframe(df))


@lru_cache(maxsize=8)
def _build_entry_index(root_str: str) -> dict[str, ExperimentLogEntry]:
    return {
        entry.normalized_folder: entry
        for entry in _load_entries_for_root(root_str)
    }


def list_experiment_log_entries(data_root: str | Path) -> list[ExperimentLogEntry]:
    """Return all parsed experiment-log entries for a root."""
    root_str = str(Path(data_root).expanduser().resolve())
    return list(_load_entries_for_root(root_str))


def lookup_experiment_log_entry(
    data_root: str | Path | None,
    run_name: str,
) -> ExperimentLogEntry | None:
    """Look up one test folder in the experiment log."""
    if data_root is None:
        return None

    root_str = str(Path(data_root).expanduser().resolve())
    normalized_name = normalize_test_identifier(run_name)
    if not normalized_name:
        return None

    index = _build_entry_index(root_str)
    direct = index.get(normalized_name)
    if direct is not None:
        return direct

    for key, entry in index.items():
        if key.startswith(normalized_name) or normalized_name.startswith(key):
            return entry
    return None


__all__ = [
    "EXPERIMENT_LOG_FILENAME",
    "ExperimentLogEntry",
    "experiment_log_storage_signature",
    "extract_constant_frequency_hz",
    "extract_duration_seconds",
    "extract_sweep_parameters",
    "find_experiment_log",
    "infer_test_type_from_text",
    "list_experiment_log_entries",
    "lookup_experiment_log_entry",
    "normalize_test_identifier",
    "parse_experiment_log_dataframe",
]
