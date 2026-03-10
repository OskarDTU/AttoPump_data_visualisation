"""Resolved per-test records for search, overview, and defaults."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from .data_processor import detect_test_type, load_test_metadata, parse_sweep_spec_from_name
from .experiment_log import lookup_experiment_log_entry
from .io_local import pick_best_csv
from .pump_registry import PumpRegistry, get_pump_for_folder, load_registry as _load_pump_registry
from .test_configs import get_test_config


@dataclass(frozen=True)
class ResolvedTestRecord:
    """Merged metadata about one discovered test folder."""

    run_name: str
    has_csv: bool
    csv_name: str = ""
    test_type: str = "unknown"
    detection_method: str = "unknown"
    pump_bar_id: str = ""
    raw_test_type: str = ""
    frequency_hz: float | None = None
    sweep_start_hz: float | None = None
    sweep_end_hz: float | None = None
    duration_s: float | None = None
    date: str = ""
    time: str = ""
    author: str = ""
    voltage: str = ""
    success: str = ""
    note: str = ""
    log_found: bool = False
    saved_config_found: bool = False
    metadata_found: bool = False
    issues: tuple[str, ...] = field(default_factory=tuple)


def _coalesce(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _build_note(log_entry, meta_entry: dict | None, saved_cfg) -> str:
    parts = []
    if log_entry is not None:
        if log_entry.raw_test_type:
            parts.append(log_entry.raw_test_type)
        if log_entry.combined_notes:
            parts.append(log_entry.combined_notes)
    if meta_entry and meta_entry.get("note"):
        parts.append(str(meta_entry["note"]).strip())
    if saved_cfg and saved_cfg.note:
        parts.append(saved_cfg.note.strip())

    seen: set[str] = set()
    merged: list[str] = []
    for part in parts:
        if not part or part in seen:
            continue
        seen.add(part)
        merged.append(part)
    return " | ".join(merged)


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def format_classification_summary(record: ResolvedTestRecord) -> str:
    """Return a compact human-readable classification label."""
    if record.test_type == "constant":
        if record.frequency_hz is not None:
            return f"Constant @ {_format_number(record.frequency_hz)} Hz"
        return "Constant"

    if record.test_type == "sweep":
        if record.sweep_start_hz is not None and record.sweep_end_hz is not None:
            summary = (
                f"Sweep {_format_number(record.sweep_start_hz)}"
                f"→{_format_number(record.sweep_end_hz)} Hz"
            )
            if record.duration_s:
                summary += f" / {_format_number(record.duration_s)} s"
            return summary
        return "Sweep"

    return "Unknown"


def format_detection_method(method: str) -> str:
    """Format a detection source for display."""
    return {
        "saved_config": "Saved config",
        "freq_set_hz_column": "CSV freq_set_hz",
        "experiment_log": "Experiment log",
        "metadata": "Metadata file",
        "regex": "Folder-name regex",
        "user_regex": "User regex",
        "unknown": "Unknown",
    }.get(method, method.replace("_", " ").title())


def resolve_test_record(
    run_name: str,
    data_root: str | Path | None,
    *,
    run_dir: str | Path | None = None,
    df: pd.DataFrame | None = None,
) -> ResolvedTestRecord:
    """Resolve all currently known metadata for one test folder."""
    root_path = Path(data_root).expanduser().resolve() if data_root is not None else None
    log_entry = lookup_experiment_log_entry(root_path, run_name) if root_path is not None else None
    saved_cfg = get_test_config(run_name)
    metadata = load_test_metadata()
    meta_entry = metadata.get(run_name)
    parsed_spec = parse_sweep_spec_from_name(run_name)

    has_csv = False
    csv_name = ""
    resolved_run_dir: Path | None = None
    if run_dir is not None:
        resolved_run_dir = Path(run_dir)
    elif root_path is not None:
        candidate = root_path / run_name
        if candidate.exists():
            resolved_run_dir = candidate

    if resolved_run_dir is not None and resolved_run_dir.exists():
        try:
            csv_name = pick_best_csv(resolved_run_dir).csv_path.name
            has_csv = True
        except Exception:
            has_csv = False

    detected_type, detection_method, _ = detect_test_type(
        run_name,
        df,
        data_root=root_path,
    )

    csv_frequency_hz = None
    if df is not None and "freq_set_hz" in df.columns:
        freq_series = pd.to_numeric(df["freq_set_hz"], errors="coerce").dropna()
        if not freq_series.empty and freq_series.nunique() == 1:
            csv_frequency_hz = float(freq_series.iloc[0])

    pump_bar_id = ""
    # Priority: experiment log → pump registry → test_metadata fallback
    if log_entry and getattr(log_entry, "pump_bar_id", ""):
        pump_bar_id = str(log_entry.pump_bar_id)
    else:
        try:
            _reg = _load_pump_registry()
            _pump = get_pump_for_folder(_reg, run_name)
            if _pump is not None:
                pump_bar_id = _pump.name
        except Exception:
            pass
    if not pump_bar_id and meta_entry and meta_entry.get("pump"):
        pump_bar_id = str(meta_entry["pump"])

    raw_test_type = str(_coalesce(
        getattr(log_entry, "raw_test_type", ""),
        meta_entry.get("type") if meta_entry else "",
        "",
    ) or "")

    if saved_cfg and saved_cfg.test_type == "constant":
        frequency_hz = _float_or_none(saved_cfg.frequency_hz)
    else:
        frequency_hz = _float_or_none(_coalesce(
            csv_frequency_hz,
            getattr(log_entry, "frequency_hz", None),
            meta_entry.get("frequency_hz") if meta_entry else None,
        ))

    if saved_cfg and saved_cfg.test_type == "sweep":
        sweep_start_hz = _float_or_none(saved_cfg.start_hz)
        sweep_end_hz = _float_or_none(saved_cfg.end_hz)
        duration_s = _float_or_none(saved_cfg.duration_s)
    else:
        sweep_start_hz = _float_or_none(_coalesce(
            getattr(log_entry, "sweep_start_hz", None),
            parsed_spec.start_hz if parsed_spec else None,
        ))
        sweep_end_hz = _float_or_none(_coalesce(
            getattr(log_entry, "sweep_end_hz", None),
            parsed_spec.end_hz if parsed_spec else None,
        ))
        duration_s = _float_or_none(_coalesce(
            getattr(log_entry, "duration_s", None),
            parsed_spec.duration_s if parsed_spec and parsed_spec.duration_s > 0 else None,
        ))

    issues: list[str] = []
    if not has_csv:
        issues.append("Missing CSV")
    if detected_type == "unknown":
        issues.append("Unknown test type")
    if not pump_bar_id:
        issues.append("Missing pump/BAR")
    if detected_type == "constant" and frequency_hz is None:
        issues.append("Missing constant frequency")
    if detected_type == "sweep":
        if sweep_start_hz is None or sweep_end_hz is None:
            issues.append("Missing sweep range")
        if duration_s is None or duration_s <= 0:
            issues.append("Missing sweep duration")

    return ResolvedTestRecord(
        run_name=run_name,
        has_csv=has_csv,
        csv_name=csv_name,
        test_type=detected_type,
        detection_method=detection_method,
        pump_bar_id=pump_bar_id,
        raw_test_type=raw_test_type,
        frequency_hz=frequency_hz,
        sweep_start_hz=sweep_start_hz,
        sweep_end_hz=sweep_end_hz,
        duration_s=duration_s,
        date=str(getattr(log_entry, "date", "") or ""),
        time=str(getattr(log_entry, "time", "") or ""),
        author=str(getattr(log_entry, "author", "") or ""),
        voltage=str(getattr(log_entry, "voltage", "") or ""),
        success=str(getattr(log_entry, "success", "") or ""),
        note=_build_note(log_entry, meta_entry, saved_cfg),
        log_found=log_entry is not None,
        saved_config_found=saved_cfg is not None,
        metadata_found=meta_entry is not None,
        issues=tuple(issues),
    )


def build_test_catalog_dataframe(
    data_root: str | Path,
    run_dirs: list[Path],
) -> pd.DataFrame:
    """Build a dataframe suitable for the overview page."""
    rows: list[dict] = []
    for run_dir in run_dirs:
        record = resolve_test_record(
            run_dir.name,
            data_root,
            run_dir=run_dir,
        )
        row = asdict(record)
        row["classification"] = format_classification_summary(record)
        row["detection_source"] = format_detection_method(record.detection_method)
        row["issues"] = ", ".join(record.issues)
        row["is_complete"] = not bool(record.issues)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(
        by=["is_complete", "run_name"],
        ascending=[True, True],
    ).reset_index(drop=True)


def rank_test_names(
    run_names: list[str],
    query: str,
    *,
    limit: int = 100,
) -> list[str]:
    """Return run names ranked by relevance to a free-text query."""
    if not query.strip():
        return list(run_names[:limit])

    normalized_query = query.lower().strip()
    query_tokens = [tok for tok in re.split(r"[^a-z0-9]+", normalized_query) if tok]

    def score(name: str) -> tuple[float, int, str]:
        lowered = name.lower()
        name_tokens = [tok for tok in re.split(r"[^a-z0-9]+", lowered) if tok]
        value = 0.0

        if lowered == normalized_query:
            value += 1000
        if lowered.startswith(normalized_query):
            value += 700
        if normalized_query in lowered:
            value += 500 - lowered.index(normalized_query)

        token_hits = sum(
            1 for token in query_tokens if token in lowered or token in name_tokens
        )
        value += token_hits * 100

        if query_tokens:
            joined = "".join(query_tokens)
            if joined and joined in re.sub(r"[^a-z0-9]+", "", lowered):
                value += 75

        value += SequenceMatcher(None, normalized_query, lowered).ratio() * 50
        return (value, -len(name), lowered)

    ranked = sorted(run_names, key=score, reverse=True)
    return ranked[:limit]


__all__ = [
    "ResolvedTestRecord",
    "build_test_catalog_dataframe",
    "format_classification_summary",
    "format_detection_method",
    "rank_test_names",
    "resolve_test_record",
]
