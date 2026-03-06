"""BAR Registry — maps physical pumps / BARs to test folders with audit trail.

This module extends the BAR concept from ``bar_groups.py`` with richer
metadata and a full, append-only audit log that records every create /
update / delete operation.

Data model
----------
- **RegistryEntry** — one physical pump or BAR.  Contains a display
  name, a list of linked test folders with metadata, and free-text notes.
- **TestLink** — one test linked to an entry (folder name, test type,
  description, success/fail flag).
- **AuditRecord** — one line in the audit trail (timestamp, action,
  target, details, author).
- **BarRegistry** — top-level container; dict of entries keyed by ID.

Persistence
-----------
Two JSON files live alongside this module:

- ``bar_registry.json``  — current state of the registry.
- ``bar_registry_log.json`` — append-only audit log.

Spreadsheet import
------------------
``import_from_dataframe()`` accepts a ``pd.DataFrame`` with the same
column layout as the user's Excel logbook and bulk-creates or updates
registry entries + test links.

Public API
----------
- ``load_registry()`` / ``save_registry()``
- ``add_entry()`` / ``remove_entry()`` / ``rename_entry()``
- ``link_test()`` / ``unlink_test()``
- ``update_entry_notes()``
- ``import_from_dataframe()``
- ``load_audit_log()``
- ``get_display_name()`` — resolve a test folder to its parent BAR display name.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── File locations ──────────────────────────────────────────────────────
_REGISTRY_PATH = Path(__file__).parent / "bar_registry.json"
_AUDIT_LOG_PATH = Path(__file__).parent / "bar_registry_log.json"


# ══════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TestLink:
    """One test linked to a registry entry.

    Attributes
    ----------
    folder : str
        Test folder name (e.g. ``"20260227-1832-PUMP-20260227-2"``).
    test_type : str
        ``"constant"``, ``"sweep"``, or ``""`` (unknown).
    description : str
        Human-readable note (e.g. ``"500 Hz constant frequency"``).
    success : bool | None
        ``True`` = pass, ``False`` = fail, ``None`` = not assessed.
    date : str
        ISO date string of the test (e.g. ``"2026-02-27"``).
    author : str
        Who ran / logged the test.
    voltage : str
        Drive voltage (e.g. ``"100 V"``).
    notes : str
        Extra free-text notes.
    """

    folder: str
    test_type: str = ""
    description: str = ""
    success: bool | None = None
    date: str = ""
    author: str = ""
    voltage: str = ""
    notes: str = ""


@dataclass
class RegistryEntry:
    """One physical pump or BAR in the registry.

    Attributes
    ----------
    entry_id : str
        Unique key — typically the BAR / pump ID from the logbook
        (e.g. ``"Pump 270226-2"`` or ``"BAR 060204 O4 I4"``).
    display_name : str
        Short label for reports (can be same as ``entry_id`` or a
        user-defined alias like ``"Pump #3"``).
    tests : list[TestLink]
        Linked test folders with metadata.
    notes : str
        Free-text description of the pump / BAR.
    created_at : str
        ISO timestamp of creation.
    """

    entry_id: str
    display_name: str = ""
    tests: list[TestLink] = field(default_factory=list)
    notes: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.entry_id
        if not self.created_at:
            self.created_at = _now_iso()


@dataclass
class AuditRecord:
    """One entry in the append-only audit log.

    Attributes
    ----------
    timestamp : str
        ISO-8601 timestamp.
    action : str
        One of ``create_entry``, ``remove_entry``, ``rename_entry``,
        ``link_test``, ``unlink_test``, ``update_notes``,
        ``bulk_import``.
    target : str
        The entry_id affected.
    details : dict
        Action-specific payload.
    author : str
        Who performed the action (default ``"system"``).
    """

    timestamp: str
    action: str
    target: str
    details: dict = field(default_factory=dict)
    author: str = "system"


@dataclass
class BarRegistry:
    """Top-level container for the full registry."""

    entries: dict[str, RegistryEntry] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    """Current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_audit(record: AuditRecord) -> None:
    """Append a single audit record to the log file."""
    log = _read_audit_log_raw()
    log.append(asdict(record))
    with open(_AUDIT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def _read_audit_log_raw() -> list[dict]:
    """Read the raw audit log JSON."""
    if _AUDIT_LOG_PATH.exists():
        with open(_AUDIT_LOG_PATH, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    return []


# ══════════════════════════════════════════════════════════════════════════
# SERIALISATION
# ══════════════════════════════════════════════════════════════════════════

def _registry_to_dict(reg: BarRegistry) -> dict[str, Any]:
    return {
        "_description": "BAR / Pump registry — maps physical devices to test folders.",
        "entries": {eid: asdict(e) for eid, e in reg.entries.items()},
    }


def _dict_to_registry(raw: dict[str, Any]) -> BarRegistry:
    entries: dict[str, RegistryEntry] = {}
    for eid, d in raw.get("entries", {}).items():
        tests = []
        for t in d.get("tests", []):
            tests.append(TestLink(
                folder=t.get("folder", ""),
                test_type=t.get("test_type", ""),
                description=t.get("description", ""),
                success=t.get("success"),
                date=t.get("date", ""),
                author=t.get("author", ""),
                voltage=t.get("voltage", ""),
                notes=t.get("notes", ""),
            ))
        entries[eid] = RegistryEntry(
            entry_id=d.get("entry_id", eid),
            display_name=d.get("display_name", eid),
            tests=tests,
            notes=d.get("notes", ""),
            created_at=d.get("created_at", ""),
        )
    return BarRegistry(entries=entries)


# ══════════════════════════════════════════════════════════════════════════
# LOAD / SAVE
# ══════════════════════════════════════════════════════════════════════════

def load_registry() -> BarRegistry:
    """Load registry from disk (or return empty)."""
    if _REGISTRY_PATH.exists():
        with open(_REGISTRY_PATH, "r") as f:
            raw = json.load(f)
        return _dict_to_registry(raw)
    return BarRegistry()


def save_registry(reg: BarRegistry) -> None:
    """Persist the full registry to disk."""
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(_registry_to_dict(reg), f, indent=2)


def load_audit_log() -> list[AuditRecord]:
    """Load the full audit log as a list of ``AuditRecord`` objects."""
    raw = _read_audit_log_raw()
    records: list[AuditRecord] = []
    for d in raw:
        records.append(AuditRecord(
            timestamp=d.get("timestamp", ""),
            action=d.get("action", ""),
            target=d.get("target", ""),
            details=d.get("details", {}),
            author=d.get("author", "system"),
        ))
    return records


# ══════════════════════════════════════════════════════════════════════════
# CRUD — ENTRIES
# ══════════════════════════════════════════════════════════════════════════

def add_entry(
    reg: BarRegistry,
    entry: RegistryEntry,
    *,
    author: str = "user",
) -> BarRegistry:
    """Add or overwrite a registry entry.  Logs the action."""
    is_new = entry.entry_id not in reg.entries
    reg.entries[entry.entry_id] = entry
    _append_audit(AuditRecord(
        timestamp=_now_iso(),
        action="create_entry" if is_new else "update_entry",
        target=entry.entry_id,
        details={
            "display_name": entry.display_name,
            "notes": entry.notes,
            "n_tests": len(entry.tests),
        },
        author=author,
    ))
    return reg


def remove_entry(
    reg: BarRegistry,
    entry_id: str,
    *,
    author: str = "user",
) -> BarRegistry:
    """Remove an entry and log the deletion."""
    old = reg.entries.pop(entry_id, None)
    if old is not None:
        _append_audit(AuditRecord(
            timestamp=_now_iso(),
            action="remove_entry",
            target=entry_id,
            details={
                "display_name": old.display_name,
                "had_tests": [t.folder for t in old.tests],
            },
            author=author,
        ))
    return reg


def rename_entry(
    reg: BarRegistry,
    old_id: str,
    new_display_name: str,
    *,
    author: str = "user",
) -> BarRegistry:
    """Change the display name (NOT the key) of an entry."""
    entry = reg.entries.get(old_id)
    if entry is not None:
        prev = entry.display_name
        entry.display_name = new_display_name
        _append_audit(AuditRecord(
            timestamp=_now_iso(),
            action="rename_entry",
            target=old_id,
            details={"old_name": prev, "new_name": new_display_name},
            author=author,
        ))
    return reg


def update_entry_notes(
    reg: BarRegistry,
    entry_id: str,
    notes: str,
    *,
    author: str = "user",
) -> BarRegistry:
    """Update the free-text notes of an entry."""
    entry = reg.entries.get(entry_id)
    if entry is not None:
        old_notes = entry.notes
        entry.notes = notes
        _append_audit(AuditRecord(
            timestamp=_now_iso(),
            action="update_notes",
            target=entry_id,
            details={"old": old_notes, "new": notes},
            author=author,
        ))
    return reg


# ══════════════════════════════════════════════════════════════════════════
# CRUD — TEST LINKS
# ══════════════════════════════════════════════════════════════════════════

def link_test(
    reg: BarRegistry,
    entry_id: str,
    test: TestLink,
    *,
    author: str = "user",
) -> BarRegistry:
    """Link a test folder to a registry entry.  Logs the action."""
    entry = reg.entries.get(entry_id)
    if entry is None:
        return reg
    # Avoid duplicates
    existing_folders = {t.folder for t in entry.tests}
    if test.folder in existing_folders:
        return reg
    entry.tests.append(test)
    _append_audit(AuditRecord(
        timestamp=_now_iso(),
        action="link_test",
        target=entry_id,
        details={"folder": test.folder, "test_type": test.test_type,
                 "description": test.description},
        author=author,
    ))
    return reg


def unlink_test(
    reg: BarRegistry,
    entry_id: str,
    folder_name: str,
    *,
    author: str = "user",
) -> BarRegistry:
    """Remove a test link from an entry.  Logs the action."""
    entry = reg.entries.get(entry_id)
    if entry is None:
        return reg
    before = len(entry.tests)
    entry.tests = [t for t in entry.tests if t.folder != folder_name]
    if len(entry.tests) < before:
        _append_audit(AuditRecord(
            timestamp=_now_iso(),
            action="unlink_test",
            target=entry_id,
            details={"folder": folder_name},
            author=author,
        ))
    return reg


# ══════════════════════════════════════════════════════════════════════════
# LOOKUP HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_display_name(reg: BarRegistry, folder_name: str) -> str:
    """Resolve a test folder to its parent entry's display name.

    Returns the first matching entry's ``display_name``, or the
    ``folder_name`` itself if not found.
    """
    for entry in reg.entries.values():
        for t in entry.tests:
            if t.folder == folder_name:
                return entry.display_name
    return folder_name


def get_entry_for_folder(reg: BarRegistry, folder_name: str) -> RegistryEntry | None:
    """Return the registry entry that contains the given test folder."""
    for entry in reg.entries.values():
        for t in entry.tests:
            if t.folder == folder_name:
                return entry
    return None


def all_test_folders(reg: BarRegistry) -> list[str]:
    """Return a flat list of every test folder across all entries."""
    folders: list[str] = []
    for entry in reg.entries.values():
        for t in entry.tests:
            folders.append(t.folder)
    return folders


# ══════════════════════════════════════════════════════════════════════════
# SPREADSHEET IMPORT
# ══════════════════════════════════════════════════════════════════════════

def import_from_dataframe(
    reg: BarRegistry,
    df: "pd.DataFrame",
    *,
    author: str = "spreadsheet_import",
) -> tuple[BarRegistry, int, int]:
    """Bulk-import BAR/pump → test mappings from a logbook DataFrame.

    Expected columns (case-insensitive, partial matching):
        - Pump/BAR ID  (or "pump", "bar", "device")
        - Test ID       (or "test", "folder")
        - Test type     (optional)
        - Success/fail  (optional)
        - Date          (optional)
        - Author        (optional)
        - Voltage       (optional)
        - Notes / Data note (optional)

    Returns
    -------
    (registry, n_entries_created, n_tests_linked)
    """
    import pandas as pd

    # ── Normalise column names ──────────────────────────────────
    col_map: dict[str, str] = {}
    lower_cols = {c: c.lower().strip() for c in df.columns}

    for orig, low in lower_cols.items():
        if any(kw in low for kw in ["pump", "bar", "device"]):
            col_map.setdefault("bar_id", orig)
        elif "test" in low and ("id" in low or "folder" in low or "name" in low):
            col_map.setdefault("test_id", orig)
        elif "test" in low and "type" in low:
            col_map.setdefault("test_type", orig)
        elif "success" in low or "fail" in low or "pass" in low:
            col_map.setdefault("success", orig)
        elif low in ("date", "dato"):
            col_map.setdefault("date", orig)
        elif low in ("time", "tid"):
            pass  # skip time-of-day column
        elif "author" in low or "person" in low or "operator" in low:
            col_map.setdefault("author", orig)
        elif "volt" in low:
            col_map.setdefault("voltage", orig)
        elif "note" in low or "comment" in low:
            col_map.setdefault("notes", orig)

    if "bar_id" not in col_map or "test_id" not in col_map:
        raise ValueError(
            "Could not find required columns: need a 'Pump/BAR ID' column "
            f"and a 'Test ID' column.  Found columns: {list(df.columns)}"
        )

    n_created = 0
    n_linked = 0

    for _, row in df.iterrows():
        bar_id = str(row.get(col_map["bar_id"], "")).strip()
        test_id = str(row.get(col_map["test_id"], "")).strip()
        if not bar_id or not test_id or bar_id == "nan" or test_id == "nan":
            continue

        # Create entry if new
        if bar_id not in reg.entries:
            add_entry(reg, RegistryEntry(entry_id=bar_id), author=author)
            n_created += 1

        # Parse optional fields
        test_type_raw = str(row.get(col_map.get("test_type", ""), "")).strip()
        test_type = ""
        if test_type_raw:
            low = test_type_raw.lower()
            if "sweep" in low:
                test_type = "sweep"
            elif "constant" in low or "manual" in low:
                test_type = "constant"

        success_raw = str(row.get(col_map.get("success", ""), "")).strip().lower()
        success: bool | None = None
        if success_raw in ("yes", "true", "1", "success", "pass", "ok", "ja"):
            success = True
        elif success_raw in ("no", "false", "0", "fail", "failed", "nej"):
            success = False

        date_str = str(row.get(col_map.get("date", ""), "")).strip()
        if date_str == "nan":
            date_str = ""

        author_str = str(row.get(col_map.get("author", ""), "")).strip()
        if author_str == "nan":
            author_str = ""

        voltage_str = str(row.get(col_map.get("voltage", ""), "")).strip()
        if voltage_str == "nan":
            voltage_str = ""

        notes_str = str(row.get(col_map.get("notes", ""), "")).strip()
        if notes_str == "nan":
            notes_str = ""

        tl = TestLink(
            folder=test_id,
            test_type=test_type,
            description=test_type_raw if test_type_raw != "nan" else "",
            success=success,
            date=date_str,
            author=author_str,
            voltage=voltage_str,
            notes=notes_str,
        )
        before = len(reg.entries[bar_id].tests)
        link_test(reg, bar_id, tl, author=author)
        if len(reg.entries[bar_id].tests) > before:
            n_linked += 1

    # One top-level audit record summarising the bulk import
    _append_audit(AuditRecord(
        timestamp=_now_iso(),
        action="bulk_import",
        target="(all)",
        details={"entries_created": n_created, "tests_linked": n_linked,
                 "rows_processed": len(df)},
        author=author,
    ))

    return reg, n_created, n_linked
