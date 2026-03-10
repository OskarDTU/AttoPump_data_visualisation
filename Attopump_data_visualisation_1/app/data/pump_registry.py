"""Unified Pump Registry — single source of truth for pump-test mappings.

Replaces both ``bar_groups.py`` (simple pump→[folder] mapping) and
``bar_registry.py`` (rich metadata + audit trail) with ONE consistent
data model and persistence layer.

Data model
----------
- **Pump**         — one physical pump with linked tests, sub-groups.
- **TestLink**     — one test linked to a pump (folder + metadata).
- **PumpSubGroup** — a named subset of tests within a pump.
- **Shipment**     — groups pumps under a recipient label.
- **TestGroup**    — arbitrary saved test-folder collection.
- **AnalysisConfig** — saved analysis preset (plots, bin width, …).
- **AuditRecord**  — append-only change log entry.
- **PumpRegistry** — top-level container.

Persistence
-----------
``pump_registry.json`` — current state (replaces bar_groups.json).
``pump_registry_log.json`` — append-only audit trail.

Migration: ``migrate_legacy_files()`` reads old bar_groups.json and
bar_registry.json and merges them into the new format.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── File locations ──────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent
_REGISTRY_PATH = _DATA_DIR / "pump_registry.json"
_AUDIT_LOG_PATH = _DATA_DIR / "pump_registry_log.json"

# Legacy paths (for migration)
_LEGACY_BAR_GROUPS_PATH = _DATA_DIR / "bar_groups.json"
_LEGACY_BAR_REGISTRY_PATH = _DATA_DIR / "bar_registry.json"
_LEGACY_BAR_REGISTRY_LOG_PATH = _DATA_DIR / "bar_registry_log.json"


# ══════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TestLink:
    """One test linked to a pump with metadata."""

    folder: str
    test_type: str = ""  # "sweep" | "constant" | ""
    description: str = ""
    success: bool | None = None
    date: str = ""
    author: str = ""
    voltage: str = ""
    notes: str = ""


@dataclass
class PumpSubGroup:
    """A named subset of tests within a pump for focused comparison."""

    name: str
    tests: list[str] = field(default_factory=list)  # folder names
    description: str = ""


@dataclass
class Pump:
    """A physical pump with linked tests and optional sub-groups."""

    name: str
    tests: list[TestLink] = field(default_factory=list)
    description: str = ""
    notes: str = ""
    sub_groups: dict[str, PumpSubGroup] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _now_iso()

    @property
    def test_folders(self) -> list[str]:
        """Flat list of all linked test folder names."""
        return [t.folder for t in self.tests]


@dataclass
class Shipment:
    """A group of pumps shipped to a recipient."""

    name: str
    pumps: list[str] = field(default_factory=list)
    recipient: str = ""
    description: str = ""


@dataclass
class TestGroup:
    """An arbitrary saved collection of test folders."""

    name: str
    tests: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class AnalysisConfig:
    """A saved analysis configuration (which plots, settings)."""

    name: str
    description: str = ""
    plots: list[str] = field(default_factory=list)
    bin_hz: float = 5.0
    avg_bin_hz: float = 3.0
    freq_tol: float = 5.0
    show_error_bars: bool = True
    show_all_data_points: bool = True
    max_raw_points: int = 500_000
    plot_mode: str = "lines+markers"
    marker_size: int = 6
    opacity: float = 0.8
    mean_threshold_pct: int = 75
    std_threshold_pct: int = 10


# Available plot types for AnalysisConfig
AVAILABLE_PLOTS: dict[str, str] = {
    "sweep_overlay": "Sweep overlay (binned mean ± std)",
    "sweep_relative": "Relative sweep (0–100 %)",
    "individual_sweeps": "Individual test sweeps",
    "global_average": "Global average curve",
    "boxplots": "Flow distribution boxplots",
    "histograms": "Flow histograms",
    "summary_table": "Summary statistics table",
    "raw_points": "All raw data points",
    "std_vs_mean": "Std vs Mean scatter",
    "best_region": "Best operating region",
    "correlation": "Inter-test correlation heatmap",
    "time_series": "Time series overlay",
}


@dataclass
class AuditRecord:
    """One entry in the append-only change log."""

    timestamp: str
    action: str
    target: str
    details: dict = field(default_factory=dict)
    author: str = "system"


@dataclass
class PumpRegistry:
    """Top-level container for all pump-related data."""

    pumps: dict[str, Pump] = field(default_factory=dict)
    shipments: dict[str, Shipment] = field(default_factory=dict)
    test_groups: dict[str, TestGroup] = field(default_factory=dict)
    analysis_configs: dict[str, AnalysisConfig] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_audit(record: AuditRecord) -> None:
    log = _read_audit_log_raw()
    log.append(asdict(record))
    with open(_AUDIT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def _read_audit_log_raw() -> list[dict]:
    if _AUDIT_LOG_PATH.exists():
        try:
            with open(_AUDIT_LOG_PATH, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


# ══════════════════════════════════════════════════════════════════════════
# SERIALISATION
# ══════════════════════════════════════════════════════════════════════════

def _registry_to_dict(reg: PumpRegistry) -> dict[str, Any]:
    """Serialise the registry to a JSON-safe dict."""
    return {
        "_description": "Unified pump registry — maps physical pumps to test folders.",
        "_version": 2,
        "pumps": {
            name: {
                **asdict(pump),
                "sub_groups": {
                    sg_name: asdict(sg)
                    for sg_name, sg in pump.sub_groups.items()
                },
            }
            for name, pump in reg.pumps.items()
        },
        "shipments": {name: asdict(s) for name, s in reg.shipments.items()},
        "test_groups": {name: asdict(tg) for name, tg in reg.test_groups.items()},
        "analysis_configs": {name: asdict(ac) for name, ac in reg.analysis_configs.items()},
    }


def _dict_to_registry(raw: dict[str, Any]) -> PumpRegistry:
    """Deserialise from JSON dict."""
    pumps: dict[str, Pump] = {}
    for name, d in raw.get("pumps", {}).items():
        tests: list[TestLink] = []
        for t in d.get("tests", []):
            if isinstance(t, str):
                # Legacy bar_groups format: just folder name strings
                tests.append(TestLink(folder=t))
            elif isinstance(t, dict):
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

        sub_groups: dict[str, PumpSubGroup] = {}
        for sg_name, sg_d in d.get("sub_groups", {}).items():
            sub_groups[sg_name] = PumpSubGroup(
                name=sg_d.get("name", sg_name),
                tests=sg_d.get("tests", []),
                description=sg_d.get("description", ""),
            )

        pumps[name] = Pump(
            name=d.get("name", name),
            tests=tests,
            description=d.get("description", ""),
            notes=d.get("notes", ""),
            sub_groups=sub_groups,
            created_at=d.get("created_at", ""),
        )

    shipments: dict[str, Shipment] = {}
    for name, d in raw.get("shipments", {}).items():
        shipments[name] = Shipment(
            name=d.get("name", name),
            pumps=d.get("pumps", d.get("bars", [])),  # legacy compat
            recipient=d.get("recipient", ""),
            description=d.get("description", ""),
        )

    test_groups: dict[str, TestGroup] = {}
    for name, d in raw.get("test_groups", {}).items():
        test_groups[name] = TestGroup(
            name=d.get("name", name),
            tests=d.get("tests", []),
            description=d.get("description", ""),
        )

    analysis_configs: dict[str, AnalysisConfig] = {}
    for name, d in raw.get("analysis_configs", {}).items():
        analysis_configs[name] = AnalysisConfig(
            name=d.get("name", name),
            description=d.get("description", ""),
            plots=d.get("plots", []),
            bin_hz=d.get("bin_hz", 5.0),
            avg_bin_hz=d.get("avg_bin_hz", 3.0),
            freq_tol=d.get("freq_tol", 5.0),
            show_error_bars=d.get("show_error_bars", True),
            show_all_data_points=d.get("show_all_data_points", True),
            max_raw_points=d.get("max_raw_points", 500_000),
            plot_mode=d.get("plot_mode", "lines+markers"),
            marker_size=d.get("marker_size", 6),
            opacity=d.get("opacity", 0.8),
            mean_threshold_pct=d.get("mean_threshold_pct", 75),
            std_threshold_pct=d.get("std_threshold_pct", 10),
        )

    return PumpRegistry(
        pumps=pumps,
        shipments=shipments,
        test_groups=test_groups,
        analysis_configs=analysis_configs,
    )


# ══════════════════════════════════════════════════════════════════════════
# LEGACY MIGRATION
# ══════════════════════════════════════════════════════════════════════════

def _migrate_bar_groups(reg: PumpRegistry) -> bool:
    """Import data from legacy bar_groups.json into *reg*. Returns True if data was found."""
    if not _LEGACY_BAR_GROUPS_PATH.exists():
        return False
    try:
        with open(_LEGACY_BAR_GROUPS_PATH, "r") as f:
            raw = json.load(f)
    except Exception:
        return False

    changed = False
    # Migrate pumps (stored as "bars")
    for name, d in raw.get("bars", {}).items():
        if name in reg.pumps:
            continue  # already exists
        tests_raw = d.get("tests", [])
        tests = [TestLink(folder=t) if isinstance(t, str) else TestLink(
            folder=t.get("folder", ""),
            test_type=t.get("test_type", ""),
        ) for t in tests_raw]
        reg.pumps[name] = Pump(
            name=name,
            tests=tests,
            description=d.get("description", ""),
        )
        changed = True

    for name, d in raw.get("shipments", {}).items():
        if name in reg.shipments:
            continue
        reg.shipments[name] = Shipment(
            name=name,
            pumps=d.get("bars", []),
            recipient=d.get("recipient", ""),
            description=d.get("description", ""),
        )
        changed = True

    for name, d in raw.get("test_groups", {}).items():
        if name in reg.test_groups:
            continue
        reg.test_groups[name] = TestGroup(
            name=name,
            tests=d.get("tests", []),
            description=d.get("description", ""),
        )
        changed = True

    return changed


def _migrate_bar_registry(reg: PumpRegistry) -> bool:
    """Import data from legacy bar_registry.json. Returns True if data was found."""
    if not _LEGACY_BAR_REGISTRY_PATH.exists():
        return False
    try:
        with open(_LEGACY_BAR_REGISTRY_PATH, "r") as f:
            raw = json.load(f)
    except Exception:
        return False

    changed = False
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

        if eid in reg.pumps:
            # Merge tests that don't already exist
            existing_folders = {t.folder for t in reg.pumps[eid].tests}
            for tl in tests:
                if tl.folder not in existing_folders:
                    reg.pumps[eid].tests.append(tl)
                    existing_folders.add(tl.folder)
                    changed = True
            # Upgrade notes if richer
            if d.get("notes") and not reg.pumps[eid].notes:
                reg.pumps[eid].notes = d["notes"]
                changed = True
        else:
            reg.pumps[eid] = Pump(
                name=eid,
                tests=tests,
                description="",
                notes=d.get("notes", ""),
                created_at=d.get("created_at", ""),
            )
            changed = True

    return changed


def migrate_legacy_files() -> PumpRegistry:
    """Load or create the registry, migrating from legacy files if needed.

    Call this instead of ``load_registry()`` when you want auto-migration.
    """
    reg = load_registry()
    migrated = False

    if _migrate_bar_groups(reg):
        migrated = True
    if _migrate_bar_registry(reg):
        migrated = True

    if migrated:
        save_registry(reg)
        _append_audit(AuditRecord(
            timestamp=_now_iso(),
            action="legacy_migration",
            target="(all)",
            details={
                "pumps": len(reg.pumps),
                "shipments": len(reg.shipments),
                "test_groups": len(reg.test_groups),
                "sources": [
                    p.name for p in [_LEGACY_BAR_GROUPS_PATH, _LEGACY_BAR_REGISTRY_PATH]
                    if p.exists()
                ],
            },
        ))

    return reg


# ══════════════════════════════════════════════════════════════════════════
# LOAD / SAVE
# ══════════════════════════════════════════════════════════════════════════

def load_registry() -> PumpRegistry:
    """Load registry from disk (or return empty)."""
    if _REGISTRY_PATH.exists():
        try:
            with open(_REGISTRY_PATH, "r") as f:
                raw = json.load(f)
            return _dict_to_registry(raw)
        except Exception:
            return PumpRegistry()
    return PumpRegistry()


def save_registry(reg: PumpRegistry) -> None:
    """Persist the full registry to disk."""
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(_registry_to_dict(reg), f, indent=2)


def load_audit_log() -> list[AuditRecord]:
    """Load the full audit log."""
    raw = _read_audit_log_raw()
    return [
        AuditRecord(
            timestamp=d.get("timestamp", ""),
            action=d.get("action", ""),
            target=d.get("target", ""),
            details=d.get("details", {}),
            author=d.get("author", "system"),
        )
        for d in raw
    ]


# ══════════════════════════════════════════════════════════════════════════
# CRUD — PUMPS
# ══════════════════════════════════════════════════════════════════════════

def add_pump(reg: PumpRegistry, pump: Pump, *, author: str = "user") -> PumpRegistry:
    """Add or overwrite a pump. Logs the action."""
    is_new = pump.name not in reg.pumps
    reg.pumps[pump.name] = pump
    _append_audit(AuditRecord(
        timestamp=_now_iso(),
        action="create_pump" if is_new else "update_pump",
        target=pump.name,
        details={"n_tests": len(pump.tests), "notes": pump.notes},
        author=author,
    ))
    return reg


def remove_pump(reg: PumpRegistry, pump_name: str, *, author: str = "user") -> PumpRegistry:
    """Remove a pump and clean up shipment references."""
    old = reg.pumps.pop(pump_name, None)
    if old is not None:
        for ship in reg.shipments.values():
            if pump_name in ship.pumps:
                ship.pumps.remove(pump_name)
        _append_audit(AuditRecord(
            timestamp=_now_iso(),
            action="remove_pump",
            target=pump_name,
            details={"had_tests": [t.folder for t in old.tests]},
            author=author,
        ))
    return reg


def rename_pump(
    reg: PumpRegistry, old_name: str, new_name: str, *, author: str = "user",
) -> PumpRegistry:
    """Rename a pump and update shipment references."""
    if old_name not in reg.pumps:
        return reg
    pump = reg.pumps.pop(old_name)
    pump.name = new_name
    reg.pumps[new_name] = pump
    for ship in reg.shipments.values():
        ship.pumps = [new_name if p == old_name else p for p in ship.pumps]
    _append_audit(AuditRecord(
        timestamp=_now_iso(),
        action="rename_pump",
        target=old_name,
        details={"new_name": new_name},
        author=author,
    ))
    return reg


# ══════════════════════════════════════════════════════════════════════════
# CRUD — TEST LINKS
# ══════════════════════════════════════════════════════════════════════════

def link_test(
    reg: PumpRegistry, pump_name: str, test: TestLink, *, author: str = "user",
) -> PumpRegistry:
    """Link a test folder to a pump (no duplicates)."""
    pump = reg.pumps.get(pump_name)
    if pump is None:
        return reg
    if test.folder in {t.folder for t in pump.tests}:
        return reg
    pump.tests.append(test)
    _append_audit(AuditRecord(
        timestamp=_now_iso(),
        action="link_test",
        target=pump_name,
        details={"folder": test.folder, "test_type": test.test_type},
        author=author,
    ))
    return reg


def unlink_test(
    reg: PumpRegistry, pump_name: str, folder_name: str, *, author: str = "user",
) -> PumpRegistry:
    """Remove a test link from a pump."""
    pump = reg.pumps.get(pump_name)
    if pump is None:
        return reg
    before = len(pump.tests)
    pump.tests = [t for t in pump.tests if t.folder != folder_name]
    if len(pump.tests) < before:
        # Also remove from sub-groups
        for sg in pump.sub_groups.values():
            if folder_name in sg.tests:
                sg.tests.remove(folder_name)
        _append_audit(AuditRecord(
            timestamp=_now_iso(),
            action="unlink_test",
            target=pump_name,
            details={"folder": folder_name},
            author=author,
        ))
    return reg


# ══════════════════════════════════════════════════════════════════════════
# CRUD — SUB-GROUPS
# ══════════════════════════════════════════════════════════════════════════

def add_sub_group(
    reg: PumpRegistry,
    pump_name: str,
    group: PumpSubGroup,
    *,
    author: str = "user",
) -> PumpRegistry:
    """Add or overwrite a sub-group within a pump."""
    return upsert_sub_group(
        reg,
        pump_name,
        group,
        previous_name=group.name,
        author=author,
    )


def upsert_sub_group(
    reg: PumpRegistry,
    pump_name: str,
    group: PumpSubGroup,
    *,
    previous_name: str | None = None,
    author: str = "user",
) -> PumpRegistry:
    """Create, update, or rename a sub-group within a pump."""
    pump = reg.pumps.get(pump_name)
    if pump is None:
        return reg
    old_name = (previous_name or group.name).strip()
    new_name = group.name.strip()
    if not new_name:
        return reg
    group.name = new_name
    renamed = old_name != new_name
    existed_before = old_name in pump.sub_groups or new_name in pump.sub_groups
    if renamed and old_name in pump.sub_groups:
        pump.sub_groups.pop(old_name, None)
    pump.sub_groups[new_name] = group
    _append_audit(AuditRecord(
        timestamp=_now_iso(),
        action="rename_sub_group" if renamed else (
            "update_sub_group" if existed_before else "add_sub_group"
        ),
        target=pump_name,
        details={
            "group": group.name,
            "previous_name": old_name,
            "tests": group.tests,
            "description": group.description,
        },
        author=author,
    ))
    return reg


def remove_sub_group(
    reg: PumpRegistry, pump_name: str, group_name: str, *, author: str = "user",
) -> PumpRegistry:
    """Remove a sub-group from a pump."""
    pump = reg.pumps.get(pump_name)
    if pump is None:
        return reg
    removed = pump.sub_groups.pop(group_name, None)
    if removed is not None:
        _append_audit(AuditRecord(
            timestamp=_now_iso(),
            action="remove_sub_group",
            target=pump_name,
            details={"group": group_name},
            author=author,
        ))
    return reg


# ══════════════════════════════════════════════════════════════════════════
# CRUD — SHIPMENTS
# ══════════════════════════════════════════════════════════════════════════

def add_shipment(reg: PumpRegistry, shipment: Shipment, *, author: str = "user") -> PumpRegistry:
    reg.shipments[shipment.name] = shipment
    _append_audit(AuditRecord(
        timestamp=_now_iso(),
        action="add_shipment",
        target=shipment.name,
        details={"pumps": shipment.pumps, "recipient": shipment.recipient},
        author=author,
    ))
    return reg


def remove_shipment(reg: PumpRegistry, name: str, *, author: str = "user") -> PumpRegistry:
    removed = reg.shipments.pop(name, None)
    if removed is not None:
        _append_audit(AuditRecord(
            timestamp=_now_iso(),
            action="remove_shipment",
            target=name,
            author=author,
        ))
    return reg


# ══════════════════════════════════════════════════════════════════════════
# CRUD — TEST GROUPS
# ══════════════════════════════════════════════════════════════════════════

def add_test_group(reg: PumpRegistry, tg: TestGroup, *, author: str = "user") -> PumpRegistry:
    reg.test_groups[tg.name] = tg
    return reg


def remove_test_group(reg: PumpRegistry, name: str, *, author: str = "user") -> PumpRegistry:
    reg.test_groups.pop(name, None)
    return reg


# ══════════════════════════════════════════════════════════════════════════
# CRUD — ANALYSIS CONFIGS
# ══════════════════════════════════════════════════════════════════════════

def add_analysis_config(
    reg: PumpRegistry, config: AnalysisConfig, *, author: str = "user",
) -> PumpRegistry:
    return upsert_analysis_config(
        reg,
        config,
        previous_name=config.name,
        author=author,
    )


def upsert_analysis_config(
    reg: PumpRegistry,
    config: AnalysisConfig,
    *,
    previous_name: str | None = None,
    author: str = "user",
) -> PumpRegistry:
    """Create, update, or rename a saved analysis configuration."""
    old_name = (previous_name or config.name).strip()
    new_name = config.name.strip()
    if not new_name:
        return reg
    config.name = new_name
    renamed = old_name != new_name
    existed_before = old_name in reg.analysis_configs or new_name in reg.analysis_configs
    if renamed and old_name in reg.analysis_configs:
        reg.analysis_configs.pop(old_name, None)
    reg.analysis_configs[new_name] = config
    _append_audit(AuditRecord(
        timestamp=_now_iso(),
        action="rename_analysis_config" if renamed else (
            "update_analysis_config" if existed_before else "save_analysis_config"
        ),
        target=new_name,
        details={
            "previous_name": old_name,
            "plots": config.plots,
            "bin_hz": config.bin_hz,
            "avg_bin_hz": config.avg_bin_hz,
            "freq_tol": config.freq_tol,
            "show_all_data_points": config.show_all_data_points,
            "max_raw_points": config.max_raw_points,
            "mean_threshold_pct": config.mean_threshold_pct,
            "std_threshold_pct": config.std_threshold_pct,
        },
        author=author,
    ))
    return reg


def remove_analysis_config(
    reg: PumpRegistry, name: str, *, author: str = "user",
) -> PumpRegistry:
    removed = reg.analysis_configs.pop(name, None)
    if removed is not None:
        _append_audit(AuditRecord(
            timestamp=_now_iso(),
            action="remove_analysis_config",
            target=name,
            author=author,
        ))
    return reg


# ══════════════════════════════════════════════════════════════════════════
# LOOKUP HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_pump_for_folder(reg: PumpRegistry, folder_name: str) -> Pump | None:
    """Return the pump that contains the given test folder, or None."""
    for pump in reg.pumps.values():
        for t in pump.tests:
            if t.folder == folder_name:
                return pump
    return None


def get_pump_display_name(reg: PumpRegistry, folder_name: str) -> str:
    """Resolve a test folder to its parent pump's name, or the folder itself."""
    pump = get_pump_for_folder(reg, folder_name)
    return pump.name if pump else folder_name


def all_test_folders(reg: PumpRegistry) -> list[str]:
    """Flat list of every test folder across all pumps."""
    return [t.folder for pump in reg.pumps.values() for t in pump.tests]


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT LOG SYNC
# ══════════════════════════════════════════════════════════════════════════

def sync_pumps_from_experiment_log(
    reg: PumpRegistry,
    data_root: str | Path,
    *,
    available_folders: list[str] | None = None,
) -> tuple[PumpRegistry, dict[str, list[str]]]:
    """Populate pump→test associations from the experiment log.

    Reads the shared Excel logbook, groups rows by Pump/BAR ID, and
    merges into the registry.  Only PUMP rows (not BAR rows) are synced.

    Returns (registry, {pump_name: [newly_linked_folders]}).
    """
    from .experiment_log import list_experiment_log_entries

    entries = list_experiment_log_entries(data_root)
    available_set = set(available_folders) if available_folders else None

    pump_tests: dict[str, list[tuple[str, Any]]] = {}  # name -> [(folder, entry)]
    for entry in entries:
        pid = entry.pump_bar_id.strip()
        folder = entry.folder.strip()
        if not pid or not folder:
            continue
        if pid.upper().startswith("BAR"):
            continue
        # Normalise "PUMP …" → "Pump …"
        display_name = pid
        if display_name.upper().startswith("PUMP "):
            display_name = "Pump" + display_name[4:]
        if available_set is not None and folder not in available_set:
            continue
        pump_tests.setdefault(display_name, []).append((folder, entry))

    changes: dict[str, list[str]] = {}
    for pump_name, folder_entries in pump_tests.items():
        # Case-insensitive match with existing pump
        existing = reg.pumps.get(pump_name)
        if existing is None:
            for k, v in reg.pumps.items():
                if k.lower() == pump_name.lower():
                    existing = v
                    pump_name = k
                    break

        if existing is not None:
            existing_folders = {t.folder for t in existing.tests}
            newly_added = []
            for folder, log_entry in folder_entries:
                if folder not in existing_folders:
                    existing.tests.append(TestLink(
                        folder=folder,
                        test_type=getattr(log_entry, "raw_test_type", ""),
                        date=str(getattr(log_entry, "date", "") or ""),
                        author=str(getattr(log_entry, "author", "") or ""),
                        voltage=str(getattr(log_entry, "voltage", "") or ""),
                    ))
                    existing_folders.add(folder)
                    newly_added.append(folder)
            if newly_added:
                changes[pump_name] = newly_added
        else:
            tests = []
            for folder, log_entry in folder_entries:
                tests.append(TestLink(
                    folder=folder,
                    test_type=getattr(log_entry, "raw_test_type", ""),
                    date=str(getattr(log_entry, "date", "") or ""),
                    author=str(getattr(log_entry, "author", "") or ""),
                    voltage=str(getattr(log_entry, "voltage", "") or ""),
                ))
            reg.pumps[pump_name] = Pump(
                name=pump_name,
                tests=tests,
                description="Auto-imported from experiment log",
            )
            changes[pump_name] = [t.folder for t in tests]

    return reg, changes


# ══════════════════════════════════════════════════════════════════════════
# SPREADSHEET IMPORT (from bar_registry.py)
# ══════════════════════════════════════════════════════════════════════════

def import_from_dataframe(
    reg: PumpRegistry,
    df: "pd.DataFrame",
    *,
    author: str = "spreadsheet_import",
) -> tuple[PumpRegistry, int, int]:
    """Bulk-import pump→test mappings from a logbook DataFrame.

    Returns (registry, n_entries_created, n_tests_linked).
    """
    import pandas as pd

    col_map: dict[str, str] = {}
    lower_cols = {c: c.lower().strip() for c in df.columns}

    for orig, low in lower_cols.items():
        if any(kw in low for kw in ["pump", "bar", "device"]):
            col_map.setdefault("pump_id", orig)
        elif "test" in low and ("id" in low or "folder" in low or "name" in low):
            col_map.setdefault("test_id", orig)
        elif "test" in low and "type" in low:
            col_map.setdefault("test_type", orig)
        elif "success" in low or "fail" in low or "pass" in low:
            col_map.setdefault("success", orig)
        elif low in ("date", "dato"):
            col_map.setdefault("date", orig)
        elif "author" in low or "person" in low or "operator" in low:
            col_map.setdefault("author", orig)
        elif "volt" in low:
            col_map.setdefault("voltage", orig)
        elif "note" in low or "comment" in low:
            col_map.setdefault("notes", orig)

    if "pump_id" not in col_map or "test_id" not in col_map:
        raise ValueError(
            f"Need 'Pump/BAR ID' and 'Test ID' columns. Found: {list(df.columns)}"
        )

    n_created = 0
    n_linked = 0

    for _, row in df.iterrows():
        pump_id = str(row.get(col_map["pump_id"], "")).strip()
        test_id = str(row.get(col_map["test_id"], "")).strip()
        if not pump_id or not test_id or pump_id == "nan" or test_id == "nan":
            continue

        if pump_id not in reg.pumps:
            add_pump(reg, Pump(name=pump_id), author=author)
            n_created += 1

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

        def _clean(val: str) -> str:
            s = str(val).strip()
            return "" if s == "nan" else s

        tl = TestLink(
            folder=test_id,
            test_type=test_type,
            description=_clean(test_type_raw),
            success=success,
            date=_clean(row.get(col_map.get("date", ""), "")),
            author=_clean(row.get(col_map.get("author", ""), "")),
            voltage=_clean(row.get(col_map.get("voltage", ""), "")),
            notes=_clean(row.get(col_map.get("notes", ""), "")),
        )
        before = len(reg.pumps[pump_id].tests)
        link_test(reg, pump_id, tl, author=author)
        if len(reg.pumps[pump_id].tests) > before:
            n_linked += 1

    _append_audit(AuditRecord(
        timestamp=_now_iso(),
        action="bulk_import",
        target="(all)",
        details={"entries_created": n_created, "tests_linked": n_linked},
        author=author,
    ))

    return reg, n_created, n_linked
