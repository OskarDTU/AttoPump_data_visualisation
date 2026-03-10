"""Persistence layer for Pump definitions and Shipment groups.

A **pump** is a named collection of test folders that belong to the same
physical pump (e.g. "Pump 1" → ["test_folder_A", "test_folder_B"]).

A **shipment** groups multiple pumps together under a recipient label
(e.g. "Niels' pumps" → ["Pump 1", "Pump 2"]).

Data is stored in ``bar_groups.json`` next to ``test_metadata.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

_BAR_GROUPS_PATH = Path(__file__).parent / "bar_groups.json"


# ─── dataclasses ────────────────────────────────────────────────────────────

@dataclass
class Bar:
    """A named pump with its associated test folders."""
    name: str
    tests: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class Shipment:
    """A shipment groups pumps under a recipient name."""
    name: str
    bars: list[str] = field(default_factory=list)
    recipient: str = ""
    description: str = ""


@dataclass
class TestGroup:
    """A named collection of test folders for quick re-comparison.

    Unlike a pump (which represents a physical pump), a test group is an
    arbitrary set of test folders saved for convenience.
    """
    name: str
    tests: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class BarGroupsStore:
    """Top-level container for all pumps, shipments, and test groups."""
    bars: dict[str, Bar] = field(default_factory=dict)
    shipments: dict[str, Shipment] = field(default_factory=dict)
    test_groups: dict[str, TestGroup] = field(default_factory=dict)


# ─── serialisation helpers ──────────────────────────────────────────────────

def _store_to_dict(store: BarGroupsStore) -> dict[str, Any]:
    return {
        "_description": (
            "Pump groups, shipments, and test groups for AttoPump comparisons. "
            "Managed by the Manage Groups page."
        ),
        "bars": {name: asdict(bar) for name, bar in store.bars.items()},
        "shipments": {name: asdict(s) for name, s in store.shipments.items()},
        "test_groups": {name: asdict(tg) for name, tg in store.test_groups.items()},
    }


def _dict_to_store(raw: dict[str, Any]) -> BarGroupsStore:
    bars: dict[str, Bar] = {}
    for name, d in raw.get("bars", {}).items():
        bars[name] = Bar(
            name=d.get("name", name),
            tests=d.get("tests", []),
            description=d.get("description", ""),
        )
    shipments: dict[str, Shipment] = {}
    for name, d in raw.get("shipments", {}).items():
        shipments[name] = Shipment(
            name=d.get("name", name),
            bars=d.get("bars", []),
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
    return BarGroupsStore(bars=bars, shipments=shipments, test_groups=test_groups)


# ─── public API ─────────────────────────────────────────────────────────────

def load_bar_groups() -> BarGroupsStore:
    """Load bar groups from disk (or return empty store)."""
    if _BAR_GROUPS_PATH.exists():
        with open(_BAR_GROUPS_PATH, "r") as f:
            raw = json.load(f)
        return _dict_to_store(raw)
    return BarGroupsStore()


def save_bar_groups(store: BarGroupsStore) -> None:
    """Persist the full store to disk."""
    with open(_BAR_GROUPS_PATH, "w") as f:
        json.dump(_store_to_dict(store), f, indent=2)


# ── Bar CRUD ────────────────────────────────────────────────────────────────

def add_bar(store: BarGroupsStore, bar: Bar) -> BarGroupsStore:
    """Add (or overwrite) a bar in the store.

    Parameters
    ----------
    store : BarGroupsStore
        In-memory store to mutate.
    bar : Bar
        Bar to insert (keyed by ``bar.name``).

    Returns
    -------
    BarGroupsStore
        The same ``store`` object (mutated in place) for chaining.
    """
    store.bars[bar.name] = bar
    return store


def remove_bar(store: BarGroupsStore, bar_name: str) -> BarGroupsStore:
    """Remove a bar and clean up any shipment references to it.

    Parameters
    ----------
    store : BarGroupsStore
        In-memory store to mutate.
    bar_name : str
        Name of the bar to remove.  No-op if it doesn't exist.

    Returns
    -------
    BarGroupsStore
        The same ``store`` object (mutated in place).
    """
    store.bars.pop(bar_name, None)
    # Also remove from any shipments that reference it
    for shipment in store.shipments.values():
        if bar_name in shipment.bars:
            shipment.bars.remove(bar_name)
    return store


def rename_bar(store: BarGroupsStore, old_name: str, new_name: str) -> BarGroupsStore:
    """Rename a bar and update all shipment references.

    Parameters
    ----------
    store : BarGroupsStore
        In-memory store to mutate.
    old_name : str
        Current name of the bar.  No-op if it doesn't exist.
    new_name : str
        New name for the bar.

    Returns
    -------
    BarGroupsStore
        The same ``store`` object (mutated in place).
    """
    if old_name not in store.bars:
        return store
    bar = store.bars.pop(old_name)
    bar.name = new_name
    store.bars[new_name] = bar
    # Update references in shipments
    for shipment in store.shipments.values():
        shipment.bars = [new_name if b == old_name else b for b in shipment.bars]
    return store


# ── Shipment CRUD ───────────────────────────────────────────────────────────

def add_shipment(store: BarGroupsStore, shipment: Shipment) -> BarGroupsStore:
    """Add (or overwrite) a shipment in the store.

    Parameters
    ----------
    store : BarGroupsStore
        In-memory store to mutate.
    shipment : Shipment
        Shipment to insert (keyed by ``shipment.name``).

    Returns
    -------
    BarGroupsStore
        The same ``store`` object (mutated in place).
    """
    store.shipments[shipment.name] = shipment
    return store


def remove_shipment(store: BarGroupsStore, shipment_name: str) -> BarGroupsStore:
    """Remove a shipment from the store.

    Parameters
    ----------
    store : BarGroupsStore
        In-memory store to mutate.
    shipment_name : str
        Name of the shipment to remove.  No-op if it doesn't exist.

    Returns
    -------
    BarGroupsStore
        The same ``store`` object (mutated in place).
    """
    store.shipments.pop(shipment_name, None)
    return store


# ── Test Group CRUD ─────────────────────────────────────────────────────

def add_test_group(store: BarGroupsStore, tg: TestGroup) -> BarGroupsStore:
    """Add (or overwrite) a test group in the store.

    Parameters
    ----------
    store : BarGroupsStore
        In-memory store to mutate.
    tg : TestGroup
        Test group to insert (keyed by ``tg.name``).

    Returns
    -------
    BarGroupsStore
        The same ``store`` object (mutated in place).
    """
    store.test_groups[tg.name] = tg
    return store


def remove_test_group(store: BarGroupsStore, group_name: str) -> BarGroupsStore:
    """Remove a test group from the store.

    Parameters
    ----------
    store : BarGroupsStore
        In-memory store to mutate.
    group_name : str
        Name of the test group to remove.  No-op if it doesn't exist.

    Returns
    -------
    BarGroupsStore
        The same ``store`` object (mutated in place).
    """
    store.test_groups.pop(group_name, None)
    return store


# ── Experiment-log sync ─────────────────────────────────────────────────

def sync_pumps_from_experiment_log(
    store: BarGroupsStore,
    data_root: str | Path,
    *,
    available_folders: list[str] | None = None,
) -> tuple[BarGroupsStore, dict[str, list[str]]]:
    """Populate pump → test-folder associations from the experiment log.

    Reads the shared experiment-log spreadsheet (located relative to
    *data_root*), groups every row that has both a Pump/BAR ID and at
    least one Test ID, and updates the pump entries in *store*.

    Only rows whose Pump/BAR ID is **not** prefixed with ``BAR`` are
    considered (i.e. only PUMP rows).  If *available_folders* is
    provided, only test folders that actually exist on disk are linked.

    New pump entries are created as needed; existing ones get their test
    lists **merged** (no duplicates, order preserved).

    Parameters
    ----------
    store : BarGroupsStore
        In-memory store to mutate.
    data_root : str | Path
        Root path that ``find_experiment_log`` uses to locate the xlsx.
    available_folders : list[str] | None
        If given, only link test folders present in this list.

    Returns
    -------
    tuple[BarGroupsStore, dict[str, list[str]]]
        The mutated store and a ``{pump_name: [newly_linked_folders]}``
        summary of what changed.
    """
    from .experiment_log import list_experiment_log_entries

    entries = list_experiment_log_entries(data_root)
    available_set = set(available_folders) if available_folders else None

    # Group test folders by normalised pump ID
    pump_tests: dict[str, list[str]] = {}
    for entry in entries:
        pid = entry.pump_bar_id.strip()
        folder = entry.folder.strip()
        if not pid or not folder:
            continue
        # Skip BAR entries — only pump entries
        if pid.upper().startswith("BAR"):
            continue
        # Normalise common prefix variants to "Pump …"
        display_name = pid
        if display_name.upper().startswith("PUMP "):
            display_name = "Pump" + display_name[4:]  # keep the rest
        # Filter to folders that exist on disk (if requested)
        if available_set is not None and folder not in available_set:
            continue
        pump_tests.setdefault(display_name, []).append(folder)

    # Merge into store
    changes: dict[str, list[str]] = {}
    for pump_name, folders in pump_tests.items():
        existing = store.bars.get(pump_name)
        if existing is None:
            # Also check for case-insensitive match
            for k, v in store.bars.items():
                if k.lower() == pump_name.lower():
                    existing = v
                    pump_name = k  # use existing key
                    break

        if existing is not None:
            before = set(existing.tests)
            merged = list(existing.tests)
            for f in folders:
                if f not in before:
                    merged.append(f)
                    before.add(f)
            newly_added = [f for f in merged if f not in set(store.bars[pump_name].tests)]
            existing.tests = merged
            if newly_added:
                changes[pump_name] = newly_added
        else:
            store.bars[pump_name] = Bar(
                name=pump_name,
                tests=folders,
                description=f"Auto-imported from experiment log",
            )
            changes[pump_name] = folders

    return store, changes
