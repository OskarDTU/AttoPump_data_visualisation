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
