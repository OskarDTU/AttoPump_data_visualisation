"""Regression tests for report-builder subgroup selection."""

from __future__ import annotations

from app.data.pump_registry import Pump, PumpRegistry, PumpSubGroup, TestLink
from app.pages.report_builder import (
    _decode_sub_group_entry_id,
    _encode_sub_group_entry_id,
    _resolve_report_entry,
)


def test_sub_group_entry_ids_round_trip() -> None:
    """Saved subgroup entry IDs should survive encode/decode."""
    entry_id = _encode_sub_group_entry_id("Pump A", "Stable sweeps")

    assert _decode_sub_group_entry_id(entry_id) == ("Pump A", "Stable sweeps")


def test_resolve_report_entry_uses_only_sub_group_tests() -> None:
    """Report targets for pump sub-groups should only include subgroup tests."""
    reg = PumpRegistry(
        pumps={
            "Pump A": Pump(
                name="Pump A",
                notes="Pump note",
                tests=[
                    TestLink(folder="test-1", description="baseline"),
                    TestLink(folder="test-2", description="variant"),
                ],
                sub_groups={
                    "Stable sweeps": PumpSubGroup(
                        name="Stable sweeps",
                        tests=["test-2"],
                        description="Focus set",
                    )
                },
            )
        }
    )

    resolved = _resolve_report_entry(
        _encode_sub_group_entry_id("Pump A", "Stable sweeps"),
        reg,
        ["test-1", "test-2"],
    )

    assert resolved is not None
    display_name, folders, metadata = resolved
    assert display_name == "Pump A / Stable sweeps"
    assert folders == ["test-2"]
    assert [test["folder"] for test in metadata["tests"]] == ["test-2"]
    assert "Focus set" in metadata["notes"]
