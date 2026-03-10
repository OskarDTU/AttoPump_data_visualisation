"""Regression tests for saved-group selection in comprehensive analysis."""

from __future__ import annotations

from app.data.pump_registry import Pump, PumpRegistry, PumpSubGroup, TestGroup
from app.pages.analysis import _build_saved_test_group_options


def test_saved_group_options_include_pump_sub_groups() -> None:
    """Pump sub-groups should be exposed separately from top-level test groups."""
    reg = PumpRegistry(
        test_groups={
            "Shared sweeps": TestGroup(
                name="Shared sweeps",
                tests=["test-1", "test-2"],
                description="Global selection",
            )
        },
        pumps={
            "Pump A": Pump(
                name="Pump A",
                sub_groups={
                    "Stable": PumpSubGroup(
                        name="Stable",
                        tests=["test-2"],
                        description="Pump-specific focus set",
                    )
                },
            )
        },
    )

    saved_groups, pump_sub_groups = _build_saved_test_group_options(reg)

    assert saved_groups["Shared sweeps"]["tests"] == ["test-1", "test-2"]
    assert pump_sub_groups["Pump A"]["Stable"]["tests"] == ["test-2"]
    assert pump_sub_groups["Pump A"]["Stable"]["description"] == "Pump-specific focus set"
