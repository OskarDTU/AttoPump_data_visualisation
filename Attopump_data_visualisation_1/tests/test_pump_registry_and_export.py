"""Regression tests for saved analysis state and HTML export."""

from __future__ import annotations

import sys
import types

import numpy as np
import plotly.graph_objects as go

from app.data import pump_registry
from app.data.pump_registry import (
    AnalysisConfig,
    Pump,
    PumpRegistry,
    PumpSubGroup,
    TestLink as PumpTestLink,
)

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

fake_px = types.ModuleType("plotly.express")
fake_px.colors = types.SimpleNamespace(
    qualitative=types.SimpleNamespace(
        Plotly=["#636EFA"],
        Set2=["#00CC96"],
        Dark24=["#19D3F3"],
        Bold=["#EF553B"],
        Set1=["#AB63FA"],
    )
)
sys.modules.setdefault("plotly.express", fake_px)

from app.plots.plot_generator import _prepare_figure_for_html_export, export_html


def test_upsert_sub_group_can_rename_and_update(monkeypatch) -> None:
    """Renaming a pump sub-group should replace the old key."""
    monkeypatch.setattr(pump_registry, "_append_audit", lambda record: None)
    reg = PumpRegistry(
        pumps={
            "Pump A": Pump(
                name="Pump A",
                tests=[PumpTestLink(folder="test-1"), PumpTestLink(folder="test-2")],
                sub_groups={
                    "Original": PumpSubGroup(
                        name="Original",
                        tests=["test-1"],
                        description="before",
                    )
                },
            )
        }
    )

    pump_registry.upsert_sub_group(
        reg,
        "Pump A",
        PumpSubGroup(
            name="Renamed",
            tests=["test-2"],
            description="after",
        ),
        previous_name="Original",
    )

    assert "Original" not in reg.pumps["Pump A"].sub_groups
    assert reg.pumps["Pump A"].sub_groups["Renamed"].tests == ["test-2"]
    assert reg.pumps["Pump A"].sub_groups["Renamed"].description == "after"


def test_upsert_analysis_config_can_rename_and_update(monkeypatch) -> None:
    """Renaming a saved analysis config should preserve the updated fields."""
    monkeypatch.setattr(pump_registry, "_append_audit", lambda record: None)
    reg = PumpRegistry(
        analysis_configs={
            "Baseline": AnalysisConfig(
                name="Baseline",
                plots=["sweep_overlay"],
                bin_hz=5.0,
            )
        }
    )

    pump_registry.upsert_analysis_config(
        reg,
        AnalysisConfig(
            name="Pump Focus",
            plots=["summary_table", "boxplots"],
            bin_hz=2.5,
            show_all_data_points=False,
            max_raw_points=25000,
            plot_mode="markers",
            marker_size=4,
            opacity=0.4,
        ),
        previous_name="Baseline",
    )

    assert "Baseline" not in reg.analysis_configs
    assert reg.analysis_configs["Pump Focus"].plots == ["summary_table", "boxplots"]
    assert reg.analysis_configs["Pump Focus"].max_raw_points == 25000
    assert reg.analysis_configs["Pump Focus"].plot_mode == "markers"


def test_export_html_converts_scattergl_traces(tmp_path) -> None:
    """Standalone HTML export should downgrade Scattergl to Scatter."""
    fig = go.Figure(
        go.Scattergl(
            x=[1, 2, 3],
            y=[4, 5, 6],
            mode="markers",
            name="raw",
        )
    )

    export_fig = _prepare_figure_for_html_export(fig)
    exported = export_html(fig, "scattergl.html", export_dir=tmp_path)

    assert exported.exists()
    assert export_fig.data[0].type == "scatter"
