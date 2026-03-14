"""Regression tests for report-builder plot settings and best-region output."""

from __future__ import annotations

import sys
import types

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

fake_bn = types.ModuleType("bottleneck")
fake_bn.__version__ = "1.3.8"
sys.modules.setdefault("bottleneck", fake_bn)


def _identity_decorator(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def _wrap(func):
        return func

    return _wrap


class _FakeStreamlit(types.ModuleType):
    session_state: dict = {}

    def __getattr__(self, name: str):
        if name in {"cache_data", "dialog"}:
            return _identity_decorator
        if name == "session_state":
            return self.session_state
        return lambda *args, **kwargs: None


sys.modules.setdefault("streamlit", _FakeStreamlit("streamlit"))

import pandas as pd
import plotly.graph_objects as go

from app.pages import report_builder
from app import plot_guidance
from app.reports import generator
from app.reports.generator import AxisBounds, ReportDefinition


def test_report_definition_round_trips_extended_plot_settings(
    tmp_path,
    monkeypatch,
) -> None:
    """Saved report templates should preserve the extended plot settings."""
    monkeypatch.setattr(generator, "_REPORTS_PATH", tmp_path / "saved_reports.json")

    defn = ReportDefinition(
        title="Configured report",
        author="tester",
        entry_ids=["Pump A"],
        comparisons=["sweep_overlay", "best_region"],
        notes="notes",
        bin_hz=4.5,
        avg_bin_hz=7.5,
        show_error_bars=False,
        show_individual_tests=True,
        show_raw_all_sweeps=False,
        plot_mode="markers",
        plot_modes={
            "sweep_overlay_target": "lines+markers",
            "global_average": "lines",
            "constant_time_series": "markers",
        },
        marker_size=9,
        opacity=0.45,
        max_raw_points=12345,
        mean_threshold_pct=88,
        std_threshold_pct=7,
        sweep_axis=AxisBounds(x_min=100.0, x_max=1000.0, y_min=0.0, y_max=400.0),
        relative_axis=AxisBounds(x_min=150.0, x_max=900.0, y_min=0.0, y_max=100.0),
        time_axis=AxisBounds(x_min=0.0, x_max=60.0, y_min=10.0, y_max=80.0),
        variability_axis=AxisBounds(x_min=50.0, x_max=200.0, y_min=0.0, y_max=25.0),
        selection_mode="sub_groups",
        overlay_entry_ids=["Pump A"],
        auto_use_recommended_avg_bin=False,
        overlay_include_sweep_drilldown=True,
    )

    generator.save_report_definition("demo", defn)
    loaded = generator.load_report_definition("demo")

    assert loaded is not None
    assert loaded.avg_bin_hz == 7.5
    assert loaded.plot_mode == "markers"
    assert loaded.plot_modes["sweep_overlay_target"] == "lines+markers"
    assert loaded.plot_modes["global_average"] == "lines"
    assert loaded.plot_modes["constant_time_series"] == "markers"
    assert loaded.marker_size == 9
    assert loaded.opacity == 0.45
    assert loaded.max_raw_points == 12345
    assert loaded.mean_threshold_pct == 88
    assert loaded.std_threshold_pct == 7
    assert loaded.sweep_axis.x_min == 100.0
    assert loaded.relative_axis.y_max == 100.0
    assert loaded.time_axis.x_max == 60.0
    assert loaded.variability_axis.y_max == 25.0
    assert loaded.selection_mode == "sub_groups"
    assert loaded.overlay_entry_ids == ["Pump A"]
    assert loaded.auto_use_recommended_avg_bin is False
    assert loaded.overlay_include_sweep_drilldown is True


def test_best_region_section_uses_report_settings_and_stability_sort(
    monkeypatch,
) -> None:
    """Best-region report sections should reflect configured thresholds and sort stably."""
    fake_best_df = pd.DataFrame(
        {
            "test": ["Pump A", "Pump B", "Pump C"],
            "freq_center": [300.0, 200.0, 250.0],
            "mean": [120.0, 110.0, 130.0],
            "std": [7.0, 5.0, 5.5],
        }
    )
    fake_ap = types.SimpleNamespace(
        plot_stability_cloud=lambda *args, **kwargs: (go.Figure(), fake_best_df),
    )
    fake_bcp = types.SimpleNamespace()
    monkeypatch.setattr(report_builder, "st", types.SimpleNamespace(session_state={}))

    defn = ReportDefinition(
        comparisons=["best_region"],
        bin_hz=5.0,
        avg_bin_hz=8.0,
        marker_size=8,
        mean_threshold_pct=85,
        std_threshold_pct=12,
        sweep_axis=AxisBounds(x_min=100.0, x_max=1000.0, y_min=0.0, y_max=500.0),
        variability_axis=AxisBounds(x_min=50.0, x_max=150.0, y_min=0.0, y_max=10.0),
    )
    sections: list = []

    report_builder._add_comparison_section(
        "best_region",
        sections,
        defn,
        display_binned={"Pump A": pd.DataFrame({"freq_center": [1.0], "mean": [1.0], "std": [1.0]})},
        display_avg_binned={},
        overlay_display_binned={},
        display_raw={},
        display_sweep={},
        display_const={},
        bar_binned={},
        bar_avg_binned={},
        overlay_bar_sweep={},
        bar_const={},
        plot_cache_contexts=None,
        signal_col="flow",
        ap=fake_ap,
        bcp=fake_bcp,
    )

    assert len(sections) == 2
    assert sections[0].description.startswith("What this plot is for:")
    assert plot_guidance.PLOT_GUIDANCE["best_region"]["purpose"] in sections[0].description
    assert "top 85% by mean flow" in sections[0].description
    assert "lowest 12% by standard deviation" in sections[0].description
    assert tuple(sections[0].content.layout.xaxis.range) == (50.0, 150.0)
    assert tuple(sections[0].content.layout.yaxis.range) == (0.0, 10.0)
    assert sections[1].title == "Top Operating Points"
    assert sections[1].content["Test"].tolist() == ["Pump B", "Pump C", "Pump A"]


def test_build_guided_section_description_uses_shared_plot_guide() -> None:
    """Report descriptions should reuse the same shared plot guidance text."""
    description = report_builder._build_guided_section_description(
        "raw_points",
        "Section-specific note.",
    )

    assert description.startswith("What this plot is for:")
    assert plot_guidance.PLOT_GUIDANCE["raw_points"]["purpose"] in description
    assert "Method:" in description
    assert "How to read it:" in description
    assert description.endswith("Section-specific note.")


def test_resolve_report_plot_mode_prefers_specific_mode() -> None:
    """Per-plot report settings should override the legacy fallback mode."""
    defn = ReportDefinition(
        plot_mode="markers",
        plot_modes={
            "sweep_overlay_target": "lines+markers",
            "global_average": "lines",
        },
    )

    assert report_builder._resolve_report_plot_mode(defn, "sweep_overlay_target") == "lines+markers"
    assert report_builder._resolve_report_plot_mode(defn, "global_average") == "lines"
    assert report_builder._resolve_report_plot_mode(defn, "constant_time_series") == "markers"


def test_default_report_plot_modes_keep_raw_views_as_markers() -> None:
    """New report drafts should default raw-point views to markers only."""
    defaults = report_builder._default_report_plot_mode_widgets()

    assert defaults["rb_mode_raw_points"] == "markers"
    assert defaults["rb_mode_individual_raw_all_sweeps"] == "markers"


def test_apply_report_plot_layout_adds_spacing_legend_and_style_toggle() -> None:
    """Exported report figures should get readable spacing and a mode switcher."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[100, 200],
            y=[1.0, 2.0],
            mode="lines+markers",
            name="Pump A / very long subgroup name / 20260313-161148-1Hz_1500H_Hz_5",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[100, 200],
            y=[2.0, 3.0],
            mode="lines+markers",
            name="Pump A / very long subgroup name / 20260313-161100-1Hz_1500H_Hz_1_s",
        )
    )
    fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Flow (µL/min)")

    report_builder._apply_report_plot_layout(fig)

    assert fig.layout.margin.l == 92
    assert fig.layout.legend.orientation == "h"
    assert len(fig.layout.updatemenus) == 1
    assert "..." in fig.data[0].name


def test_build_common_grid_binned_sweeps_uses_shared_frequency_grid() -> None:
    """Average-style report plots should bin raw sweeps directly on one common grid."""
    sweep_data = {
        "Test A": pd.DataFrame(
            {
                "Frequency": [0.0, 1.0, 2.0, 3.0],
                "flow": [10.0, 12.0, 14.0, 16.0],
            }
        ),
        "Test B": pd.DataFrame(
            {
                "Frequency": [0.5, 1.5, 2.5, 3.5],
                "flow": [20.0, 22.0, 24.0, 26.0],
            }
        ),
    }

    binned = report_builder._build_common_grid_binned_sweeps(
        sweep_data,
        signal_col="flow",
        bin_hz=2.0,
    )

    assert sorted(binned.keys()) == ["Test A", "Test B"]
    assert binned["Test A"]["freq_center"].tolist() == [1.0, 3.0]
    assert binned["Test B"]["freq_center"].tolist() == [1.0, 3.0]
    assert binned["Test A"]["mean"].tolist() == [11.0, 15.0]
    assert binned["Test B"]["mean"].tolist() == [21.0, 25.0]


def test_split_report_section_description_prefers_visual_summary() -> None:
    """Section descriptions should surface the concrete visual summary first."""
    subtitle, details = report_builder._split_report_section_description(
        "What this plot is for: Background context.\n\n"
        "Method: Shared method details.\n\n"
        "What you are seeing: This is the short subtitle.\n\n"
        "Interpretation: Extra detail."
    )

    assert subtitle == "What you are seeing: This is the short subtitle."
    assert "Background context." in details
    assert "Shared method details." in details


def test_build_report_html_renders_subtitle_and_collapsible_table() -> None:
    """Exported HTML should keep long detail collapsed and long tables collapsible."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[1, 2], mode="lines"))
    fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Flow (µL/min)")

    html = generator.build_report_html(
        ReportDefinition(title="Demo"),
        [
            generator.ReportSection(
                kind="plot",
                title="Focused Plot",
                content=fig,
                description=(
                    "What this plot is for: Context.\n\n"
                    "What you are seeing: Short visible subtitle.\n\n"
                    "Method: Hidden detail."
                ),
            ),
            generator.ReportSection(
                kind="table",
                title="Average Curve Data",
                content=pd.DataFrame({"Frequency": [1.0, 2.0], "Mean": [3.0, 4.0]}),
                description="Tabulated frequency-grid values behind the plot.",
                collapsible=True,
                collapsed=True,
            ),
        ],
    )

    assert "section-subtitle" in html
    assert "Short visible subtitle." in html
    assert "More detail" in html
    assert "section-table" in html
    assert "Show table" in html
