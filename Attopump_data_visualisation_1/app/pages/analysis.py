"""Unified Comprehensive Analysis page — compare tests, pump internals, and pumps.

This page consolidates all multi-test and multi-pump comparison into a
single interface with three modes:

Modes
-----
1. **Compare Tests** — select individual tests (ad-hoc or from a saved
   test group) and run cross-test analysis with 7+ chart types.

2. **Single Pump Analysis** — focus on one pump, compare its linked tests,
   analyse one saved sub-group, or compare saved sub-groups.

3. **Compare Pumps** — select pumps or a shipment and compare aggregated
   results across pumps.
"""

from __future__ import annotations

import re
import traceback
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..data.io_local import normalize_root, pick_best_csv
from ..data.config import (
    DATA_EXPORT_DIR,
    DEFAULT_CONSTANT_FREQUENCY_HZ,
    PLOT_BIN_WIDTH_HZ,
    PLOT_HEIGHT,
)
from ..data.data_processor import (
    bin_by_frequency,
    detect_constant_frequency,
    detect_test_type,
    detect_time_format,
    extract_frequency_slice,
    guess_signal_column,
    guess_time_column,
    parse_sweep_spec_from_name,
    prepare_sweep_data,
    prepare_time_series_data,
)
from ..data.loader import load_csv_cached, resolve_data_path
from ..data.pump_registry import (
    AnalysisConfig,
    AVAILABLE_PLOTS,
    PumpSubGroup,
    PumpRegistry,
    TestGroup,
    add_test_group,
    migrate_legacy_files,
    save_registry,
    sync_pumps_from_experiment_log,
    upsert_analysis_config,
    upsert_sub_group,
)
from ..plots.analysis_plots import (
    build_summary_table,
    plot_all_raw_points,
    plot_combined_boxplots,
    plot_combined_histograms,
    plot_combined_overlay,
    plot_correlation_heatmap,
    plot_global_average,
    plot_per_test_sweeps,
    plot_relative_comparison,
    plot_stability_cloud,
    plot_std_vs_mean,
)
from ..plots.bar_comparison_plots import (
    build_bar_summary_table,
    plot_bar_constant_aggregated,
    plot_bar_constant_boxplots,
    plot_bar_constant_histograms,
    plot_bar_sweep_overlay,
    plot_bar_sweep_relative,
)
from ..plots.plot_generator import export_html, plot_sweep_binned
from .unknown_test_prompt import classify_tests_quick, render_unknown_test_prompt


# ────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ────────────────────────────────────────────────────────────────────────

def _get_registry() -> PumpRegistry:
    """Return the in-memory pump registry (loads from disk on first access)."""
    key = "_pump_registry"
    if key not in st.session_state:
        st.session_state[key] = migrate_legacy_files()
    return st.session_state[key]


def _persist() -> None:
    """Flush the pump registry to ``pump_registry.json``."""
    save_registry(_get_registry())


def _build_saved_test_group_options(
    reg: PumpRegistry,
) -> tuple[
    OrderedDict[str, dict[str, str | list[str]]],
    OrderedDict[str, OrderedDict[str, dict[str, str | list[str]]]],
]:
    """Collect saved top-level test groups and pump sub-groups for selection UI."""
    saved_test_groups: OrderedDict[str, dict[str, str | list[str]]] = OrderedDict()
    pump_sub_groups: OrderedDict[
        str, OrderedDict[str, dict[str, str | list[str]]]
    ] = OrderedDict()

    for group_name, group in sorted(reg.test_groups.items()):
        saved_test_groups[group_name] = {
            "tests": list(group.tests),
            "description": group.description,
        }

    for pump_name, pump in sorted(reg.pumps.items()):
        if not pump.sub_groups:
            continue
        pump_sub_groups[pump_name] = OrderedDict()
        for group_name, group in sorted(pump.sub_groups.items()):
            pump_sub_groups[pump_name][group_name] = {
                "tests": list(group.tests),
                "description": group.description,
            }

    return saved_test_groups, pump_sub_groups


TEST_ANALYSIS_PLOTS = [
    "time_series",
    "individual_sweeps",
    "sweep_overlay",
    "sweep_relative",
    "global_average",
    "raw_points",
    "summary_table",
    "boxplots",
    "histograms",
    "correlation",
    "std_vs_mean",
    "best_region",
]
DEFAULT_TEST_ANALYSIS_PLOTS = [
    "individual_sweeps",
    "sweep_overlay",
    "sweep_relative",
    "global_average",
    "raw_points",
    "summary_table",
    "boxplots",
    "histograms",
    "correlation",
    "std_vs_mean",
    "best_region",
]
COLLECTION_ANALYSIS_PLOTS = [
    "sweep_overlay",
    "sweep_relative",
    "boxplots",
    "histograms",
    "summary_table",
]


def _apply_analysis_config_to_session(cfg: AnalysisConfig) -> None:
    """Populate session-state controls from a saved analysis config."""
    st.session_state["a_bin"] = float(cfg.bin_hz)
    st.session_state["a_avg"] = float(cfg.avg_bin_hz)
    st.session_state["a_ftol"] = float(cfg.freq_tol)
    st.session_state["a_err"] = bool(cfg.show_error_bars)
    st.session_state["a_allpts"] = bool(cfg.show_all_data_points)
    st.session_state["a_maxpts"] = int(cfg.max_raw_points)
    st.session_state["a_mode"] = cfg.plot_mode
    st.session_state["a_msz"] = int(cfg.marker_size)
    st.session_state["a_opa"] = float(cfg.opacity)
    st.session_state["a_mpct"] = int(cfg.mean_threshold_pct)
    st.session_state["a_spct"] = int(cfg.std_threshold_pct)
    st.session_state["_analysis_saved_plots"] = list(cfg.plots)
    st.session_state["_analysis_cfg_version"] = (
        st.session_state.get("_analysis_cfg_version", 0) + 1
    )


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "analysis"


def _get_export_dir(scope_label: str) -> Path:
    """Return a stable export directory for the current analysis scope."""
    state = st.session_state.get("_analysis_export_state")
    if not state or state.get("scope") != scope_label:
        export_dir = DATA_EXPORT_DIR / (
            f"{datetime.now():%Y%m%d-%H%M%S}_{_slugify(scope_label)}"
        )
        st.session_state["_analysis_export_state"] = {
            "scope": scope_label,
            "path": str(export_dir),
        }
        state = st.session_state["_analysis_export_state"]
    return Path(state["path"])


def _auto_sync_registry(
    reg: PumpRegistry,
    data_folder_str: str,
    run_names: list[str],
    *,
    key_prefix: str,
) -> dict[str, list[str]]:
    """Auto-sync pump mappings from the experiment log once per root."""
    if not data_folder_str.strip():
        return {}

    try:
        root = normalize_root(data_folder_str)
    except Exception:
        return {}

    sync_key = f"_{key_prefix}_auto_sync::{root}"
    if st.session_state.get(sync_key):
        return {}

    _, changes = sync_pumps_from_experiment_log(
        reg,
        root,
        available_folders=run_names if run_names else None,
    )
    if changes:
        _persist()
    st.session_state[sync_key] = True
    return changes


def _prime_plot_selection_state(
    widget_key: str,
    allowed_plot_ids: list[str],
    default_plot_ids: list[str],
) -> None:
    """Seed a plot-selector widget, also after a config load."""
    version = st.session_state.get("_analysis_cfg_version", 0)
    version_key = f"_{widget_key}_cfg_version"
    if widget_key not in st.session_state or st.session_state.get(version_key) != version:
        seed = [
            plot_id
            for plot_id in st.session_state.get(
                "_analysis_saved_plots",
                default_plot_ids,
            )
            if plot_id in allowed_plot_ids
        ]
        st.session_state[widget_key] = seed or list(default_plot_ids)
        st.session_state[version_key] = version


def _render_plot_selector(
    title: str,
    *,
    widget_key: str,
    allowed_plot_ids: list[str],
    default_plot_ids: list[str],
) -> list[str]:
    """Render the current plot-selection control."""
    _prime_plot_selection_state(widget_key, allowed_plot_ids, default_plot_ids)
    st.markdown(f"**{title}**")
    return st.multiselect(
        "Plots / analyses to run",
        allowed_plot_ids,
        format_func=lambda plot_id: AVAILABLE_PLOTS.get(plot_id, plot_id),
        key=widget_key,
    )


def _render_save_analysis_config_form(
    reg: PumpRegistry,
    S: dict,
    selected_plots: list[str],
    *,
    key_suffix: str,
) -> None:
    """Save the current analysis settings as a reusable config."""
    with st.expander("💾 Save current analysis configuration", expanded=False):
        with st.form(f"save_analysis_config_{key_suffix}", clear_on_submit=True):
            cfg_name = st.text_input("Config name")
            cfg_desc = st.text_input("Description (optional)")
            save_clicked = st.form_submit_button("Save configuration")

        if save_clicked:
            if not cfg_name.strip():
                st.error("Config name required.")
            else:
                upsert_analysis_config(
                    reg,
                    AnalysisConfig(
                        name=cfg_name.strip(),
                        description=cfg_desc.strip(),
                        plots=list(selected_plots),
                        bin_hz=float(S["bin_hz"]),
                        avg_bin_hz=float(S["avg_bin_hz"]),
                        freq_tol=float(S["freq_tol"]),
                        show_error_bars=bool(S["err_bars"]),
                        show_all_data_points=bool(S["show_all_points"]),
                        max_raw_points=int(S["max_raw"]),
                        plot_mode=str(S["plot_mode"]),
                        marker_size=int(S["marker_sz"]),
                        opacity=float(S["opacity"]),
                        mean_threshold_pct=int(S["mean_pct"]),
                        std_threshold_pct=int(S["std_pct"]),
                    ),
                    previous_name=cfg_name.strip(),
                )
                _persist()
                st.session_state["a_cfg"] = cfg_name.strip()
                _apply_analysis_config_to_session(reg.analysis_configs[cfg_name.strip()])
                st.success(f"Saved configuration **{cfg_name.strip()}**.")
                st.rerun()


def _render_save_pump_group_form(
    reg: PumpRegistry,
    pump_name: str,
    selected_tests: list[str],
    *,
    key_suffix: str,
) -> None:
    """Save the current in-pump test selection as a persisted pump sub-group."""
    if not selected_tests:
        return

    with st.expander("💾 Save selected tests as a pump group", expanded=False):
        with st.form(f"save_pump_group_{key_suffix}", clear_on_submit=True):
            group_name = st.text_input("Group name")
            group_desc = st.text_input("Description (optional)")
            save_clicked = st.form_submit_button("Save group")

        if save_clicked:
            if not group_name.strip():
                st.error("Group name required.")
            else:
                upsert_sub_group(
                    reg,
                    pump_name,
                    PumpSubGroup(
                        name=group_name.strip(),
                        tests=list(selected_tests),
                        description=group_desc.strip(),
                    ),
                    previous_name=group_name.strip(),
                )
                _persist()
                st.success(
                    f"Saved **{group_name.strip()}** under **{pump_name}** "
                    f"({len(selected_tests)} test(s))."
                )
                st.rerun()


def _maybe_export(fig, filename: str, S: dict) -> None:
    """Export a figure to HTML if the toggle is on."""
    if S.get("export"):
        try:
            export_dir = _get_export_dir(S.get("export_scope", "analysis"))
            path = export_html(fig, filename, export_dir=export_dir)
            st.success(f"✅ Exported: {path.name}")
        except Exception as exc:
            st.warning(f"Export failed for {filename}: {exc}")


def _render_export_destination(S: dict) -> None:
    """Show where exported HTML files are being written."""
    if not S.get("export"):
        return
    export_dir = _get_export_dir(S.get("export_scope", "analysis"))
    st.caption(f"Export folder: `{export_dir}`")


def _plot_time_series_overlay(
    all_data: dict[str, pd.DataFrame],
    signal_col: str,
    S: dict,
) -> go.Figure:
    """Overlay all selected tests in time/elapsed-time space."""
    fig = go.Figure()
    x_axis_title = "Time"

    for name, df in all_data.items():
        if df.empty or signal_col not in df.columns:
            continue
        x_col = df.columns[0]
        x_series = df[x_col]

        if pd.api.types.is_datetime64_any_dtype(x_series):
            parsed = pd.to_datetime(x_series, errors="coerce")
            x_values = (parsed - parsed.min()).dt.total_seconds()
            x_axis_title = "Elapsed time (s)"
        else:
            x_values = pd.to_numeric(x_series, errors="coerce")
            if x_col.lower() in ("t_s", "elapsed_s", "elapsed_seconds", "t"):
                x_axis_title = "Elapsed time (s)"
            else:
                x_axis_title = x_col

        y_values = pd.to_numeric(df[signal_col], errors="coerce")
        mask = x_values.notna() & y_values.notna()
        if not mask.any():
            continue

        fig.add_trace(
            go.Scattergl(
                x=x_values[mask],
                y=y_values[mask],
                mode=S["plot_mode"],
                name=name,
                marker=dict(size=S["marker_sz"], opacity=S["opacity"]),
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="Time Series Overlay",
        height=PLOT_HEIGHT,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        dragmode="zoom",
        xaxis_title=x_axis_title,
        yaxis_title=f"{signal_col} (µL/min)",
        font=dict(size=12),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def main() -> None:  # noqa: C901
    """Entry point for the Comprehensive Analysis page."""
    try:
        st.title("🔬 Comprehensive Analysis")

        # ── Sidebar: data source (shared widget) ────────────────
        data_folder_str, run_dirs, run_names = resolve_data_path(
            key_suffix="an",
            render_widget=False,
        )

        # ── Sidebar: analysis config loading ────────────────────
        reg = _get_registry()
        auto_sync_changes = _auto_sync_registry(
            reg,
            data_folder_str,
            run_names,
            key_prefix="analysis",
        )
        with st.sidebar:
            st.divider()
            st.header("📂 Load Config")
            cfg_names = list(reg.analysis_configs.keys())
            chosen_cfg = st.selectbox(
                "Saved config",
                ["(manual)"] + cfg_names,
                key="a_cfg",
            )
            last_loaded_cfg = st.session_state.get("_analysis_last_loaded_cfg")
            if chosen_cfg != "(manual)" and chosen_cfg != last_loaded_cfg:
                _apply_analysis_config_to_session(reg.analysis_configs[chosen_cfg])
                st.session_state["_analysis_last_loaded_cfg"] = chosen_cfg
            elif chosen_cfg == "(manual)" and last_loaded_cfg != "(manual)":
                st.session_state["_analysis_last_loaded_cfg"] = "(manual)"

            st.divider()
            st.header("⚙️ Analysis Settings")
            bin_hz = st.slider(
                "Frequency bin width (Hz)", 0.5, 100.0,
                float(PLOT_BIN_WIDTH_HZ), 0.5, key="a_bin",
            )
            avg_bin_hz = st.slider(
                "Average-curve bin width (Hz)", 0.5, 100.0,
                3.0, 0.5, key="a_avg",
            )
            show_all_points = st.checkbox(
                "Plot all raw data points",
                value=True,
                key="a_allpts",
            )
            max_raw = st.slider(
                "Max raw points per test", 1000, 500000,
                500000, 5000, key="a_maxpts",
                disabled=show_all_points,
                help=(
                    "Used only when 'Plot all raw data points' is disabled."
                ),
            )
            freq_tol = st.slider(
                "Freq-match tolerance (Hz)", 0.5, 50.0, 5.0, 0.5,
                key="a_ftol",
                help=(
                    "When comparing mixed test types, sweep data within "
                    "±this of the constant frequency is used."
                ),
            )

            st.divider()
            st.header("🎨 Plot Appearance")
            plot_mode = st.selectbox(
                "Plot type",
                ["lines+markers", "lines", "markers"],
                key="a_mode",
            )
            marker_sz = st.slider("Marker size", 1, 20, 6, key="a_msz")
            opacity = st.slider(
                "Opacity", 0.1, 1.0, 0.8, 0.05, key="a_opa",
            )
            err_bars = st.checkbox(
                "Show ±1 std bands", True, key="a_err",
            )
            export = st.checkbox(
                "Export plots as HTML folder", False, key="a_exp",
            )

            st.divider()
            st.header("🎯 Stability Cloud")
            mean_pct = st.slider(
                "High-flow %", 50, 99, 75, key="a_mpct",
            )
            std_pct = st.slider(
                "Stability %", 1, 50, 10, key="a_spct",
            )

        # Pack settings for easy passing
        S = dict(
            bin_hz=bin_hz, avg_bin_hz=avg_bin_hz, max_raw=max_raw,
            show_all_points=show_all_points,
            freq_tol=freq_tol, plot_mode=plot_mode, marker_sz=marker_sz,
            opacity=opacity, err_bars=err_bars, export=export,
            mean_pct=mean_pct, std_pct=std_pct,
        )

        if auto_sync_changes:
            st.caption(
                "Auto-synced pump links from the experiment log for "
                + ", ".join(
                    f"{pump} ({len(folders)} test(s))"
                    for pump, folders in auto_sync_changes.items()
                )
            )

        # ── Mode selector ──────────────────────────────────────
        mode = st.radio(
            "What would you like to do?",
            ["📊 Compare Tests", "🔧 Single Pump Analysis", "📦 Compare Pumps"],
            horizontal=True,
            key="a_toplevel",
        )
        st.divider()

        if mode == "📊 Compare Tests":
            _render_compare_tests(S, run_dirs, run_names)
        elif mode == "🔧 Single Pump Analysis":
            _render_single_pump_analysis(S, run_dirs, run_names)
        else:
            _render_compare_bars(S, run_dirs, run_names)

    except Exception as e:
        st.error(f"❌ **CRITICAL ERROR:** {str(e)}")
        with st.expander("🔍 Debug"):
            st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════
# MODE 1 — COMPARE TESTS
# ════════════════════════════════════════════════════════════════════════

def _render_compare_tests(
    S: dict, run_dirs: list, run_names: list[str],
) -> None:
    """Compare Tests mode — select tests, load, analyse."""
    reg = _get_registry()
    active_plots = _render_plot_selector(
        "Test comparison output",
        widget_key="analysis_plots_compare_tests",
        allowed_plot_ids=TEST_ANALYSIS_PLOTS,
        default_plot_ids=DEFAULT_TEST_ANALYSIS_PLOTS,
    )
    _render_save_analysis_config_form(
        reg,
        S,
        active_plots,
        key_suffix="compare_tests",
    )

    preselected_test = st.session_state.pop("analysis_preselect", None)
    if (
        preselected_test
        and preselected_test in run_names
        and "ct_multi" not in st.session_state
    ):
        st.session_state["ct_multi"] = [preselected_test]

    # ── Selection method ────────────────────────────────────────
    sel = st.radio(
        "How would you like to select tests?",
        ["Pick from available tests", "Load a saved test group"],
        horizontal=True,
        key="ct_sel",
    )

    selected_names: list[str] = []
    selection_label = "custom_test_selection"

    if sel == "Load a saved test group":
        saved_test_groups, pump_sub_groups = _build_saved_test_group_options(reg)

        if not saved_test_groups and not pump_sub_groups:
            st.info(
                "No saved test groups or pump sub-groups exist yet. "
                "Create them on **Manage Groups** or pick tests manually."
            )
            return

        saved_sources: list[str] = []
        if saved_test_groups:
            saved_sources.append("Test groups")
        if pump_sub_groups:
            saved_sources.append("Pump sub-groups")
        if saved_test_groups and pump_sub_groups:
            saved_sources.append("All saved selections")

        chosen_payload: dict[str, str | list[str]]
        chosen_label: str

        if len(saved_sources) == 1:
            saved_source = saved_sources[0]
            st.caption(f"Saved selection source: **{saved_source}**")
        else:
            saved_source = st.radio(
                "Saved selection source",
                saved_sources,
                horizontal=True,
                key="ct_saved_source",
            )

        if saved_source == "Pump sub-groups":
            pump_choice = st.selectbox(
                "🔧 Pump",
                list(pump_sub_groups.keys()),
                key="ct_saved_pump",
            )
            group_choice = st.selectbox(
                "📋 Pump sub-group",
                list(pump_sub_groups[pump_choice].keys()),
                key="ct_saved_subgroup",
            )
            chosen_label = f"Pump sub-group / {pump_choice} / {group_choice}"
            chosen_payload = pump_sub_groups[pump_choice][group_choice]
        else:
            combined_groups: OrderedDict[str, dict[str, str | list[str]]] = OrderedDict()
            if saved_source in {"Test groups", "All saved selections"}:
                for group_name, payload in saved_test_groups.items():
                    combined_groups[f"Test group / {group_name}"] = payload
            if saved_source == "All saved selections":
                for pump_name, groups in pump_sub_groups.items():
                    for group_name, payload in groups.items():
                        combined_groups[
                            f"Pump sub-group / {pump_name} / {group_name}"
                        ] = payload

            chosen_label = st.selectbox(
                "📋 Saved test group or pump sub-group",
                list(combined_groups.keys()),
                key="ct_grp",
            )
            chosen_payload = combined_groups[chosen_label]

        group_tests = chosen_payload["tests"]
        selected_names = [t for t in group_tests if t in run_names]
        selection_label = _slugify(f"saved_group_{chosen_label}")
        st.caption(
            f"Selection contains {len(group_tests)} test(s), "
            f"{len(selected_names)} available in current data folder."
        )
        group_description = str(chosen_payload["description"]).strip()
        if group_description:
            st.caption(group_description)
    else:
        if not run_names:
            st.info("ℹ️ Enter a data folder path in the sidebar.")
            return
        # Quick helpers: select-all / clear
        c1, c2, c3 = st.columns([3, 0.6, 0.6])
        with c2:
            if st.button("All", key="ct_all"):
                st.session_state["ct_multi"] = run_names
                st.rerun()
        with c3:
            if st.button("Clear", key="ct_clr"):
                st.session_state["ct_multi"] = []
                st.rerun()
        with c1:
            selected_names = st.multiselect(
                "📂 Select tests", run_names, key="ct_multi",
            )
        selection_label = "manual_compare_tests"

    if not selected_names:
        st.info("👆 Select at least one test to begin.")
        return

    # ── Optional: save as group ─────────────────────────────────
    with st.expander("💾 Save selection as a test group", expanded=False):
        with st.form("ct_save_grp", clear_on_submit=True):
            grp_name = st.text_input("Group name")
            grp_desc = st.text_input("Description (optional)")
            if st.form_submit_button("Save"):
                if grp_name.strip():
                    add_test_group(
                        reg,
                        TestGroup(grp_name, list(selected_names), grp_desc),
                    )
                    _persist()
                    st.success(
                        f"Saved **{grp_name}** ({len(selected_names)} tests)"
                    )
                else:
                    st.error("Name required.")

    _render_selected_test_analysis(
        selected_names,
        run_dirs,
        run_names,
        S,
        active_plots,
        unknown_key_prefix="ct_utp",
        scope_label=selection_label,
        context_label="test comparison",
    )


def _render_selected_test_analysis(
    selected_names: list[str],
    run_dirs: list,
    run_names: list[str],
    S: dict,
    active_plots: list[str],
    *,
    unknown_key_prefix: str,
    scope_label: str,
    context_label: str,
) -> None:
    """Run the shared test-analysis flow for a concrete test selection."""
    if not active_plots:
        st.info("Select at least one plot or analysis to run.")
        return

    _, unknowns = classify_tests_quick(selected_names, run_dirs, run_names)
    if unknowns:
        all_classified = render_unknown_test_prompt(
            unknowns,
            run_dirs,
            run_names,
            key_prefix=unknown_key_prefix,
        )
        if not all_classified:
            st.info(
                "💡 Classify the unknown test(s) above, then the "
                f"{context_label} will load automatically."
            )
            return

    scoped_settings = dict(S)
    scoped_settings["export_scope"] = scope_label
    _render_export_destination(scoped_settings)

    data = _load_tests(selected_names, run_dirs, run_names, scoped_settings)
    if data is None:
        return

    (all_data, sweep_data, binned_data, const_data,
     signal_col, load_errors, _test_types, const_freqs) = data

    if load_errors:
        with st.expander(f"⚠️ {len(load_errors)} load issue(s)", expanded=False):
            for err in load_errors:
                st.warning(err)

    n_sweep = len(sweep_data)
    n_const = len(const_data)
    st.success(
        f"✅ Loaded **{len(all_data)}** test(s) — "
        f"**{n_sweep}** sweep, **{n_const}** constant-freq"
    )

    if "time_series" in active_plots and all_data:
        fig_ts = _plot_time_series_overlay(all_data, signal_col, scoped_settings)
        st.plotly_chart(fig_ts, use_container_width=True)
        _maybe_export(fig_ts, "time_series_overlay.html", scoped_settings)
        st.divider()

    mixed_mode = False
    matched_const_data: dict[str, pd.DataFrame] = {}
    target_freq: float | None = None

    if n_sweep > 0 and n_const > 0:
        target_freq = _detect_target_freq(const_freqs)

        st.warning(
            f"⚠️ **Mixed test types detected:** {n_sweep} frequency-sweep "
            f"test(s) and {n_const} constant-frequency test(s).\n\n"
            "Sweep tests cover a range of frequencies while constant-"
            "frequency tests measure at a single frequency.  "
            "Direct comparison is not straightforward."
        )

        mixed_mode = st.checkbox(
            f"Compare anyway — extract sweep data at "
            f"**{target_freq:.0f} Hz** (±{scoped_settings['freq_tol']:.0f} Hz) "
            f"to match constant-frequency tests",
            key=f"{scope_label}_mixed_mode",
        )

        if mixed_mode:
            target_freq = st.number_input(
                "Override target frequency (Hz)",
                value=float(target_freq),
                min_value=0.1,
                step=10.0,
                key=f"{scope_label}_target_freq",
            )

            for name, df in const_data.items():
                matched_const_data[name] = df
            for name, sdf in sweep_data.items():
                sliced = extract_frequency_slice(
                    sdf,
                    target_freq,
                    scoped_settings["freq_tol"],
                    signal_col,
                )
                if not sliced.empty:
                    matched_const_data[f"{name} @{target_freq:.0f}Hz"] = sliced

    if mixed_mode and matched_const_data:
        _render_mixed_comparison(
            matched_const_data,
            signal_col,
            target_freq,
            scoped_settings,
            active_plots,
        )
        st.divider()

    if n_sweep > 0:
        if n_const > 0:
            st.subheader(f"📈 Frequency-Sweep Tests ({n_sweep})")
        _render_sweep_analysis(
            sweep_data,
            binned_data,
            signal_col,
            scoped_settings,
            active_plots,
        )

    if n_const > 0 and not mixed_mode:
        if n_sweep > 0:
            st.subheader(f"📊 Constant-Frequency Tests ({n_const})")
        _render_constant_analysis(
            const_data,
            signal_col,
            scoped_settings,
            active_plots,
        )


def _render_single_pump_analysis(
    S: dict,
    run_dirs: list,
    run_names: list[str],
) -> None:
    """Analyse one pump's own tests or saved sub-groups."""
    reg = _get_registry()
    if not reg.pumps:
        st.info(
            "No pumps defined yet. Go to **Manage Groups** or sync from the "
            "experiment log first."
        )
        return

    pump_names = sorted(reg.pumps.keys())
    pump_name = st.selectbox("🔧 Pump", pump_names, key="single_pump_name")
    pump = reg.pumps[pump_name]
    available_tests = [t.folder for t in pump.tests if t.folder in run_names]
    missing_tests = [t.folder for t in pump.tests if t.folder not in run_names]

    st.caption(
        f"Linked tests: {len(pump.tests)} total · "
        f"{len(available_tests)} available in this data folder · "
        f"{len(pump.sub_groups)} saved group(s)"
    )
    if missing_tests:
        st.warning(
            f"{len(missing_tests)} linked test(s) are not present in the current "
            "data folder and will be skipped."
        )

    scope = st.radio(
        "What would you like to analyse?",
        [
            "Analyze tests in this pump",
            "Analyze one saved group",
            "Compare saved groups",
        ],
        horizontal=True,
        key="single_pump_scope",
    )

    if scope == "Compare saved groups":
        active_plots = _render_plot_selector(
            "Saved-group comparison output",
            widget_key="analysis_plots_single_pump_groups",
            allowed_plot_ids=COLLECTION_ANALYSIS_PLOTS,
            default_plot_ids=COLLECTION_ANALYSIS_PLOTS,
        )
        _render_save_analysis_config_form(
            reg,
            S,
            active_plots,
            key_suffix=f"single_pump_groups_{_slugify(pump_name)}",
        )
        if not pump.sub_groups:
            st.info("This pump has no saved groups yet.")
            return
        group_names = list(pump.sub_groups.keys())
        chosen_groups = st.multiselect(
            "Groups to compare",
            group_names,
            default=group_names[: min(2, len(group_names))],
            key="single_pump_group_compare",
        )
        if not chosen_groups:
            st.info("Choose at least one saved group.")
            return

        collections = {
            group_name: [
                test_name
                for test_name in pump.sub_groups[group_name].tests
                if test_name in run_names
            ]
            for group_name in chosen_groups
        }
        _render_collection_comparison(
            collections,
            run_dirs,
            run_names,
            S,
            active_plots,
            collection_label="group",
            scope_label=f"{pump_name}_group_compare",
        )
        return

    active_plots = _render_plot_selector(
        "Single-pump output",
        widget_key="analysis_plots_single_pump_tests",
        allowed_plot_ids=TEST_ANALYSIS_PLOTS,
        default_plot_ids=DEFAULT_TEST_ANALYSIS_PLOTS,
    )
    _render_save_analysis_config_form(
        reg,
        S,
        active_plots,
        key_suffix=f"single_pump_tests_{_slugify(pump_name)}",
    )

    selected_names: list[str] = []
    selection_label = f"{pump_name}_all_tests"

    if scope == "Analyze one saved group":
        if not pump.sub_groups:
            st.info("This pump has no saved groups yet.")
            return
        group_name = st.selectbox(
            "Saved group",
            list(pump.sub_groups.keys()),
            key="single_pump_group_one",
        )
        group = pump.sub_groups[group_name]
        selected_names = [test for test in group.tests if test in run_names]
        selection_label = f"{pump_name}_{group_name}"
        st.caption(
            f"Group **{group_name}** contains {len(group.tests)} test(s); "
            f"{len(selected_names)} are available in the current data folder."
        )
        if group.description:
            st.caption(group.description)
    else:
        selection_mode = st.radio(
            "Tests to include",
            ["All linked tests", "Pick linked tests manually"],
            horizontal=True,
            key="single_pump_test_selection_mode",
        )
        if selection_mode == "All linked tests":
            selected_names = list(available_tests)
        else:
            c1, c2, c3 = st.columns([3, 0.7, 0.7])
            with c2:
                if st.button("All", key="single_pump_all"):
                    st.session_state["single_pump_manual_tests"] = list(available_tests)
                    st.rerun()
            with c3:
                if st.button("Clear", key="single_pump_clear"):
                    st.session_state["single_pump_manual_tests"] = []
                    st.rerun()
            with c1:
                selected_names = st.multiselect(
                    "Linked tests",
                    available_tests,
                    key="single_pump_manual_tests",
                )
            selection_label = f"{pump_name}_manual_tests"

        _render_save_pump_group_form(
            reg,
            pump_name,
            selected_names,
            key_suffix=_slugify(pump_name),
        )

    if not selected_names:
        st.info("Select at least one test in this pump.")
        return

    _render_selected_test_analysis(
        selected_names,
        run_dirs,
        run_names,
        S,
        active_plots,
        unknown_key_prefix="single_pump_utp",
        scope_label=selection_label,
        context_label="single-pump analysis",
    )


def _detect_target_freq(const_freqs: dict[str, float]) -> float:
    """Pick the most common constant frequency across tests."""
    if not const_freqs:
        return DEFAULT_CONSTANT_FREQUENCY_HZ
    counts = Counter(round(f, 1) for f in const_freqs.values())
    return counts.most_common(1)[0][0]


# ════════════════════════════════════════════════════════════════════════
# SHARED DATA LOADER
# ════════════════════════════════════════════════════════════════════════

def _load_tests(
    selected_names: list[str],
    run_dirs: list,
    run_names: list[str],
    S: dict,
) -> tuple | None:
    """Load and classify all selected test folders.

    Returns
    -------
    (all_data, sweep_data, binned_data, const_data, signal_col,
     errors, test_types, const_freqs)
    or ``None`` if nothing loaded.
    """
    all_data: dict[str, pd.DataFrame] = {}
    sweep_data: dict[str, pd.DataFrame] = {}
    binned_data: dict[str, pd.DataFrame] = {}
    const_data: dict[str, pd.DataFrame] = {}
    load_errors: list[str] = []
    signal_col: str | None = None
    test_types: dict[str, str] = {}
    const_freqs: dict[str, float] = {}

    with st.spinner(f"Loading {len(selected_names)} test(s)…"):
        for name in selected_names:
            idx = run_names.index(name) if name in run_names else -1
            if idx < 0:
                load_errors.append(f"{name}: not found in data folder")
                continue
            run_dir = run_dirs[idx]

            try:
                pick = pick_best_csv(run_dir)
                df = load_csv_cached(str(pick.csv_path))
                if df.empty:
                    load_errors.append(f"{name}: empty CSV")
                    continue

                time_col = guess_time_column(df)
                sig_col = guess_signal_column(df, "flow")
                if not time_col or not sig_col:
                    load_errors.append(f"{name}: cannot detect columns")
                    continue
                if signal_col is None:
                    signal_col = sig_col

                time_fmt = detect_time_format(df, time_col)
                ts_df = prepare_time_series_data(
                    df, time_col, sig_col,
                    parse_time=(time_fmt == "absolute_timestamp"),
                )
                all_data[name] = ts_df

                # Classify
                ttype, _, _ = detect_test_type(name, df, data_root=run_dir.parent)
                test_types[name] = ttype
                has_freq = "freq_set_hz" in df.columns

                if (ttype == "sweep"
                        or (has_freq and df["freq_set_hz"].dropna().nunique() > 1)):
                    # ── Sweep path ──────────────────────────────────────
                    spec = parse_sweep_spec_from_name(name)
                    if has_freq or (spec and spec.duration_s > 0):
                        sweep_df = prepare_sweep_data(
                            ts_df, time_col, sig_col,
                            spec=spec,
                            parse_time=(time_fmt == "absolute_timestamp"),
                            full_df=df if has_freq else None,
                        )
                        sweep_data[name] = sweep_df
                        try:
                            binned = bin_by_frequency(
                                sweep_df, value_col=sig_col,
                                freq_col="Frequency", bin_hz=S["bin_hz"],
                            )
                            binned_data[name] = binned
                        except Exception as be:
                            load_errors.append(f"{name}: binning — {be}")
                    else:
                        # Sweep by name but no usable freq data → treat as const
                        const_data[name] = ts_df
                        cf = detect_constant_frequency(
                            df,
                            name,
                            data_root=run_dir.parent,
                        )
                        if cf:
                            const_freqs[name] = cf
                else:
                    # ── Constant frequency path ─────────────────────────
                    const_data[name] = ts_df
                    cf = detect_constant_frequency(
                        df,
                        name,
                        data_root=run_dir.parent,
                    )
                    if cf:
                        const_freqs[name] = cf

            except Exception as e:
                load_errors.append(f"{name}: {e}")

    if not all_data:
        st.error("❌ No tests loaded successfully.")
        return None

    if signal_col is None:
        signal_col = "flow"

    return (
        all_data, sweep_data, binned_data, const_data,
        signal_col, load_errors, test_types, const_freqs,
    )


# ════════════════════════════════════════════════════════════════════════
# SWEEP ANALYSIS TABS
# ════════════════════════════════════════════════════════════════════════

def _render_sweep_analysis(
    sweep_data: dict[str, pd.DataFrame],
    binned_data: dict[str, pd.DataFrame],
    signal_col: str,
    S: dict,
    active_plots: list[str],
) -> None:
    """Render the selected sweep-analysis tabs."""
    n_binned = len(binned_data)
    tab_specs: list[tuple[str, str]] = []
    if "individual_sweeps" in active_plots:
        tab_specs.append(("📊 Individual", "individual"))
    if {"sweep_overlay", "sweep_relative"} & set(active_plots):
        tab_specs.append(("🔀 Combined", "combined"))
    if "global_average" in active_plots:
        tab_specs.append(("📏 Average", "average"))
    if "raw_points" in active_plots:
        tab_specs.append(("⚡ Raw Points", "raw_points"))
    if {"summary_table", "boxplots", "histograms", "correlation"} & set(active_plots):
        tab_specs.append(("📦 EDA", "eda"))
    if "std_vs_mean" in active_plots:
        tab_specs.append(("📐 Std vs Mean", "std_vs_mean"))
    if "best_region" in active_plots:
        tab_specs.append(("🎯 Best Region", "best_region"))

    if not tab_specs:
        st.info("No sweep analyses selected.")
        return

    tabs = st.tabs([label for label, _ in tab_specs])

    for tab, (_, tab_key) in zip(tabs, tab_specs):
        with tab:
            if tab_key == "individual":
                st.subheader("Individual Test Results")
                if not binned_data:
                    st.info("No frequency-binned data available.")
                    continue
                for name in binned_data:
                    with st.expander(f"📈 {name}", expanded=(n_binned <= 3)):
                        fig_b = plot_sweep_binned(
                            binned_data[name],
                            title=f"{name} — Binned (Δf = {S['bin_hz']:g} Hz)",
                            mode=S["plot_mode"],
                            marker_size=S["marker_sz"],
                            show_error_bars=S["err_bars"],
                        )
                        st.plotly_chart(fig_b, use_container_width=True)
                        _maybe_export(fig_b, f"{_slugify(name)}_binned.html", S)

                        if name in sweep_data:
                            fig_sw = plot_per_test_sweeps(
                                sweep_data[name],
                                signal_col=signal_col,
                                bin_hz=S["bin_hz"],
                                title=f"{name} — Per Sweep",
                                mode=S["plot_mode"],
                                marker_size=max(1, S["marker_sz"] - 1),
                            )
                            st.plotly_chart(fig_sw, use_container_width=True)
                            _maybe_export(
                                fig_sw,
                                f"{_slugify(name)}_per_sweep.html",
                                S,
                            )

                        if name in sweep_data and signal_col in sweep_data[name].columns:
                            s = sweep_data[name][signal_col].dropna()
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Mean", f"{s.mean():.2f} µL/min")
                            c2.metric("Std", f"{s.std():.2f} µL/min")
                            c3.metric("Min", f"{s.min():.2f} µL/min")
                            c4.metric("Max", f"{s.max():.2f} µL/min")

            elif tab_key == "combined":
                st.subheader("Combined Frequency Sweep Comparison")
                if n_binned < 1:
                    st.info("Need at least 1 test with frequency data.")
                    continue
                if "sweep_overlay" in active_plots:
                    fig_comb = plot_combined_overlay(
                        binned_data,
                        show_error_bars=S["err_bars"],
                        mode=S["plot_mode"],
                        marker_size=S["marker_sz"],
                    )
                    st.plotly_chart(fig_comb, use_container_width=True)
                    _maybe_export(fig_comb, "combined_overlay.html", S)
                if "sweep_relative" in active_plots:
                    if "sweep_overlay" in active_plots:
                        st.divider()
                    st.subheader("Relative (0–100 %) Comparison")
                    fig_rel = plot_relative_comparison(
                        binned_data,
                        mode=S["plot_mode"],
                        marker_size=S["marker_sz"],
                    )
                    st.plotly_chart(fig_rel, use_container_width=True)
                    _maybe_export(fig_rel, "relative_comparison.html", S)

            elif tab_key == "average":
                st.subheader("Global Average Across Tests")
                if n_binned < 2:
                    st.info("Need ≥ 2 tests for averaging.")
                    continue
                fig_avg, avg_df = plot_global_average(
                    binned_data,
                    bin_hz=S["avg_bin_hz"],
                    mode=S["plot_mode"],
                    marker_size=S["marker_sz"],
                    show_error_bars=S["err_bars"],
                )
                st.plotly_chart(fig_avg, use_container_width=True)
                _maybe_export(fig_avg, "global_average.html", S)
                with st.expander("📋 Average Curve Data"):
                    st.dataframe(avg_df, use_container_width=True, hide_index=True)

            elif tab_key == "raw_points":
                st.subheader("All Raw Data Points")
                if not sweep_data:
                    st.info("No frequency data.")
                    continue
                if S["show_all_points"]:
                    raw_points = dict(sweep_data)
                    total_pts = sum(len(d) for d in raw_points.values())
                    st.caption(f"Showing all {total_pts:,} points.")
                else:
                    raw_points: dict[str, pd.DataFrame] = {}
                    for name, df in sweep_data.items():
                        raw_points[name] = (
                            df.sample(n=S["max_raw"], random_state=42)
                            if len(df) > S["max_raw"] else df
                        )
                    total_pts = sum(len(d) for d in raw_points.values())
                    st.caption(
                        f"Showing {total_pts:,} points "
                        f"(capped at {S['max_raw']:,}/test)"
                    )
                fig_raw = plot_all_raw_points(
                    raw_points,
                    freq_col="Frequency",
                    signal_col=signal_col,
                    marker_size=S["marker_sz"],
                    opacity=S["opacity"],
                )
                st.plotly_chart(fig_raw, use_container_width=True)
                _maybe_export(fig_raw, "all_raw_points.html", S)

            elif tab_key == "eda":
                st.subheader("Exploratory Data Analysis")
                eda_data = {
                    name: df
                    for name, df in sweep_data.items()
                    if signal_col in df.columns
                }
                if not eda_data:
                    eda_data = dict(sweep_data)

                if "summary_table" in active_plots:
                    st.markdown("### 📋 Summary Statistics")
                    stats = build_summary_table(eda_data, signal_col=signal_col)
                    if not stats.empty:
                        st.dataframe(
                            stats,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "CV (%)": st.column_config.NumberColumn(format="%.2f"),
                            },
                        )

                if "boxplots" in active_plots or "histograms" in active_plots:
                    col_l, col_r = st.columns(2)
                    if "boxplots" in active_plots:
                        with col_l:
                            st.markdown("### 📦 Boxplots")
                            fig_box = plot_combined_boxplots(
                                eda_data,
                                signal_col=signal_col,
                            )
                            st.plotly_chart(fig_box, use_container_width=True)
                            _maybe_export(fig_box, "eda_boxplots.html", S)
                    if "histograms" in active_plots:
                        with col_r:
                            st.markdown("### 📊 Histograms")
                            nbins = st.slider(
                                "Bins",
                                10,
                                200,
                                50,
                                key=f"{S['export_scope']}_eda_bins",
                            )
                            fig_hist = plot_combined_histograms(
                                eda_data,
                                signal_col=signal_col,
                                nbins=nbins,
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                            _maybe_export(fig_hist, "eda_histograms.html", S)

                if "correlation" in active_plots:
                    st.markdown("### 🔗 Inter-Test Correlation")
                    if n_binned >= 2:
                        fig_corr = plot_correlation_heatmap(binned_data)
                        st.plotly_chart(fig_corr, use_container_width=True)
                        _maybe_export(fig_corr, "correlation_heatmap.html", S)
                    else:
                        st.info("Need at least 2 binned tests for correlation.")

            elif tab_key == "std_vs_mean":
                st.subheader("Variability: Std vs Mean")
                if not binned_data:
                    st.info("No binned data.")
                    continue
                fig_svm = plot_std_vs_mean(
                    binned_data,
                    marker_size=S["marker_sz"],
                )
                st.plotly_chart(fig_svm, use_container_width=True)
                _maybe_export(fig_svm, "std_vs_mean.html", S)
                with st.expander("ℹ️ Interpretation guide"):
                    st.markdown(
                        """
| Pattern | Meaning |
|---------|---------|
| **R² ≈ 1** | Noise scales with flow (proportional) |
| **R² ≈ 0** | Constant noise regardless of flow |
| Below trend line | More stable than expected |
| Above trend line | Less predictable than expected |
| Steep slope | Higher flow → disproportionately more noise |
| Flat slope | Noise similar across all flow levels |
"""
                    )

            elif tab_key == "best_region":
                st.subheader("Best Operating Region")
                if not binned_data:
                    st.info("No binned data.")
                    continue
                fig_stab, best_df = plot_stability_cloud(
                    binned_data,
                    mean_threshold_pct=float(S["mean_pct"]),
                    std_threshold_pct=float(S["std_pct"]),
                    marker_size=S["marker_sz"],
                )
                st.plotly_chart(fig_stab, use_container_width=True)
                st.markdown(
                    f"**Method:** Top {S['mean_pct']}% by mean flow → orange.  "
                    f"Bottom {S['std_pct']}% by std within those → 🟢 green stars."
                )
                if not best_df.empty:
                    st.markdown("### 🏆 Best Operating Points")
                    st.dataframe(
                        best_df[["test", "freq_center", "mean", "std"]]
                        .rename(columns={
                            "test": "Test",
                            "freq_center": "Frequency (Hz)",
                            "mean": "Mean (µL/min)",
                            "std": "Std (µL/min)",
                        })
                        .sort_values("Mean (µL/min)", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )
                _maybe_export(fig_stab, "stability_cloud.html", S)


# ════════════════════════════════════════════════════════════════════════
# CONSTANT-FREQ ANALYSIS TABS
# ════════════════════════════════════════════════════════════════════════

def _render_constant_analysis(
    const_data: dict[str, pd.DataFrame],
    signal_col: str,
    S: dict,
    active_plots: list[str],
) -> None:
    """Render comparison tabs for constant-frequency tests."""
    tab_specs: list[tuple[str, str]] = []
    if "boxplots" in active_plots:
        tab_specs.append(("📦 Boxplots", "boxplots"))
    if "histograms" in active_plots:
        tab_specs.append(("📊 Histograms", "histograms"))
    if "summary_table" in active_plots:
        tab_specs.append(("📋 Summary", "summary"))

    if not tab_specs:
        st.info("No constant-frequency analyses selected.")
        return

    tabs = st.tabs([label for label, _ in tab_specs])
    for tab, (_, tab_key) in zip(tabs, tab_specs):
        with tab:
            if tab_key == "boxplots":
                fig = plot_combined_boxplots(
                    const_data,
                    signal_col=signal_col,
                    title="Constant-Frequency Flow Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)
                _maybe_export(fig, "constant_boxplots.html", S)
            elif tab_key == "histograms":
                nbins = st.slider(
                    "Bins",
                    10,
                    200,
                    50,
                    key=f"{S['export_scope']}_const_bins",
                )
                fig = plot_combined_histograms(
                    const_data,
                    signal_col=signal_col,
                    nbins=nbins,
                    title="Constant-Frequency Flow Histograms",
                )
                st.plotly_chart(fig, use_container_width=True)
                _maybe_export(fig, "constant_histograms.html", S)
            elif tab_key == "summary":
                stats = build_summary_table(const_data, signal_col=signal_col)
                if not stats.empty:
                    st.dataframe(
                        stats,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "CV (%)": st.column_config.NumberColumn(format="%.2f"),
                        },
                    )
                else:
                    st.info("No signal data available.")


# ════════════════════════════════════════════════════════════════════════
# MIXED-TYPE COMPARISON
# ════════════════════════════════════════════════════════════════════════

def _render_mixed_comparison(
    matched_data: dict[str, pd.DataFrame],
    signal_col: str,
    target_freq: float,
    S: dict,
    active_plots: list[str],
) -> None:
    """Render cross-type comparison at the matched frequency."""
    st.subheader(f"🔀 Cross-Type Comparison at {target_freq:.0f} Hz")
    st.caption(
        f"Constant-frequency tests shown as-is.  Sweep tests filtered to "
        f"{target_freq:.0f} ± {S['freq_tol']:.0f} Hz."
    )
    tab_specs: list[tuple[str, str]] = []
    if "boxplots" in active_plots:
        tab_specs.append(("📦 Boxplots", "boxplots"))
    if "histograms" in active_plots:
        tab_specs.append(("📊 Histograms", "histograms"))
    if "summary_table" in active_plots:
        tab_specs.append(("📋 Summary", "summary"))

    if not tab_specs:
        st.info("No mixed-type comparison analyses selected.")
        return

    tabs = st.tabs([label for label, _ in tab_specs])
    for tab, (_, tab_key) in zip(tabs, tab_specs):
        with tab:
            if tab_key == "boxplots":
                fig = plot_combined_boxplots(
                    matched_data,
                    signal_col=signal_col,
                    title=f"Flow at {target_freq:.0f} Hz — All Tests",
                )
                st.plotly_chart(fig, use_container_width=True)
                _maybe_export(fig, "mixed_boxplots.html", S)
            elif tab_key == "histograms":
                nbins = st.slider(
                    "Bins",
                    10,
                    200,
                    50,
                    key=f"{S['export_scope']}_mixed_bins",
                )
                fig = plot_combined_histograms(
                    matched_data,
                    signal_col=signal_col,
                    nbins=nbins,
                    title=f"Flow Distribution at {target_freq:.0f} Hz",
                )
                st.plotly_chart(fig, use_container_width=True)
                _maybe_export(fig, "mixed_histograms.html", S)
            elif tab_key == "summary":
                stats = build_summary_table(matched_data, signal_col=signal_col)
                if not stats.empty:
                    st.dataframe(
                        stats,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "CV (%)": st.column_config.NumberColumn(format="%.2f"),
                        },
                    )


# ════════════════════════════════════════════════════════════════════════
# MODE 2 — COMPARE PUMPS
# ════════════════════════════════════════════════════════════════════════

def _load_collection_data(
    collections: dict[str, list[str]],
    run_dirs: list,
    run_names: list[str],
    S: dict,
) -> tuple[
    dict[str, dict[str, pd.DataFrame]],
    dict[str, dict[str, pd.DataFrame]],
    dict[str, dict[str, pd.DataFrame]],
    list[str],
    str,
]:
    """Load nested sweep/constant data for arbitrary named collections."""
    run_map = {path.name: path for path in run_dirs}
    collection_sweep_binned: dict[str, dict[str, pd.DataFrame]] = {}
    collection_sweep_raw: dict[str, dict[str, pd.DataFrame]] = {}
    collection_const_data: dict[str, dict[str, pd.DataFrame]] = {}
    load_errors: list[str] = []
    signal_col: str | None = None

    total_requested = sum(len(test_names) for test_names in collections.values())
    with st.spinner(f"Loading {total_requested} test(s)…"):
        for collection_name, test_names in collections.items():
            collection_sweep_binned[collection_name] = {}
            collection_sweep_raw[collection_name] = {}
            collection_const_data[collection_name] = {}

            for test_name in test_names:
                run_dir = run_map.get(test_name)
                if run_dir is None:
                    load_errors.append(
                        f"{collection_name}/{test_name}: not in data folder"
                    )
                    continue

                try:
                    pick = pick_best_csv(run_dir)
                    df = load_csv_cached(str(pick.csv_path))
                    if df.empty:
                        load_errors.append(f"{collection_name}/{test_name}: empty")
                        continue

                    time_col = guess_time_column(df)
                    sig_col = guess_signal_column(df, "flow")
                    if not time_col or not sig_col:
                        load_errors.append(
                            f"{collection_name}/{test_name}: cannot detect columns"
                        )
                        continue
                    if signal_col is None:
                        signal_col = sig_col

                    time_fmt = detect_time_format(df, time_col)
                    ts_df = prepare_time_series_data(
                        df,
                        time_col,
                        sig_col,
                        parse_time=(time_fmt == "absolute_timestamp"),
                    )

                    test_type, _, _ = detect_test_type(
                        test_name,
                        df,
                        data_root=run_dir.parent,
                    )
                    has_freq = "freq_set_hz" in df.columns
                    spec = parse_sweep_spec_from_name(test_name)

                    if test_type == "sweep" or (
                        has_freq and df["freq_set_hz"].dropna().nunique() > 1
                    ):
                        if has_freq or (spec and spec.duration_s > 0):
                            sweep_df = prepare_sweep_data(
                                ts_df,
                                time_col,
                                sig_col,
                                spec=spec,
                                parse_time=(time_fmt == "absolute_timestamp"),
                                full_df=df if has_freq else None,
                            )
                            collection_sweep_raw[collection_name][test_name] = sweep_df
                            try:
                                collection_sweep_binned[collection_name][test_name] = (
                                    bin_by_frequency(
                                        sweep_df,
                                        value_col=sig_col,
                                        freq_col="Frequency",
                                        bin_hz=S["bin_hz"],
                                    )
                                )
                            except Exception as exc:
                                load_errors.append(
                                    f"{collection_name}/{test_name}: binning — {exc}"
                                )
                        else:
                            collection_const_data[collection_name][test_name] = ts_df
                    else:
                        collection_const_data[collection_name][test_name] = ts_df

                except Exception as exc:
                    load_errors.append(f"{collection_name}/{test_name}: {exc}")

    return (
        collection_sweep_binned,
        collection_sweep_raw,
        collection_const_data,
        load_errors,
        signal_col or "flow",
    )


def _render_collection_comparison(
    collections: dict[str, list[str]],
    run_dirs: list,
    run_names: list[str],
    S: dict,
    active_plots: list[str],
    *,
    collection_label: str,
    scope_label: str,
) -> None:
    """Compare named collections such as pumps or saved groups."""
    if not active_plots:
        st.info("Select at least one plot or analysis to run.")
        return

    all_tests = sorted(
        {
            test_name
            for test_names in collections.values()
            for test_name in test_names
        }
    )
    if all_tests:
        _, unknowns = classify_tests_quick(all_tests, run_dirs, run_names)
        if unknowns:
            all_classified = render_unknown_test_prompt(
                unknowns,
                run_dirs,
                run_names,
                key_prefix=f"{scope_label}_utp",
            )
            if not all_classified:
                st.info(
                    "💡 Classify the unknown test(s) above, then the "
                    f"{collection_label} comparison will load automatically."
                )
                return

    scoped_settings = dict(S)
    scoped_settings["export_scope"] = scope_label
    _render_export_destination(scoped_settings)

    (
        collection_sweep_binned,
        collection_sweep_raw,
        collection_const_data,
        load_errors,
        signal_col,
    ) = _load_collection_data(collections, run_dirs, run_names, scoped_settings)

    if load_errors:
        with st.expander(f"⚠️ {len(load_errors)} load issue(s)", expanded=False):
            for err in load_errors:
                st.warning(err)

    n_sw = sum(1 for test_map in collection_sweep_binned.values() if test_map)
    n_cf = sum(1 for test_map in collection_const_data.values() if test_map)
    total = sum(
        len(collection_sweep_binned.get(name, {})) + len(collection_const_data.get(name, {}))
        for name in collections
    )

    if total == 0:
        st.error("❌ No data loaded for the selected collections.")
        return

    st.success(
        f"✅ **{total}** tests across **{len(collections)}** {collection_label}(s) — "
        f"**{n_sw}** sweep, **{n_cf}** constant"
    )

    if n_sw > 0 and n_cf > 0:
        st.warning(
            "⚠️ Mixed sweep and constant-frequency tests were found. "
            "Results are shown separately by test type."
        )

    with st.expander(f"📋 {collection_label.title()} contents", expanded=False):
        for collection_name in collections:
            sweep_tests = list(collection_sweep_binned.get(collection_name, {}).keys())
            const_tests = list(collection_const_data.get(collection_name, {}).keys())
            st.markdown(f"**{collection_name}**")
            if sweep_tests:
                st.markdown(
                    f"  - Sweep ({len(sweep_tests)}): "
                    + ", ".join(f"`{test}`" for test in sweep_tests)
                )
            if const_tests:
                st.markdown(
                    f"  - Constant ({len(const_tests)}): "
                    + ", ".join(f"`{test}`" for test in const_tests)
                )
            if not sweep_tests and not const_tests:
                st.markdown("  - _no loaded tests_")

    tab_specs: list[tuple[str, str]] = []
    if n_sw and "sweep_overlay" in active_plots:
        tab_specs.append(("🔀 Sweep Overlay", "sweep_overlay"))
    if n_sw and "sweep_relative" in active_plots:
        tab_specs.append(("📏 Sweep Relative", "sweep_relative"))
    if n_cf and "boxplots" in active_plots:
        tab_specs.append(("📦 Const Boxplots", "boxplots"))
    if n_cf and "histograms" in active_plots:
        tab_specs.append(("📊 Const Histograms", "histograms"))
    if "summary_table" in active_plots:
        tab_specs.append(("📋 Summary", "summary"))

    if not tab_specs:
        st.info("No applicable collection-comparison analyses were selected.")
        return

    tabs = st.tabs([label for label, _ in tab_specs])
    for tab, (_, tab_key) in zip(tabs, tab_specs):
        with tab:
            if tab_key == "sweep_overlay":
                st.subheader(f"Frequency Sweep — {collection_label.title()} Comparison")
                show_individual = st.checkbox(
                    "Show individual test traces",
                    value=False,
                    key=f"{scope_label}_show_individual",
                )
                fig = plot_bar_sweep_overlay(
                    collection_sweep_binned,
                    show_error_bars=scoped_settings["err_bars"],
                    show_individual=show_individual,
                    mode=scoped_settings["plot_mode"],
                    marker_size=scoped_settings["marker_sz"],
                )
                st.plotly_chart(fig, use_container_width=True)
                _maybe_export(fig, f"{collection_label}_sweep_overlay.html", scoped_settings)

            elif tab_key == "sweep_relative":
                st.subheader(
                    f"Relative (0–100 %) — {collection_label.title()} Comparison"
                )
                fig = plot_bar_sweep_relative(
                    collection_sweep_binned,
                    mode=scoped_settings["plot_mode"],
                    marker_size=scoped_settings["marker_sz"],
                )
                st.plotly_chart(fig, use_container_width=True)
                _maybe_export(fig, f"{collection_label}_sweep_relative.html", scoped_settings)

            elif tab_key == "boxplots":
                st.subheader(
                    f"Constant-Frequency — {collection_label.title()} Comparison"
                )
                view = st.radio(
                    "View",
                    [
                        f"Per-test (grouped by {collection_label})",
                        f"Aggregated per {collection_label}",
                    ],
                    horizontal=True,
                    key=f"{scope_label}_const_view",
                )
                if view.startswith("Per-test"):
                    fig = plot_bar_constant_boxplots(
                        collection_const_data,
                        signal_col=signal_col,
                    )
                else:
                    fig = plot_bar_constant_aggregated(
                        collection_const_data,
                        signal_col=signal_col,
                    )
                st.plotly_chart(fig, use_container_width=True)
                _maybe_export(fig, f"{collection_label}_constant_boxplots.html", scoped_settings)

            elif tab_key == "histograms":
                st.subheader("Constant-Frequency Histograms")
                nbins = st.slider(
                    "Bins",
                    10,
                    200,
                    50,
                    key=f"{scope_label}_hist_bins",
                )
                fig = plot_bar_constant_histograms(
                    collection_const_data,
                    signal_col=signal_col,
                    nbins=nbins,
                )
                st.plotly_chart(fig, use_container_width=True)
                _maybe_export(fig, f"{collection_label}_constant_histograms.html", scoped_settings)

            elif tab_key == "summary":
                st.subheader(f"Summary Statistics per {collection_label.title()}")
                dfs: list[pd.DataFrame] = []
                sweep_pool = {
                    name: test_map
                    for name, test_map in collection_sweep_raw.items()
                    if test_map
                }
                const_pool = {
                    name: test_map
                    for name, test_map in collection_const_data.items()
                    if test_map
                }
                if sweep_pool:
                    dfs.append(
                        build_bar_summary_table(
                            sweep_pool,
                            signal_col=signal_col,
                            test_type="Sweep",
                        )
                    )
                if const_pool:
                    dfs.append(
                        build_bar_summary_table(
                            const_pool,
                            signal_col=signal_col,
                            test_type="Constant",
                        )
                    )
                if dfs:
                    summary = pd.concat(dfs, ignore_index=True)
                    st.dataframe(
                        summary,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "CV (%)": st.column_config.NumberColumn(format="%.2f"),
                        },
                    )
                else:
                    st.info("No data available for summary.")


def _render_compare_bars(
    S: dict,
    run_dirs: list,
    run_names: list[str],
) -> None:
    """Compare Pumps mode — select pumps or shipment, load, compare."""
    reg = _get_registry()

    if not reg.pumps:
        st.info(
            "No pumps defined yet. Go to **Manage Groups** or sync from the "
            "experiment log first."
        )
        return

    active_plots = _render_plot_selector(
        "Pump comparison output",
        widget_key="analysis_plots_compare_pumps",
        allowed_plot_ids=COLLECTION_ANALYSIS_PLOTS,
        default_plot_ids=COLLECTION_ANALYSIS_PLOTS,
    )
    _render_save_analysis_config_form(
        reg,
        S,
        active_plots,
        key_suffix="compare_pumps",
    )

    sel = st.radio(
        "How would you like to select pumps?",
        ["Pick pumps manually", "Load a saved shipment"],
        horizontal=True,
        key="cb_sel",
    )

    selected_pump_names: list[str] = []
    scope_label = "pump_compare"

    if sel == "Load a saved shipment":
        if not reg.shipments:
            st.info(
                "No shipments saved. Create one on the **Manage Groups** page, "
                "or pick pumps manually."
            )
            return
        shipment_names = list(reg.shipments.keys())
        labels = [
            f"{name} ({reg.shipments[name].recipient or '—'})"
            for name in shipment_names
        ]
        chosen_idx = st.selectbox(
            "🚚 Shipment",
            range(len(shipment_names)),
            format_func=lambda idx: labels[idx],
            key="cb_ship",
        )
        shipment = reg.shipments[shipment_names[chosen_idx]]
        selected_pump_names = [name for name in shipment.pumps if name in reg.pumps]
        scope_label = f"shipment_{shipment.name}"
        st.markdown(
            f"**Shipment:** {shipment.name}  ·  "
            f"**Recipient:** {shipment.recipient or '—'}  ·  "
            f"**Pumps:** {', '.join(selected_pump_names) or 'none'}"
        )
    else:
        selected_pump_names = st.multiselect(
            "🔧 Pumps to compare",
            sorted(reg.pumps.keys()),
            key="cb_pumps",
        )
        if selected_pump_names:
            scope_label = f"pump_compare_{len(selected_pump_names)}"

    if not selected_pump_names:
        st.info("👆 Select at least one pump.")
        return
    if len(selected_pump_names) < 2:
        st.warning("Select **≥ 2 pumps** for a real side-by-side comparison.")

    collections = {
        pump_name: [test.folder for test in reg.pumps[pump_name].tests]
        for pump_name in selected_pump_names
    }
    _render_collection_comparison(
        collections,
        run_dirs,
        run_names,
        S,
        active_plots,
        collection_label="pump",
        scope_label=scope_label,
    )
