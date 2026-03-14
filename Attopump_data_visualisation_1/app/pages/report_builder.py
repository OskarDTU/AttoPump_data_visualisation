"""Report Builder page — compose and export professional report packages.

This page lets the user:

1. **Manage the Pump Registry** — import from a logbook
   spreadsheet (Excel / CSV), or manually add / edit / delete entries
   and test links.
2. **Configure a report** — select which pump entries to
   include, which comparison charts to generate, and set options.
3. **Preview** — see the report content live before exporting.
4. **Export** — download a self-contained HTML file to ~/Downloads.
5. **Audit trail** — view the full history of registry changes.

The page is divided into tabs:
  📋 Registry — view & manage pump entries
  📝 Build Report — compose and export
  📜 Audit Log — full change history
"""

from __future__ import annotations

import traceback
from collections import OrderedDict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus, unquote_plus

import numpy as np
import pandas as pd
import streamlit as st

from ..data.config import PLOT_BIN_WIDTH_HZ, PLOT_HEIGHT
from ..data.cache_warmup import (
    REPORT_WARMUP_JOB_KEY,
    delete_warmup_job,
    launch_warmup_worker,
    load_warmup_job,
    load_warmup_worker_state,
    retry_failed_warmup_tasks,
    run_report_warmup_job,
    save_warmup_job,
    set_warmup_job_paused,
    start_report_warmup_job,
    summarize_warmup_job,
    summarize_warmup_worker_state,
)
from ..data.data_processor import (
    explain_frequency_bin_recommendation,
    format_bin_choice_label,
    recommend_frequency_bin_widths,
)
from ..data.loader import (
    get_test_cache_context,
    load_and_classify_tests,
    load_prepared_test_data,
    resolve_data_path,
)
from ..data.persistent_cache import get_or_create_cached_test_figure
from ..data.pump_registry import (
    Pump,
    PumpSubGroup,
    PumpRegistry,
    TestLink,
    add_pump,
    import_from_dataframe,
    link_test,
    load_audit_log,
    load_registry,
    migrate_legacy_files,
    remove_pump,
    remove_sub_group,
    rename_pump,
    registry_storage_signature,
    save_registry,
    unlink_test,
    upsert_sub_group,
)
from ..reports.generator import (
    AxisBounds,
    COMPARISON_OPTIONS,
    ReportDefinition,
    ReportSection,
    build_report_html,
    delete_saved_report,
    load_report_definition,
    load_saved_reports,
    save_report,
    save_report_definition,
)
from ..plot_guidance import build_plot_guide_text, render_plot_guide
from .unknown_test_prompt import classify_tests_quick, render_unknown_test_prompt

# Lazy-import heavy plot modules only when needed
_analysis_plots = None
_bar_comparison_plots = None


def _get_analysis_plots():
    global _analysis_plots
    if _analysis_plots is None:
        from ..plots import analysis_plots as _ap
        _analysis_plots = _ap
    return _analysis_plots


def _get_bar_comparison_plots():
    global _bar_comparison_plots
    if _bar_comparison_plots is None:
        from ..plots import bar_comparison_plots as _bcp
        _bar_comparison_plots = _bcp
    return _bar_comparison_plots


# ────────────────────────────────────────────────────────────────────────
# Session state helpers
# ────────────────────────────────────────────────────────────────────────

def _get_registry() -> PumpRegistry:
    """Return the in-memory pump registry (loads from disk on first access)."""
    key = "_pump_registry"
    sig_key = "_pump_registry_signature"
    current_sig = registry_storage_signature()
    if (
        key not in st.session_state
        or st.session_state.get(sig_key) != current_sig
    ):
        st.session_state[key] = migrate_legacy_files()
        st.session_state[sig_key] = registry_storage_signature()
    return st.session_state[key]


def _persist_registry() -> None:
    """Flush the registry to disk."""
    save_registry(_get_registry())
    st.session_state["_pump_registry_signature"] = registry_storage_signature()


_SELECTION_MODE_OPTIONS = {
    "pumps": "Whole pumps",
    "sub_groups": "Saved pump sub-groups",
}
_SUB_GROUP_ENTRY_PREFIX = "__pump_sub_group__"
_DEFAULT_REPORT_TITLE = "AttoPump Test Report"
_DEFAULT_REPORT_COMPARISONS = [
    "sweep_overlay",
    "summary_table",
    "boxplots",
    "constant_time_series",
]
_PLOT_MODE_OPTIONS = ["lines+markers", "lines", "markers"]
_REPORT_PLOT_MODE_LABELS: OrderedDict[str, str] = OrderedDict(
    [
        ("sweep_overlay_target", "Frequency Sweep — Report Target Comparison"),
        ("sweep_overlay_tests", "Frequency Sweep — All Tests Overlay"),
        ("sweep_relative_target", "Relative Sweep (0–100 %) — Report Target Comparison"),
        ("sweep_relative_tests", "Relative Sweep (0–100 %) — All Tests"),
        ("individual_binned", "Individual Sweeps — Binned Mean"),
        ("individual_per_sweep", "Individual Sweeps — Per-Sweep Breakdown"),
        ("individual_raw_all_sweeps", "Individual Sweeps — Raw All-Sweeps Layer"),
        ("global_average", "Global Average Across Tests"),
        ("constant_time_series", "Constant-Frequency Flow vs Time"),
        ("raw_points", "All Raw Sweep Points"),
    ]
)
_DEFAULT_REPORT_PLOT_MODES = {
    key: "lines+markers" for key in _REPORT_PLOT_MODE_LABELS
}
_DEFAULT_REPORT_PLOT_MODES["individual_raw_all_sweeps"] = "markers"
_DEFAULT_REPORT_PLOT_MODES["raw_points"] = "markers"


def _report_plot_mode_state_key(plot_key: str) -> str:
    """Return the Streamlit widget key for one report plot-mode control."""
    return f"rb_mode_{plot_key}"


def _normalize_report_plot_mode(mode: object) -> str:
    """Coerce any stored plot mode onto the supported report options."""
    return str(mode) if str(mode) in _PLOT_MODE_OPTIONS else "lines+markers"


def _default_report_plot_mode_widgets() -> dict[str, str]:
    """Return per-plot mode defaults for a new report."""
    return {
        _report_plot_mode_state_key(plot_key): mode
        for plot_key, mode in _DEFAULT_REPORT_PLOT_MODES.items()
    }


def _report_plot_mode_widgets_from_definition(
    defn: ReportDefinition,
) -> dict[str, str]:
    """Return per-plot mode widget values for a loaded report template."""
    return {
        _report_plot_mode_state_key(plot_key): _normalize_report_plot_mode(
            defn.plot_modes.get(plot_key, defn.plot_mode)
        )
        for plot_key in _REPORT_PLOT_MODE_LABELS
    }


def _active_report_plot_mode_keys(
    selected_comparisons: list[str],
    *,
    include_raw_all_sweeps: bool,
) -> list[str]:
    """List the plot-mode controls relevant to the current report selection."""
    keys: list[str] = []
    if "sweep_overlay" in selected_comparisons:
        keys.extend(["sweep_overlay_target", "sweep_overlay_tests"])
    if "sweep_relative" in selected_comparisons:
        keys.extend(["sweep_relative_target", "sweep_relative_tests"])
    if "individual_sweeps" in selected_comparisons:
        keys.extend(["individual_binned", "individual_per_sweep"])
        if include_raw_all_sweeps:
            keys.append("individual_raw_all_sweeps")
    if "global_average" in selected_comparisons:
        keys.append("global_average")
    if "constant_time_series" in selected_comparisons:
        keys.append("constant_time_series")
    if "raw_points" in selected_comparisons:
        keys.append("raw_points")
    return keys


def _collect_report_plot_modes() -> dict[str, str]:
    """Collect the current per-plot mode settings from session state."""
    return {
        plot_key: _normalize_report_plot_mode(
            st.session_state.get(
                _report_plot_mode_state_key(plot_key),
                _DEFAULT_REPORT_PLOT_MODES[plot_key],
            )
        )
        for plot_key in _REPORT_PLOT_MODE_LABELS
    }


def _resolve_report_plot_mode(defn: ReportDefinition, plot_key: str) -> str:
    """Resolve the mode used for one concrete report plot family."""
    return _normalize_report_plot_mode(defn.plot_modes.get(plot_key, defn.plot_mode))


def _format_optional_numeric(value: float | None) -> str:
    """Render an optional numeric setting as a text-input value."""
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:g}"


def _encode_sub_group_entry_id(pump_name: str, group_name: str) -> str:
    """Create a stable report-builder entry ID for a saved pump sub-group."""
    return (
        f"{_SUB_GROUP_ENTRY_PREFIX}::"
        f"{quote_plus(pump_name)}::"
        f"{quote_plus(group_name)}"
    )


def _decode_sub_group_entry_id(entry_id: str) -> tuple[str, str] | None:
    """Decode a saved pump sub-group entry ID."""
    if not entry_id.startswith(f"{_SUB_GROUP_ENTRY_PREFIX}::"):
        return None
    parts = entry_id.split("::", 2)
    if len(parts) != 3:
        return None
    return unquote_plus(parts[1]), unquote_plus(parts[2])


def _resolve_report_entry(
    entry_id: str,
    reg: PumpRegistry,
    run_names: list[str],
) -> tuple[str, list[str], dict] | None:
    """Resolve one report target into a display label, folders, and metadata."""
    decoded = _decode_sub_group_entry_id(entry_id)
    if decoded is not None:
        pump_name, group_name = decoded
        pump = reg.pumps.get(pump_name)
        if pump is None:
            return None
        group = pump.sub_groups.get(group_name)
        if group is None:
            return None
        linked_tests = {t.folder: t for t in pump.tests}
        folders = [folder for folder in group.tests if folder in run_names]
        group_notes = " | ".join(
            part for part in [pump.notes.strip(), group.description.strip()] if part
        )
        tests = [
            asdict(linked_tests[folder])
            for folder in group.tests
            if folder in linked_tests
        ]
        return (
            f"{pump.name} / {group_name}",
            folders,
            {
                "display_name": f"{pump.name} / {group_name}",
                "notes": group_notes,
                "tests": tests,
            },
        )

    pump = reg.pumps.get(entry_id)
    if pump is None:
        return None
    folders = [t.folder for t in pump.tests if t.folder in run_names]
    return (
        pump.name,
        folders,
        {
            "display_name": pump.name,
            "notes": pump.notes,
            "tests": [asdict(t) for t in pump.tests],
        },
    )


def _template_widget_defaults(
    loaded_defn: ReportDefinition | None,
) -> dict[str, object]:
    """Build the widget-state defaults for a loaded report template."""
    def _axis_defaults(prefix: str, axis: AxisBounds) -> dict[str, str]:
        return {
            f"{prefix}_xmin": _format_optional_numeric(axis.x_min),
            f"{prefix}_xmax": _format_optional_numeric(axis.x_max),
            f"{prefix}_ymin": _format_optional_numeric(axis.y_min),
            f"{prefix}_ymax": _format_optional_numeric(axis.y_max),
        }

    if loaded_defn is None:
        defaults = {
            "rb_title": _DEFAULT_REPORT_TITLE,
            "rb_author": "",
            "rb_notes": "",
            "rb_selection_mode": "pumps",
            "rb_entries_pumps": [],
            "rb_entries_sub_groups": [],
            "rb_comparisons": list(_DEFAULT_REPORT_COMPARISONS),
            "rb_plot_bin": 5.0,
            "rb_avg_bin": 3.0,
            "rb_auto_avg_bin": True,
            "rb_err": True,
            "rb_indiv": False,
            "rb_raw_all_sweeps": True,
            "rb_overlay_entries": [],
            "rb_overlay_sweep_drilldown": False,
            "rb_mode": "lines+markers",
            "rb_marker_size": 6,
            "rb_opacity": 0.8,
            "rb_maxpts": 50_000,
            "rb_best_mean": 75,
            "rb_best_std": 10,
        }
        defaults.update(_default_report_plot_mode_widgets())
        defaults.update(_axis_defaults("rb_sweep_axis", AxisBounds()))
        defaults.update(_axis_defaults("rb_relative_axis", AxisBounds()))
        defaults.update(_axis_defaults("rb_time_axis", AxisBounds()))
        defaults.update(_axis_defaults("rb_variability_axis", AxisBounds()))
        return defaults

    selection_mode = (
        loaded_defn.selection_mode
        if loaded_defn.selection_mode in _SELECTION_MODE_OPTIONS
        else "pumps"
    )
    defaults = {
        "rb_title": loaded_defn.title,
        "rb_author": loaded_defn.author,
        "rb_notes": loaded_defn.notes,
        "rb_selection_mode": selection_mode,
        "rb_entries_pumps": list(loaded_defn.entry_ids) if selection_mode == "pumps" else [],
        "rb_entries_sub_groups": (
            list(loaded_defn.entry_ids) if selection_mode == "sub_groups" else []
        ),
        "rb_comparisons": [
            comp for comp in loaded_defn.comparisons if comp in COMPARISON_OPTIONS
        ] or list(_DEFAULT_REPORT_COMPARISONS),
        "rb_plot_bin": float(loaded_defn.bin_hz),
        "rb_avg_bin": float(loaded_defn.avg_bin_hz),
        "rb_auto_avg_bin": bool(loaded_defn.auto_use_recommended_avg_bin),
        "rb_err": bool(loaded_defn.show_error_bars),
        "rb_indiv": bool(loaded_defn.show_individual_tests),
        "rb_raw_all_sweeps": bool(loaded_defn.show_raw_all_sweeps),
        "rb_overlay_entries": list(loaded_defn.overlay_entry_ids),
        "rb_overlay_sweep_drilldown": bool(loaded_defn.overlay_include_sweep_drilldown),
        "rb_mode": loaded_defn.plot_mode,
        "rb_marker_size": int(loaded_defn.marker_size),
        "rb_opacity": float(loaded_defn.opacity),
        "rb_maxpts": int(loaded_defn.max_raw_points),
        "rb_best_mean": int(loaded_defn.mean_threshold_pct),
        "rb_best_std": int(loaded_defn.std_threshold_pct),
    }
    defaults.update(_report_plot_mode_widgets_from_definition(loaded_defn))
    defaults.update(_axis_defaults("rb_sweep_axis", loaded_defn.sweep_axis))
    defaults.update(_axis_defaults("rb_relative_axis", loaded_defn.relative_axis))
    defaults.update(_axis_defaults("rb_time_axis", loaded_defn.time_axis))
    defaults.update(_axis_defaults("rb_variability_axis", loaded_defn.variability_axis))
    return defaults


def _apply_template_to_session_state(
    load_marker: str,
    loaded_defn: ReportDefinition | None,
) -> None:
    """Apply loaded template values to Streamlit widget state once per change."""
    if st.session_state.get("_rb_loaded_template") == load_marker:
        return

    st.session_state["_rb_loaded_template"] = load_marker
    for key, value in _template_widget_defaults(loaded_defn).items():
        st.session_state[key] = value
    st.session_state.pop("rb_html", None)
    st.session_state.pop("_rb_bin_signature", None)
    st.session_state.pop("_rb_bin_recommendation", None)


@st.cache_data(show_spinner=False)
def _recommend_report_bin_cached(
    folder_names: tuple[str, ...],
    run_names: tuple[str, ...],
    run_dir_strs: tuple[str, ...],
) -> dict[str, float] | None:
    """Recommend a smoothing-oriented bin width for the selected report tests."""
    if not folder_names:
        return None

    run_map = {name: run_dir for name, run_dir in zip(run_names, run_dir_strs)}
    sweep_frames: list[pd.DataFrame] = []
    signal_col = "flow"

    for name in folder_names:
        run_dir_str = run_map.get(name)
        if run_dir_str is None:
            continue
        try:
            prepared = load_prepared_test_data(name, run_dir_str)
        except Exception:
            continue
        if prepared.sweep_data is None:
            continue
        signal_col = prepared.signal_col
        sweep_frames.append(prepared.sweep_data)

    if not sweep_frames:
        return None
    recommendation = recommend_frequency_bin_widths(
        sweep_frames,
        value_col=signal_col,
    )
    if (
        int(recommendation.get("test_series_count", 0)) < 2
        and int(recommendation.get("average_series_count", 0)) < 2
    ):
        return None
    return recommendation


def _reset_report_bin() -> None:
    """Reset the report smoothing bin to the current recommendation."""
    recommendation = st.session_state.get("_rb_bin_recommendation")
    if recommendation and int(recommendation.get("average_series_count", 0)) >= 2:
        st.session_state["rb_avg_bin"] = float(recommendation["average_bin_hz"])


def _reset_report_plot_bin() -> None:
    """Reset the report per-test bin to the current recommendation."""
    recommendation = st.session_state.get("_rb_bin_recommendation")
    if recommendation and int(recommendation.get("test_series_count", 0)) >= 2:
        st.session_state["rb_plot_bin"] = float(recommendation["test_bin_hz"])


def _comparisons_use_average_bin(selected_comparisons: list[str]) -> bool:
    """Return whether the current report needs the averaging bin."""
    return bool(set(selected_comparisons) & {"sweep_overlay", "sweep_relative", "global_average"})


def _ensure_report_bin_recommendation(
    *,
    selected_folders: list[str],
    run_names: list[str],
    run_dirs: list[Path],
) -> dict[str, float] | None:
    """Load or calculate the cached recommendation for the current selection."""
    recommendation = st.session_state.get("_rb_bin_recommendation")
    if recommendation is not None:
        return recommendation
    if not selected_folders:
        return None

    recommendation = _recommend_report_bin_cached(
        tuple(sorted(set(selected_folders))),
        tuple(run_names),
        tuple(str(path) for path in run_dirs),
    )
    if recommendation:
        st.session_state["_rb_bin_recommendation"] = recommendation
    else:
        st.session_state.pop("_rb_bin_recommendation", None)
    return recommendation


def _build_report_target_options(
    reg: PumpRegistry,
    run_names: list[str],
    selection_mode: str,
) -> OrderedDict[str, str]:
    """Build the selectable report targets for the current mode."""
    options: OrderedDict[str, str] = OrderedDict()
    if selection_mode == "sub_groups":
        for pump_name, pump in sorted(reg.pumps.items()):
            for group_name, group in sorted(pump.sub_groups.items()):
                available = sum(1 for folder in group.tests if folder in run_names)
                options[_encode_sub_group_entry_id(pump_name, group_name)] = (
                    f"{pump.name} / {group_name} "
                    f"({available}/{len(group.tests)} tests available)"
                )
    else:
        for pump_name, pump in sorted(reg.pumps.items()):
            available = sum(1 for test in pump.tests if test.folder in run_names)
            options[pump_name] = (
                f"{pump.name} ({available}/{len(pump.tests)} tests available)"
            )
    return options


def _normalize_report_selected_entries(
    widget_value: object,
    session_value: object,
    valid_options: Mapping[str, str],
) -> list[str]:
    """Normalize report-target selections from widget and session state."""

    def _coerce(raw: object) -> list[str]:
        if isinstance(raw, (list, tuple, set)):
            values = list(raw)
        elif raw in {None, ""}:
            values = []
        else:
            values = [raw]
        normalized: list[str] = []
        for item in values:
            key = str(item)
            if key in valid_options and key not in normalized:
                normalized.append(key)
        return normalized

    widget_selected = _coerce(widget_value)
    session_selected = _coerce(session_value)
    if widget_selected:
        return widget_selected
    return session_selected


def _collect_report_target_folders(
    entry_ids: list[str],
    reg: PumpRegistry,
    run_names: list[str],
) -> list[str]:
    """Resolve the selected report targets into the underlying test folders."""
    folders: list[str] = []
    for entry_id in entry_ids:
        resolved = _resolve_report_entry(entry_id, reg, run_names)
        if resolved is None:
            continue
        _, entry_folders, _ = resolved
        folders.extend(entry_folders)
    return folders


def _normalize_overlay_selected_entries(
    selected_entries: list[str],
    target_options: Mapping[str, str],
) -> list[str]:
    """Keep overlay-target focus state valid for the current report selection."""
    available_overlay_options = {
        entry_id: target_options[entry_id]
        for entry_id in selected_entries
        if entry_id in target_options
    }
    overlay_selected = _normalize_report_selected_entries(
        st.session_state.get("rb_overlay_entries", []),
        st.session_state.get("rb_overlay_entries", []),
        available_overlay_options,
    )
    if not overlay_selected and selected_entries:
        overlay_selected = (
            list(selected_entries[:1]) if len(selected_entries) > 1 else list(selected_entries)
        )
    if st.session_state.get("rb_overlay_entries") != overlay_selected:
        st.session_state["rb_overlay_entries"] = list(overlay_selected)
    return overlay_selected


def _build_common_grid_binned_sweeps(
    test_sweeps: dict[str, pd.DataFrame],
    *,
    signal_col: str,
    bin_hz: float,
) -> dict[str, pd.DataFrame]:
    """Bin raw sweep data directly onto one shared averaging grid."""
    cleaned_frames: dict[str, pd.DataFrame] = {}
    fmin_values: list[float] = []
    fmax_values: list[float] = []

    for name, sweep_df in test_sweeps.items():
        if (
            sweep_df is None
            or sweep_df.empty
            or "Frequency" not in sweep_df.columns
            or signal_col not in sweep_df.columns
        ):
            continue

        working = sweep_df.copy()
        if "IsFrequencyHold" in working.columns:
            working = working.loc[~working["IsFrequencyHold"]].copy()
        if working.empty:
            continue

        working["Frequency"] = pd.to_numeric(working["Frequency"], errors="coerce")
        working[signal_col] = pd.to_numeric(working[signal_col], errors="coerce")
        working = working.loc[
            np.isfinite(working["Frequency"]) & np.isfinite(working[signal_col])
        ].copy()
        if len(working) < 2:
            continue

        freq_values = working["Frequency"].to_numpy(dtype=float)
        fmin = float(np.nanmin(freq_values))
        fmax = float(np.nanmax(freq_values))
        if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
            continue

        cleaned_frames[name] = working
        fmin_values.append(fmin)
        fmax_values.append(fmax)

    if not cleaned_frames:
        return {}

    global_fmin = min(fmin_values)
    global_fmax = max(fmax_values)
    edges = np.arange(global_fmin, global_fmax + float(bin_hz), float(bin_hz))
    if len(edges) < 2:
        edges = np.array([global_fmin, global_fmax], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0

    common_grid: dict[str, pd.DataFrame] = {}
    for name, working in cleaned_frames.items():
        freq_values = working["Frequency"].to_numpy(dtype=float)
        signal_values = working[signal_col].to_numpy(dtype=float)
        bin_ids = np.digitize(freq_values, edges) - 1
        bin_ids = np.clip(bin_ids, 0, len(edges) - 2)
        grouped = (
            pd.DataFrame({"bin": bin_ids, "value": signal_values})
            .groupby("bin")
            .agg(
                mean=("value", "mean"),
                std=("value", "std"),
                count=("value", "count"),
            )
            .reset_index()
        )
        grouped["std"] = grouped["std"].fillna(0)
        grouped["freq_center"] = grouped["bin"].map(
            lambda idx, grid=centers: float(grid[int(idx)]) if int(idx) < len(grid) else float(global_fmax)
        )
        common_grid[name] = grouped.sort_values("freq_center").reset_index(drop=True)

    return common_grid


def _render_pump_sub_groups(
    reg: PumpRegistry,
    pump_name: str,
    pump: Pump,
) -> None:
    """Render sub-group CRUD controls for one pump inside Report Builder."""
    st.markdown("---")
    st.markdown("**Sub-groups** (use these as focused report targets)")

    available_tests = [t.folder for t in pump.tests]
    if pump.sub_groups:
        for group_name, group in list(pump.sub_groups.items()):
            with st.expander(f"{group_name} ({len(group.tests)} tests)", expanded=False):
                new_group_name = st.text_input(
                    "Group name",
                    value=group.name,
                    key=f"rb_sg_name_{pump_name}_{group_name}",
                )
                new_group_desc = st.text_input(
                    "Description",
                    value=group.description,
                    key=f"rb_sg_desc_{pump_name}_{group_name}",
                )
                new_group_tests = st.multiselect(
                    "Tests in this sub-group",
                    available_tests,
                    default=[folder for folder in group.tests if folder in available_tests],
                    key=f"rb_sg_tests_{pump_name}_{group_name}",
                )
                col_save, col_delete = st.columns(2)
                with col_save:
                    if st.button("Save sub-group", key=f"rb_sg_save_{pump_name}_{group_name}"):
                        proposed_name = new_group_name.strip()
                        if not proposed_name:
                            st.error("Sub-group name required.")
                        elif (
                            proposed_name != group_name
                            and proposed_name in pump.sub_groups
                        ):
                            st.error(
                                f"A sub-group named **{proposed_name}** already exists."
                            )
                        else:
                            upsert_sub_group(
                                reg,
                                pump_name,
                                PumpSubGroup(
                                    name=proposed_name,
                                    tests=new_group_tests,
                                    description=new_group_desc.strip(),
                                ),
                                previous_name=group_name,
                            )
                            _persist_registry()
                            st.success(f"Updated sub-group **{proposed_name}**")
                            st.rerun()
                with col_delete:
                    if st.button("Delete sub-group", key=f"rb_sg_delete_{pump_name}_{group_name}"):
                        remove_sub_group(reg, pump_name, group_name)
                        _persist_registry()
                        st.rerun()
    else:
        st.caption("No sub-groups yet for this pump.")

    with st.form(f"rb_new_sg_{pump_name}", clear_on_submit=True):
        new_name = st.text_input(
            "New sub-group name",
            placeholder="e.g. Stable sweeps",
        )
        new_tests = st.multiselect(
            "Tests in sub-group",
            available_tests,
        )
        new_desc = st.text_input("Description (optional)")
        submitted = st.form_submit_button("Create sub-group")

    if submitted:
        proposed_name = new_name.strip()
        if not proposed_name:
            st.error("Sub-group name required.")
        elif proposed_name in pump.sub_groups:
            st.error(f"A sub-group named **{proposed_name}** already exists.")
        else:
            upsert_sub_group(
                reg,
                pump_name,
                PumpSubGroup(
                    name=proposed_name,
                    tests=new_tests,
                    description=new_desc.strip(),
                ),
            )
            _persist_registry()
            st.success(f"Created sub-group **{proposed_name}**")
            st.rerun()


def _build_section_description(*parts: str) -> str:
    """Join non-empty section-description blocks."""
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _build_guided_section_description(plot_id: str, *parts: str) -> str:
    """Join the shared plot explanation with report-specific context."""
    return _build_section_description(build_plot_guide_text(plot_id), *parts)


def _split_report_section_description(description: str) -> tuple[str, str]:
    """Split long report section text into a visible subtitle and detail."""
    paragraphs = [part.strip() for part in str(description or "").split("\n\n") if part.strip()]
    if not paragraphs:
        return "", ""
    priority_prefixes = (
        "What you are seeing:",
        "Tabulated",
        "Summary:",
        "What this table shows:",
    )
    subtitle_index = next(
        (
            idx
            for idx, paragraph in enumerate(paragraphs)
            if paragraph.startswith(priority_prefixes)
        ),
        0,
    )
    subtitle = paragraphs.pop(subtitle_index)
    details = "\n\n".join(paragraphs)
    return subtitle, details


def _format_flat_test_listing(test_names: list[str]) -> str:
    """Render a compact list of included test names."""
    if not test_names:
        return "Included tests: none."
    return "Included tests: " + ", ".join(test_names)


def _format_nested_test_listing(targets: dict[str, dict[str, pd.DataFrame]]) -> str:
    """Render included targets and the tests behind each target."""
    if not targets:
        return "Included targets: none."
    lines = ["Included targets and tests:"]
    for target_name, tests in targets.items():
        test_names = list(tests.keys())
        lines.append(
            f"- {target_name}: {', '.join(test_names) if test_names else 'no tests'}"
        )
    return "\n".join(lines)


def _format_overlay_scope_suffix(
    focused_targets: list[str],
    *,
    total_target_count: int,
) -> str:
    """Build a compact title suffix describing the focused overlay scope."""
    if not focused_targets:
        return ""
    if len(focused_targets) == 1:
        return f" — {focused_targets[0]}"
    if len(focused_targets) == total_target_count:
        return ""
    return f" — {len(focused_targets)} focused targets"


def _prepare_average_overlay(binned_df: pd.DataFrame) -> pd.DataFrame:
    """Adapt a binned sweep DataFrame for the raw all-sweeps overlay."""
    overlay = binned_df.rename(columns={"freq_center": "freq"})
    overlay = overlay[[col for col in ["freq", "mean", "std"] if col in overlay.columns]].copy()
    return overlay


def _downsample_frame_evenly(
    df: pd.DataFrame,
    max_points: int | None,
) -> pd.DataFrame:
    """Reduce large plot payloads while preserving the full x-range."""
    if (
        max_points is None
        or max_points <= 0
        or df.empty
        or len(df) <= int(max_points)
    ):
        return df
    idx = np.linspace(0, len(df) - 1, num=int(max_points), dtype=int)
    return df.iloc[np.unique(idx)].copy()


def _get_cached_test_figure(
    *,
    cache_context: dict[str, object] | None,
    plot_kind: str,
    settings: dict[str, object],
    builder,
):
    """Load a persisted test figure when possible, else build it once."""
    fig, _ = get_or_create_cached_test_figure(
        cache_context=cache_context,
        plot_kind=plot_kind,
        settings=settings,
        builder=builder,
    )
    return fig


def _build_report_warmup_profile(defn: ReportDefinition) -> dict[str, object]:
    """Return the persisted warm-up settings for the current report draft."""
    return {
        "report": {
            "comparisons": list(defn.comparisons),
            "bin_hz": float(defn.bin_hz),
            "max_raw_points": int(defn.max_raw_points),
            "marker_size": int(defn.marker_size),
            "opacity": float(defn.opacity),
            "show_error_bars": bool(defn.show_error_bars),
            "show_raw_all_sweeps": bool(defn.show_raw_all_sweeps),
            "plot_modes": {
                "individual_binned": _resolve_report_plot_mode(defn, "individual_binned"),
                "individual_per_sweep": _resolve_report_plot_mode(defn, "individual_per_sweep"),
                "individual_raw_all_sweeps": _resolve_report_plot_mode(
                    defn,
                    "individual_raw_all_sweeps",
                ),
                "constant_time_series": _resolve_report_plot_mode(
                    defn,
                    "constant_time_series",
                ),
            },
        }
    }


def _render_report_warmup_controls(
    *,
    defn: ReportDefinition,
    selected_entries: list[str],
    selected_folders: list[str],
    data_folder_str: str,
) -> None:
    """Render persistent warm-up controls for the current report selection."""
    warmup_job = load_warmup_job(REPORT_WARMUP_JOB_KEY)
    warmup_summary = summarize_warmup_job(warmup_job)
    worker_state = load_warmup_worker_state(REPORT_WARMUP_JOB_KEY)
    worker_summary = summarize_warmup_worker_state(worker_state)
    default_auto_resume = bool(
        True if warmup_job is None else warmup_job.get("auto_resume", True)
    )

    has_selected_tests = bool(selected_folders)

    with st.sidebar:
        st.divider()
        st.subheader("⚡ Overnight Cache")
        st.caption(
            "Build the expensive per-test report plots once, store them on disk, "
            "and reuse them across app restarts."
        )
        st.caption(
            "Use the overnight option to queue the current selection and let a "
            "detached worker keep filling the cache even after you close the browser."
        )
        if not has_selected_tests:
            st.caption(
                "Select at least one report target with available tests to enable "
                "new overnight warm-up jobs."
            )
        auto_resume = st.checkbox(
            "Resume unfinished cache warm-up automatically when this page opens",
            value=default_auto_resume,
            key="rb_warmup_auto_resume",
        )
        if warmup_job and warmup_job.get("auto_resume", True) != bool(auto_resume):
            warmup_job["auto_resume"] = bool(auto_resume)
            save_warmup_job(warmup_job, job_key=REPORT_WARMUP_JOB_KEY)
            warmup_summary = summarize_warmup_job(warmup_job)

        start_label = (
            "Queue selected tests for warm-up"
            if not warmup_job
            else "Replace queued warm-up with current selection"
        )
        start_bg_label = (
            "Start overnight warm-up for selected tests"
            if not warmup_job
            else "Replace and start overnight warm-up"
        )
        if worker_summary.get("running"):
            st.info(
                "A background warm-up worker is already running. Pause it before "
                "replacing the queued selection."
            )

        c_start_bg, c_start_queue = st.columns(2)
        with c_start_bg:
            if st.button(
                start_bg_label,
                key="rb_warmup_start_bg",
                disabled=worker_summary.get("running", False) or not has_selected_tests,
                use_container_width=True,
            ):
                start_report_warmup_job(
                    label=f"Report cache warm-up ({len(selected_folders)} tests)",
                    entry_ids=list(selected_entries),
                    test_names=list(selected_folders),
                    selection_mode=str(defn.selection_mode),
                    profile=_build_report_warmup_profile(defn),
                    data_folder_path=data_folder_str,
                    auto_resume=bool(auto_resume),
                    job_key=REPORT_WARMUP_JOB_KEY,
                )
                try:
                    launch_warmup_worker(REPORT_WARMUP_JOB_KEY)
                except Exception as exc:
                    st.error(f"Could not start the overnight worker: {exc}")
                else:
                    st.rerun()
        with c_start_queue:
            if st.button(
                start_label,
                key="rb_warmup_start",
                disabled=worker_summary.get("running", False) or not has_selected_tests,
                use_container_width=True,
            ):
                start_report_warmup_job(
                    label=f"Report cache warm-up ({len(selected_folders)} tests)",
                    entry_ids=list(selected_entries),
                    test_names=list(selected_folders),
                    selection_mode=str(defn.selection_mode),
                    profile=_build_report_warmup_profile(defn),
                    data_folder_path=data_folder_str,
                    auto_resume=bool(auto_resume),
                    job_key=REPORT_WARMUP_JOB_KEY,
                )
                st.rerun()

        if worker_state:
            status_label = str(worker_summary.get("status", "idle")).replace("_", " ").title()
            if worker_summary.get("running"):
                st.caption(
                    f"Background worker: {status_label} · PID {worker_summary.get('pid')} · "
                    f"last heartbeat {worker_summary.get('last_heartbeat') or 'just now'}"
                )
            elif worker_summary.get("stale"):
                st.warning(
                    "The overnight worker stopped unexpectedly. Relaunch it and it will "
                    "continue from the tasks already completed on disk."
                )
            elif worker_summary.get("status") not in {"idle", ""}:
                detail = worker_summary.get("finished_at") or worker_summary.get("last_heartbeat") or "unknown time"
                st.caption(f"Background worker: {status_label} · last update {detail}")

            if worker_summary.get("error"):
                st.warning(worker_summary["error"])
            if worker_summary.get("log_path"):
                st.caption(f"Worker log: `{worker_summary['log_path']}`")

        if warmup_job:
            selection = warmup_job.get("selection", {})
            test_names = list(selection.get("test_names", []))
            st.caption(
                f"Current job: {len(test_names)} test(s), "
                f"{warmup_summary['completed']}/{warmup_summary['total']} task(s) done."
            )
            progress = (
                float(warmup_summary["completed"]) / float(warmup_summary["total"])
                if warmup_summary["total"]
                else 0.0
            )
            st.progress(
                progress,
                text=(
                    "Paused"
                    if warmup_summary.get("paused")
                    else (
                        "Completed"
                        if warmup_summary.get("complete")
                        else "Running"
                    )
                ),
            )

            if warmup_summary.get("paused") and worker_summary.get("running"):
                st.caption("Pause requested. The background worker will stop after the current task finishes.")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button(
                    (
                        "Resume in background"
                        if warmup_summary.get("paused")
                        else "Run current job in background"
                    ),
                    key="rb_warmup_run_bg",
                    disabled=(
                        worker_summary.get("running", False)
                        or warmup_summary["pending"] == 0
                    ),
                ):
                    if warmup_summary.get("paused"):
                        set_warmup_job_paused(False, REPORT_WARMUP_JOB_KEY)
                    try:
                        launch_warmup_worker(REPORT_WARMUP_JOB_KEY)
                    except Exception as exc:
                        st.error(f"Could not start the overnight worker: {exc}")
                    else:
                        st.rerun()
            with c2:
                if warmup_summary.get("paused"):
                    if st.button("Resume", key="rb_warmup_resume"):
                        set_warmup_job_paused(False, REPORT_WARMUP_JOB_KEY)
                        st.rerun()
                else:
                    if st.button(
                        "Pause",
                        key="rb_warmup_pause",
                        disabled=not warmup_summary.get("active", False),
                    ):
                        set_warmup_job_paused(True, REPORT_WARMUP_JOB_KEY)
                        st.rerun()
            with c3:
                if st.button(
                    "Retry failed",
                    key="rb_warmup_retry",
                    disabled=warmup_summary["errors"] == 0,
                ):
                    retry_failed_warmup_tasks(REPORT_WARMUP_JOB_KEY)
                    st.rerun()
            with c4:
                if st.button("Clear job", key="rb_warmup_clear"):
                    delete_warmup_job(REPORT_WARMUP_JOB_KEY)
                    st.rerun()

            if warmup_summary["errors"]:
                with st.expander(
                    f"{warmup_summary['errors']} failed task(s)",
                    expanded=False,
                ):
                    for task in warmup_job.get("tasks", []):
                        if task.get("status") == "error":
                            st.warning(
                                f"{task.get('label', 'Task')}: {task.get('error', 'Unknown error')}"
                            )


def _time_axis_seconds(series: pd.Series) -> tuple[pd.Series, str]:
    """Convert a time-like axis to elapsed seconds for correlation analysis."""
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        return (parsed - parsed.min()).dt.total_seconds(), "elapsed seconds"

    numeric = pd.to_numeric(series, errors="coerce")
    return numeric, str(series.name or "time")


def _build_time_effect_summary(df: pd.DataFrame, signal_col: str) -> str:
    """Summarize the strength of any time-effect in a constant test."""
    analysis = _analyze_time_effect(df, signal_col)
    return analysis["summary"]


def _parse_optional_float(
    raw_value: object,
    *,
    label: str,
    errors: list[str],
) -> float | None:
    """Parse an optional float from a text input."""
    text = str(raw_value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        errors.append(f"{label} must be a number or blank.")
        return None


def _read_axis_bounds_from_state(
    prefix: str,
    *,
    label: str,
    errors: list[str],
) -> AxisBounds:
    """Build one axis-bound set from Streamlit text-input state."""
    axis = AxisBounds(
        x_min=_parse_optional_float(
            st.session_state.get(f"{prefix}_xmin", ""),
            label=f"{label} x-min",
            errors=errors,
        ),
        x_max=_parse_optional_float(
            st.session_state.get(f"{prefix}_xmax", ""),
            label=f"{label} x-max",
            errors=errors,
        ),
        y_min=_parse_optional_float(
            st.session_state.get(f"{prefix}_ymin", ""),
            label=f"{label} y-min",
            errors=errors,
        ),
        y_max=_parse_optional_float(
            st.session_state.get(f"{prefix}_ymax", ""),
            label=f"{label} y-max",
            errors=errors,
        ),
    )
    if (
        axis.x_min is not None
        and axis.x_max is not None
        and axis.x_min >= axis.x_max
    ):
        errors.append(f"{label} x-min must be smaller than x-max.")
    if (
        axis.y_min is not None
        and axis.y_max is not None
        and axis.y_min >= axis.y_max
    ):
        errors.append(f"{label} y-min must be smaller than y-max.")
    return axis


def _render_axis_inputs(
    title: str,
    *,
    prefix: str,
    x_label: str,
    y_label: str,
) -> None:
    """Render four optional manual axis-limit inputs."""
    st.markdown(f"**{title}**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.text_input(
            f"{x_label} min",
            key=f"{prefix}_xmin",
            placeholder="auto",
        )
    with c2:
        st.text_input(
            f"{x_label} max",
            key=f"{prefix}_xmax",
            placeholder="auto",
        )
    with c3:
        st.text_input(
            f"{y_label} min",
            key=f"{prefix}_ymin",
            placeholder="auto",
        )
    with c4:
        st.text_input(
            f"{y_label} max",
            key=f"{prefix}_ymax",
            placeholder="auto",
        )


def _axis_bounds_text(
    axis_bounds: AxisBounds,
    *,
    x_name: str = "x",
    y_name: str = "y",
) -> str:
    """Summarize manual axis bounds for section descriptions."""
    if (
        axis_bounds.x_min is None
        and axis_bounds.x_max is None
        and axis_bounds.y_min is None
        and axis_bounds.y_max is None
    ):
        return "axis limits = auto"

    parts: list[str] = []
    if axis_bounds.x_min is not None or axis_bounds.x_max is not None:
        parts.append(
            f"{x_name} range = "
            f"{_format_optional_numeric(axis_bounds.x_min) or 'auto'} to "
            f"{_format_optional_numeric(axis_bounds.x_max) or 'auto'}"
        )
    if axis_bounds.y_min is not None or axis_bounds.y_max is not None:
        parts.append(
            f"{y_name} range = "
            f"{_format_optional_numeric(axis_bounds.y_min) or 'auto'} to "
            f"{_format_optional_numeric(axis_bounds.y_max) or 'auto'}"
        )
    return "; ".join(parts)


def _apply_axis_bounds(fig, axis_bounds: AxisBounds) -> None:
    """Apply optional manual axis limits to a Plotly figure in-place."""
    if axis_bounds.x_min is not None or axis_bounds.x_max is not None:
        fig.update_xaxes(range=[axis_bounds.x_min, axis_bounds.x_max])
    if axis_bounds.y_min is not None or axis_bounds.y_max is not None:
        fig.update_yaxes(range=[axis_bounds.y_min, axis_bounds.y_max])


def _compact_trace_label(label: object, *, max_len: int = 56) -> str:
    """Shorten long legend labels without losing the distinguishing suffix."""
    text = str(label or "").strip()
    if len(text) <= max_len:
        return text
    prefix = max(18, int(max_len * 0.55))
    suffix = max(14, max_len - prefix - 5)
    return f"{text[:prefix].rstrip()} ... {text[-suffix:].lstrip()}"


def _scatter_trace_indices(fig) -> list[int]:
    """Return the trace indices eligible for line/marker style toggles."""
    indices: list[int] = []
    for idx, trace in enumerate(fig.data):
        if getattr(trace, "type", None) not in {"scatter", "scattergl"}:
            continue
        mode = getattr(trace, "mode", None)
        if mode is None:
            continue
        indices.append(idx)
    return indices


def _supports_report_style_toggle(fig) -> bool:
    """Return whether an exported report figure should expose mode controls."""
    x_title = str(getattr(getattr(fig.layout.xaxis, "title", None), "text", "") or "")
    return bool(x_title) and any(
        token in x_title.lower() for token in ("frequency", "time")
    )


def _add_report_style_toggle(fig) -> None:
    """Embed a Plotly dropdown so exported report users can change curve style."""
    trace_indices = _scatter_trace_indices(fig)
    if not trace_indices or not _supports_report_style_toggle(fig):
        return

    first_mode = str(getattr(fig.data[trace_indices[0]], "mode", "") or "")
    active = _PLOT_MODE_OPTIONS.index(first_mode) if first_mode in _PLOT_MODE_OPTIONS else 0
    existing = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    existing.append(
        {
            "type": "dropdown",
            "direction": "down",
            "x": 1.0,
            "y": 1.18,
            "xanchor": "right",
            "yanchor": "top",
            "showactive": True,
            "active": active,
            "buttons": [
                {
                    "label": {
                        "lines+markers": "Lines + markers",
                        "lines": "Lines",
                        "markers": "Markers",
                    }[mode],
                    "method": "restyle",
                    "args": [{"mode": mode}, trace_indices],
                }
                for mode in _PLOT_MODE_OPTIONS
            ],
            "bgcolor": "rgba(255,255,255,0.94)",
            "bordercolor": "#cfd4da",
            "font": {"size": 11},
            "pad": {"l": 4, "r": 4, "t": 0, "b": 0},
        }
    )
    fig.update_layout(updatemenus=existing)


def _apply_report_plot_layout(fig) -> None:
    """Normalize report-plot spacing so axes and legends stay readable."""
    legend_names = [
        str(getattr(trace, "name", "") or "").strip()
        for trace in fig.data
        if getattr(trace, "showlegend", True) is not False
    ]
    multi_legend = len([name for name in legend_names if name]) > 1
    long_legend = any(len(name) > 48 for name in legend_names if name)

    for trace in fig.data:
        name = str(getattr(trace, "name", "") or "").strip()
        if name and len(name) > 48:
            trace.name = _compact_trace_label(name)

    bottom_margin = 96 if multi_legend else 72
    if multi_legend:
        bottom_margin = 124 if long_legend else 108

    fig.update_layout(
        margin=dict(l=92, r=36, t=92, b=bottom_margin),
        title=dict(x=0.5, xanchor="center"),
        hoverlabel=dict(namelength=-1),
    )
    fig.update_xaxes(automargin=True, title_standoff=18)
    fig.update_yaxes(automargin=True, title_standoff=18)

    if multi_legend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                x=0.0,
                xanchor="left",
                y=-0.22,
                yanchor="top",
                font=dict(size=10),
                itemclick="toggle",
                itemdoubleclick="toggleothers",
            )
        )

    _add_report_style_toggle(fig)


def _format_render_context(
    defn: ReportDefinition,
    *,
    plot_mode: str | None = None,
    axis_bounds: AxisBounds | None = None,
    x_name: str = "x",
    y_name: str = "y",
    include_opacity: bool = True,
) -> str:
    """Summarize shared rendering settings for report figures."""
    effective_mode = _normalize_report_plot_mode(plot_mode or defn.plot_mode)
    mode_label = {
        "lines": "lines",
        "markers": "markers",
        "lines+markers": "lines + markers",
    }.get(effective_mode, effective_mode)
    parts = [
        f"plot mode = {mode_label}",
        f"marker size = {int(defn.marker_size)} px",
    ]
    if include_opacity:
        parts.append(f"opacity = {float(defn.opacity):.2f}")
    if axis_bounds is not None:
        parts.append(_axis_bounds_text(axis_bounds, x_name=x_name, y_name=y_name))
    return "Render settings: " + "; ".join(parts) + "."


def _format_sweep_plot_context(
    defn: ReportDefinition,
    *,
    use_average_bin: bool = False,
    include_test_bin: bool = True,
    include_individual_tests: bool | None = None,
    raw_layer_enabled: bool | None = None,
    axis_bounds: AxisBounds | None = None,
) -> str:
    """Summarize the key report settings that govern a sweep-based figure."""
    parts: list[str] = []
    if include_test_bin:
        parts.append(f"per-test frequency bin width = {defn.bin_hz:g} Hz")
    if use_average_bin:
        parts.append(
            f"cross-test / cross-target averaging bin width = {defn.avg_bin_hz:g} Hz"
        )
    parts.append(
        "±1 standard deviation error bars shown"
        if defn.show_error_bars
        else "±1 standard deviation error bars hidden"
    )
    if include_individual_tests is not None:
        parts.append(
            "individual test traces shown behind target averages"
            if include_individual_tests
            else "individual test traces hidden behind target averages"
        )
    if raw_layer_enabled is not None:
        parts.append(
            "raw all-sweeps layer included"
            if raw_layer_enabled
            else "raw all-sweeps layer omitted"
        )
    if axis_bounds is not None:
        parts.append(
            _axis_bounds_text(
                axis_bounds,
                x_name="frequency",
                y_name="flow",
            )
        )
    return "Plot context: " + "; ".join(parts) + "."


def _format_raw_sweep_context(
    defn: ReportDefinition,
    *,
    axis_bounds: AxisBounds,
    max_raw_points: int | None = None,
    average_overlay_bin_hz: float | None = None,
) -> str:
    """Describe raw sweep-point render settings accurately."""
    parts = ["raw sweep points shown without frequency binning"]
    if max_raw_points is not None:
        parts.append(f"per-test point cap = {int(max_raw_points):,}")
    if average_overlay_bin_hz is not None:
        parts.append(
            f"black average overlay uses {float(average_overlay_bin_hz):g} Hz bins"
        )
    parts.append(
        _axis_bounds_text(axis_bounds, x_name="frequency", y_name="flow")
    )
    return "Plot context: " + "; ".join(parts) + "."


def _format_variability_context(
    defn: ReportDefinition,
    *,
    axis_bounds: AxisBounds,
    include_thresholds: bool = False,
) -> str:
    """Describe mean-vs-std plot settings."""
    parts = []
    if include_thresholds:
        parts.append(
            f"high-flow filter keeps the top {int(defn.mean_threshold_pct)}% by mean"
        )
        parts.append(
            f"stability filter keeps the lowest {int(defn.std_threshold_pct)}% by std within that subset"
        )
    parts.append(
        _axis_bounds_text(axis_bounds, x_name="mean flow", y_name="std")
    )
    return "Plot context: " + "; ".join(parts) + "."


def _build_sweep_start_alignment_summary(
    sweep_df: pd.DataFrame,
    *,
    bin_hz: float,
) -> str:
    """Explain whether differing sweep start frequencies likely drive the pattern."""
    if (
        sweep_df.empty
        or "Sweep" not in sweep_df.columns
        or "Frequency" not in sweep_df.columns
    ):
        return "Sweep-start alignment check: unavailable because sweep indices or frequency values are missing."

    start_freqs = (
        sweep_df.groupby("Sweep", sort=True)["Frequency"]
        .first()
        .dropna()
        .astype(float)
    )
    if len(start_freqs) < 2:
        return "Sweep-start alignment check: only one sweep is available, so start-to-start variation cannot be compared."

    start_span = float(start_freqs.max() - start_freqs.min())
    threshold = max(bin_hz * 2.0, 5.0)
    if start_span <= threshold:
        conclusion = (
            "The sweep-start spread is small relative to the report binning, so a comb-like look is more likely caused by discrete setpoints, "
            "up/down hysteresis, or cycle-to-cycle differences than by a small offset in start frequency."
        )
    else:
        conclusion = (
            "The sweep-start spread is large enough that start-frequency offsets may contribute to the visual spread, "
            "although discrete setpoints and hysteresis can still be important contributors."
        )

    return (
        f"Sweep-start alignment check: the first recorded frequency across sweeps spans {start_span:.2f} Hz "
        f"(bin width {bin_hz:g} Hz). {conclusion}"
    )


def _analyze_time_effect(df: pd.DataFrame, signal_col: str) -> dict[str, str | float | bool]:
    """Return numeric and textual diagnostics for constant-test time effects."""
    if df.empty or signal_col not in df.columns:
        return {
            "available": False,
            "summary": "Time-effect analysis: unavailable because the signal column is missing.",
            "correlation": float("nan"),
            "slope": float("nan"),
            "strength": "unavailable",
            "interpretation": "Signal column missing.",
            "x_label": "time",
        }

    x_col = df.columns[0]
    x_values, x_label = _time_axis_seconds(df[x_col])
    y_values = pd.to_numeric(df[signal_col], errors="coerce")
    mask = x_values.notna() & y_values.notna()
    if mask.sum() < 3:
        return {
            "available": False,
            "summary": "Time-effect analysis: not enough valid points for a correlation estimate.",
            "correlation": float("nan"),
            "slope": float("nan"),
            "strength": "unavailable",
            "interpretation": "Not enough valid points.",
            "x_label": x_label,
        }

    x = x_values[mask].astype(float).to_numpy()
    y = y_values[mask].astype(float).to_numpy()
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else 0.0
    abs_corr = abs(corr)

    if abs_corr >= 0.7:
        strength = "strong"
    elif abs_corr >= 0.4:
        strength = "moderate"
    elif abs_corr >= 0.2:
        strength = "weak"
    else:
        strength = "very weak"

    if abs_corr < 0.2:
        interpretation = (
            "There is no clear linear time-effect in this test; any drift is small "
            "relative to the point-to-point variation."
        )
    elif corr > 0:
        interpretation = (
            "Flow tends to increase over time, which suggests a positive drift "
            "or settling trend during the constant-frequency hold."
        )
    else:
        interpretation = (
            "Flow tends to decrease over time, which suggests a negative drift "
            "or decay during the constant-frequency hold."
        )

    summary = (
        f"Time-effect analysis: Pearson correlation between {x_label} and flow = "
        f"{corr:.3f} ({strength}). Estimated slope = {slope:.4f} µL/min per {x_label}. "
        f"{interpretation}"
    )
    return {
        "available": True,
        "summary": summary,
        "correlation": corr,
        "slope": slope,
        "strength": strength,
        "interpretation": interpretation,
        "x_label": x_label,
    }


def _build_time_effect_table(
    const_frames: dict[str, pd.DataFrame],
    signal_col: str,
) -> pd.DataFrame:
    """Build a compact summary table for constant-test time-effect diagnostics."""
    rows: list[dict[str, str | float]] = []
    for name, df in const_frames.items():
        analysis = _analyze_time_effect(df, signal_col)
        rows.append({
            "Test": name,
            "Correlation": analysis["correlation"],
            "Slope": analysis["slope"],
            "Strength": analysis["strength"],
            "Interpretation": analysis["interpretation"],
            "Time Axis": analysis["x_label"],
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Entry point for the Report Builder page."""
    try:
        st.title("📑 Report Builder")

        # ── Sidebar — Data Source (shared widget) ──────────────
        data_folder_str, run_dirs, run_names = resolve_data_path(
            key_suffix="rb",
            render_widget=False,
        )

        # ── Top-level tabs ──────────────────────────────────────
        tab_registry, tab_build, tab_audit = st.tabs([
            "📋 Pump Registry",
            "📝 Build Report",
            "📜 Audit Log",
        ])

        with tab_registry:
            _render_registry_tab(run_names, run_dirs)

        with tab_build:
            _render_build_tab(data_folder_str, run_names, run_dirs)

        with tab_audit:
            _render_audit_tab()

    except Exception as e:
        st.error(f"❌ **CRITICAL ERROR:** {e}")
        with st.expander("🔍 Debug"):
            st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════
# TAB 1 — REGISTRY
# ════════════════════════════════════════════════════════════════════════

def _render_registry_tab(run_names: list[str], run_dirs: list) -> None:
    """Registry management: view, add, edit, delete entries + import."""
    reg = _get_registry()

    st.subheader("📋 Pump Registry")
    st.caption(
        "Map physical pumps to their test folders.  "
        "Import from your logbook spreadsheet or add entries manually."
    )

    # ── Import from spreadsheet ─────────────────────────────────
    with st.expander("📥 Import from spreadsheet", expanded=not bool(reg.pumps)):
        st.markdown(
            "Upload your test logbook (Excel `.xlsx` or `.csv`).  "
            "The importer looks for columns named **Pump ID** and "
            "**Test ID**, plus optional columns for date, author, "
            "test type, voltage, success/fail, and notes."
        )
        uploaded = st.file_uploader(
            "Choose file",
            type=["xlsx", "xls", "csv"],
            key="rb_upload",
        )
        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    import_df = pd.read_csv(uploaded)
                else:
                    import_df = pd.read_excel(uploaded)

                st.markdown("**Preview (first 10 rows):**")
                st.dataframe(import_df.head(10), use_container_width=True)

                if st.button("🚀 Import", key="rb_do_import"):
                    reg, n_entries, n_tests = import_from_dataframe(reg, import_df)
                    _persist_registry()
                    st.success(
                        f"✅ Imported **{n_entries}** new entries, "
                        f"**{n_tests}** test links."
                    )
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Import error: {e}")

    # ── Add entry manually ──────────────────────────────────────
    with st.expander("➕ Add new pump manually"):
        with st.form("rb_add_entry", clear_on_submit=True):
            new_id = st.text_input(
                "Pump name (e.g. 'Pump 270226-2')"
            )
            new_notes = st.text_area("Notes (optional)")
            if st.form_submit_button("Add Pump"):
                if new_id.strip():
                    pump = Pump(
                        name=new_id.strip(),
                        notes=new_notes.strip(),
                    )
                    add_pump(reg, pump)
                    _persist_registry()
                    st.success(f"✅ Added **{new_id.strip()}**")
                    st.rerun()
                else:
                    st.error("Pump name is required.")

    # ── Display existing entries ────────────────────────────────
    st.divider()
    if not reg.pumps:
        st.info(
            "📭 Registry is empty.  Import a spreadsheet or add pumps "
            "manually above."
        )
        return

    st.markdown(f"**{len(reg.pumps)} pumps** in registry:")

    for pid, pump in sorted(reg.pumps.items()):
        with st.expander(
            f"{'🔵' if pump.tests else '⚪'} {pump.name}  "
            f"({len(pump.tests)} tests)",
            expanded=False,
        ):
            # ── Edit name ───────────────────────────────────────
            col_name, col_del = st.columns([4, 1])
            with col_name:
                new_name = st.text_input(
                    "Name",
                    value=pump.name,
                    key=f"rb_name_{pid}",
                )
                if new_name != pump.name:
                    if st.button("Save name", key=f"rb_savename_{pid}"):
                        rename_pump(reg, pid, new_name)
                        _persist_registry()
                        st.rerun()
            with col_del:
                st.markdown("&nbsp;")  # spacing
                if st.button("🗑️ Delete pump", key=f"rb_del_{pid}"):
                    remove_pump(reg, pid)
                    _persist_registry()
                    st.rerun()

            # ── Notes ───────────────────────────────────────────
            notes_val = st.text_area(
                "Notes",
                value=pump.notes,
                key=f"rb_notes_{pid}",
            )
            if notes_val != pump.notes:
                if st.button("Save notes", key=f"rb_savenotes_{pid}"):
                    pump.notes = notes_val
                    _persist_registry()
                    st.rerun()

            # ── Linked tests ────────────────────────────────────
            st.markdown("**Linked tests:**")
            if pump.tests:
                for i, tl in enumerate(pump.tests):
                    cols = st.columns([4, 1, 1])
                    with cols[0]:
                        badge = ""
                        if tl.test_type == "sweep":
                            badge = " 🔵 sweep"
                        elif tl.test_type == "constant":
                            badge = " 🟡 constant"
                        if tl.success is True:
                            badge += " ✅"
                        elif tl.success is False:
                            badge += " ❌"
                        avail = " ✓" if tl.folder in run_names else " ⚠️ not found"
                        st.markdown(
                            f"`{tl.folder}`{badge}{avail}"
                        )
                        if tl.description:
                            st.caption(tl.description)
                    with cols[2]:
                        if st.button(
                            "Unlink", key=f"rb_unlink_{pid}_{i}",
                        ):
                            unlink_test(reg, pid, tl.folder)
                            _persist_registry()
                            st.rerun()
            else:
                st.caption("No tests linked yet.")

            # ── Link a test ─────────────────────────────────────
            if run_names:
                already_linked = {t.folder for t in pump.tests}
                available = [n for n in run_names if n not in already_linked]
                if available:
                    with st.form(f"rb_link_{pid}", clear_on_submit=True):
                        sel_folder = st.selectbox(
                            "Link a test folder",
                            available,
                            key=f"rb_linksel_{pid}",
                        )
                        sel_type = st.selectbox(
                            "Test type",
                            ["", "sweep", "constant"],
                            key=f"rb_linktype_{pid}",
                        )
                        sel_desc = st.text_input(
                            "Description (optional)",
                            key=f"rb_linkdesc_{pid}",
                        )
                        if st.form_submit_button("Link test"):
                            link_test(
                                reg, pid,
                                TestLink(
                                    folder=sel_folder,
                                    test_type=sel_type,
                                    description=sel_desc,
                                ),
                            )
                            _persist_registry()
                            st.success(f"Linked `{sel_folder}` to {pump.name}")
                            st.rerun()

            _render_pump_sub_groups(reg, pid, pump)


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — BUILD REPORT
# ════════════════════════════════════════════════════════════════════════

def _render_build_tab(data_folder_str: str, run_names: list[str], run_dirs: list) -> None:
    """Report composition: select entries, choose charts, preview, export."""
    reg = _get_registry()

    if not reg.pumps:
        st.info(
            "📭 Registry is empty.  Go to the **Registry** tab to add "
            "pumps first."
        )
        return

    # ── Load saved report (optional) ────────────────────────────
    saved_reports = load_saved_reports()
    col_load, col_del_saved = st.columns([3, 1])
    with col_load:
        saved_names = ["(new report)"] + list(saved_reports.keys())
        load_choice = st.selectbox(
            "Load a saved report template",
            saved_names,
            key="rb_load_saved",
        )
    with col_del_saved:
        st.markdown("&nbsp;")
        if load_choice != "(new report)" and st.button(
            "Delete template",
            key="rb_del_saved",
        ):
            delete_saved_report(load_choice)
            st.rerun()

    loaded_defn: ReportDefinition | None = None
    if load_choice != "(new report)":
        loaded_defn = load_report_definition(load_choice)
    load_marker = load_choice if loaded_defn else "(new report)"
    _apply_template_to_session_state(load_marker, loaded_defn)

    st.divider()

    # ── Report metadata ─────────────────────────────────────────
    if load_choice != "(new report)":
        saved_at = saved_reports.get(load_choice, {}).get("saved_at", "")
        if saved_at:
            st.caption(f"Loaded template `{load_choice}` · saved {saved_at}")

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input(
            "Report title",
            key="rb_title",
        )
    with col2:
        author = st.text_input(
            "Author",
            key="rb_author",
        )
    notes = st.text_area(
        "Report notes / description",
        key="rb_notes",
        height=80,
    )

    # ── Select report targets ───────────────────────────────────
    st.subheader("📦 Select Report Targets")
    selection_mode = st.radio(
        "Build the report from",
        options=list(_SELECTION_MODE_OPTIONS.keys()),
        format_func=lambda key: _SELECTION_MODE_OPTIONS[key],
        key="rb_selection_mode",
        horizontal=True,
    )

    target_options = _build_report_target_options(reg, run_names, selection_mode)
    if selection_mode == "sub_groups" and not target_options:
        st.info(
            "No saved pump sub-groups exist yet. Create them in the "
            "**Pump Registry** tab above or on **Manage Groups**."
        )
        return

    entries_widget_key = (
        "rb_entries_sub_groups"
        if selection_mode == "sub_groups"
        else "rb_entries_pumps"
    )
    current_selected = [
        entry_id
        for entry_id in st.session_state.get(entries_widget_key, [])
        if entry_id in target_options
    ]
    if st.session_state.get(entries_widget_key) != current_selected:
        st.session_state[entries_widget_key] = current_selected

    missing_saved_entries = []
    if loaded_defn:
        missing_saved_entries = [
            entry_id for entry_id in loaded_defn.entry_ids if entry_id not in target_options
        ]
        if missing_saved_entries:
            st.warning(
                f"{len(missing_saved_entries)} saved target(s) from the template are not "
                "available in the current registry or data folder and were skipped."
            )

    selected_entries = st.multiselect(
        (
            "Choose saved sub-groups to include"
            if selection_mode == "sub_groups"
            else "Choose pumps to include"
        ),
        list(target_options.keys()),
        format_func=lambda entry_id: target_options.get(entry_id, entry_id),
        key=entries_widget_key,
    )
    selected_entries = _normalize_report_selected_entries(
        selected_entries,
        st.session_state.get(entries_widget_key),
        target_options,
    )
    if st.session_state.get(entries_widget_key) != selected_entries:
        st.session_state[entries_widget_key] = list(selected_entries)

    # Quick select buttons
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Select all", key="rb_sel_all"):
            st.session_state[entries_widget_key] = list(target_options.keys())
            st.rerun()
    with c2:
        if st.button("Clear", key="rb_sel_clr"):
            st.session_state[entries_widget_key] = []
            st.rerun()

    if not selected_entries:
        st.info("👆 Select at least one report target to continue.")
        return

    selected_folders = _collect_report_target_folders(selected_entries, reg, run_names)
    bin_signature = (
        "report_builder",
        tuple(sorted(selected_entries)),
        tuple(sorted(selected_folders)),
    )
    if st.session_state.get("_rb_bin_signature") != bin_signature:
        st.session_state["_rb_bin_signature"] = bin_signature
        st.session_state.pop("_rb_bin_recommendation", None)

    st.caption(
        f"Selected {len(selected_entries)} report target(s) with "
        f"{sum(1 for entry_id in selected_entries if entry_id in target_options)} currently available."
        )

    # ── Comparison types ────────────────────────────────────────
    left_col, right_col = st.columns([3.2, 1.8])
    with left_col:
        st.subheader("📊 Comparisons")
        selected_comparisons = st.multiselect(
            "Choose what to include in the report",
            list(COMPARISON_OPTIONS.keys()),
            format_func=lambda k: COMPARISON_OPTIONS[k],
            key="rb_comparisons",
        )
    with right_col:
        render_plot_guide(
            selected_comparisons or list(COMPARISON_OPTIONS.keys()),
            key_prefix="rb_comparisons",
            label_lookup=COMPARISON_OPTIONS,
        )
    if selected_comparisons:
        st.caption(
            "Included comparison blocks: "
            + ", ".join(COMPARISON_OPTIONS[c] for c in selected_comparisons)
        )

    overlay_selected_entries = _normalize_overlay_selected_entries(
        selected_entries,
        target_options,
    )

    # ── Plot settings ───────────────────────────────────────────
    with st.expander("⚙️ Plot settings", expanded=False):
        st.caption(
            "Recommended bin widths are available for any report. They are "
            "calculated on demand here and are also reused automatically when "
            "you generate average-style plots."
        )
        if st.button(
            "Calculate recommended bin widths",
            key="rb_calc_bin_recommendation",
            disabled=not selected_folders,
        ):
            recommendation = _ensure_report_bin_recommendation(
                selected_folders=selected_folders,
                run_names=run_names,
                run_dirs=run_dirs,
            )
            if recommendation:
                st.session_state["_rb_bin_recommendation"] = recommendation
            else:
                st.session_state.pop("_rb_bin_recommendation", None)
            st.rerun()

        plot_bin_hz = st.slider(
            "Per-test frequency bin width (Hz)",
            0.5,
            100.0,
            value=float(st.session_state.get("rb_plot_bin", 5.0)),
            step=0.5,
            key="rb_plot_bin",
            help=(
                "Used for per-test sweep plots such as the all-tests overlay, "
                "individual sweep means, and raw-sweep average overlays."
            ),
        )
        avg_bin_hz = st.slider(
            "Cross-test averaging bin width (Hz)",
            0.5,
            100.0,
            value=float(st.session_state.get("rb_avg_bin", 3.0)),
            step=0.5,
            key="rb_avg_bin",
            help=(
                "Used for the target-comparison and global-average plots. These "
                "plots now bin raw sweep data directly on this grid."
            ),
        )
        auto_use_recommended_avg_bin = st.checkbox(
            "Automatically use the recommended averaging bin for average-style plots",
            key="rb_auto_avg_bin",
            help=(
                "Applies to the report-target comparison, relative target "
                "comparison, and global average. The per-test plot bin above is "
                "not changed by this setting."
            ),
        )
        show_err = st.checkbox("Show ±1 std error bars", key="rb_err")
        show_indiv = st.checkbox(
            "Show individual test lines behind averages",
            key="rb_indiv",
        )
        show_raw_all_sweeps = st.checkbox(
            "Include raw all-sweeps layer for individual sweep diagnostics",
            key="rb_raw_all_sweeps",
        )
        active_plot_mode_keys = _active_report_plot_mode_keys(
            selected_comparisons,
            include_raw_all_sweeps=show_raw_all_sweeps,
        )
        if "sweep_overlay" in selected_comparisons:
            overlay_target_options = {
                entry_id: target_options[entry_id]
                for entry_id in selected_entries
                if entry_id in target_options
            }
            st.markdown("**Focused all-tests overlay**")
            st.caption(
                "This controls the crowded multi-test sweep overlay. Pick the "
                "report targets whose individual tests you actually want to see "
                "together. Choose one target if you want a readable single-pump "
                "view, and turn on the drill-down option below to also get all "
                "sweeps from those tests in one figure."
            )
            overlay_selected_entries = st.multiselect(
                "Targets shown in 'Frequency Sweep — All Tests Overlay'",
                list(overlay_target_options.keys()),
                default=overlay_selected_entries,
                format_func=lambda entry_id: overlay_target_options.get(entry_id, entry_id),
                key="rb_overlay_entries",
            )
            overlay_selected_entries = _normalize_report_selected_entries(
                overlay_selected_entries,
                st.session_state.get("rb_overlay_entries", []),
                overlay_target_options,
            )
            if not overlay_selected_entries and selected_entries:
                overlay_selected_entries = list(selected_entries[:1])
                st.session_state["rb_overlay_entries"] = list(overlay_selected_entries)
            st.checkbox(
                "Add per-target all-sweeps drill-down for the focused overlay targets",
                key="rb_overlay_sweep_drilldown",
                help=(
                    "Adds an extra section per focused target where every sweep of "
                    "every included test is shown together. Use this when you need "
                    "to inspect sweep-to-sweep repeatability across a single pump "
                    "or subgroup."
                ),
            )
        if active_plot_mode_keys:
            st.markdown("**Curve style per plot**")
            columns = st.columns(2)
            for idx, plot_key in enumerate(active_plot_mode_keys):
                with columns[idx % 2]:
                    st.selectbox(
                        _REPORT_PLOT_MODE_LABELS[plot_key],
                        _PLOT_MODE_OPTIONS,
                        key=_report_plot_mode_state_key(plot_key),
                    )
        marker_size = st.slider(
            "Marker size (px)",
            1,
            20,
            int(st.session_state.get("rb_marker_size", 6)),
            key="rb_marker_size",
        )
        opacity = st.slider(
            "Raw trace opacity",
            0.1,
            1.0,
            float(st.session_state.get("rb_opacity", 0.8)),
            0.05,
            key="rb_opacity",
        )
        max_raw_points = st.slider(
            "Max plotted points per test",
            1000,
            500000,
            int(st.session_state.get("rb_maxpts", 50000)),
            1000,
            key="rb_maxpts",
            help=(
                "Caps dense raw/time-series traces in the report preview and "
                "export so large reports stay responsive."
            ),
        )
        best_mean_pct = st.slider(
            "Best-region high-flow %",
            50,
            99,
            int(st.session_state.get("rb_best_mean", 75)),
            key="rb_best_mean",
            help="Keeps the top X% of bins by mean flow before the stability filter is applied.",
        )
        best_std_pct = st.slider(
            "Best-region stability %",
            1,
            50,
            int(st.session_state.get("rb_best_std", 10)),
            key="rb_best_std",
            help="Within the high-flow subset, keeps the lowest X% by standard deviation.",
        )
        bin_reco = st.session_state.get("_rb_bin_recommendation")
        if (
            auto_use_recommended_avg_bin
            and _comparisons_use_average_bin(selected_comparisons)
        ):
            if bin_reco and int(bin_reco.get("average_series_count", 0)) >= 2:
                st.caption(
                    "Average-style plots will automatically use "
                    f"{float(bin_reco['average_bin_hz']):g} Hz "
                    "for the shared averaging grid when you generate the report."
                )
            else:
                st.caption(
                    "Average-style plots are set to use the recommended averaging "
                    "bin automatically when the report is generated."
                )
        if bin_reco and int(bin_reco.get("test_series_count", 0)) >= 2:
            st.caption(
                "Recommended per-test plot bin: "
                + explain_frequency_bin_recommendation(
                    bin_reco,
                    include_average_bin=False,
                )
            )
            st.button(
                "Reset plot bin to recommended",
                key="rb_reset_plot_bin",
                on_click=_reset_report_plot_bin,
            )
        if bin_reco and int(bin_reco.get("average_series_count", 0)) >= 2:
            st.caption(
                "Recommended averaging bin: "
                + explain_frequency_bin_recommendation(bin_reco)
            )
            st.button(
                "Reset averaging bin to recommended",
                key="rb_reset_bin",
                on_click=_reset_report_bin,
            )

        st.caption("Leave axis-limit fields blank to keep auto-ranging.")
        sweep_axis_required = bool(
            set(selected_comparisons)
            & {"sweep_overlay", "individual_sweeps", "global_average", "raw_points"}
        )
        relative_axis_required = "sweep_relative" in selected_comparisons
        time_axis_required = "constant_time_series" in selected_comparisons
        variability_axis_required = bool(
            set(selected_comparisons) & {"std_vs_mean", "best_region"}
        )
        if sweep_axis_required:
            _render_axis_inputs(
                "Frequency vs Flow axes",
                prefix="rb_sweep_axis",
                x_label="Frequency",
                y_label="Flow",
            )
        if relative_axis_required:
            _render_axis_inputs(
                "Relative sweep axes",
                prefix="rb_relative_axis",
                x_label="Frequency",
                y_label="Relative flow",
            )
        if time_axis_required:
            _render_axis_inputs(
                "Time-series axes",
                prefix="rb_time_axis",
                x_label="Time",
                y_label="Flow",
            )
        if variability_axis_required:
            _render_axis_inputs(
                "Mean-vs-std axes",
                prefix="rb_variability_axis",
                x_label="Mean flow",
                y_label="Std",
            )

    axis_errors: list[str] = []
    sweep_axis = (
        _read_axis_bounds_from_state(
            "rb_sweep_axis",
            label="Frequency vs Flow axes",
            errors=axis_errors,
        )
        if sweep_axis_required
        else AxisBounds()
    )
    relative_axis = (
        _read_axis_bounds_from_state(
            "rb_relative_axis",
            label="Relative sweep axes",
            errors=axis_errors,
        )
        if relative_axis_required
        else AxisBounds()
    )
    time_axis = (
        _read_axis_bounds_from_state(
            "rb_time_axis",
            label="Time-series axes",
            errors=axis_errors,
        )
        if time_axis_required
        else AxisBounds()
    )
    variability_axis = (
        _read_axis_bounds_from_state(
            "rb_variability_axis",
            label="Mean-vs-std axes",
            errors=axis_errors,
        )
        if variability_axis_required
        else AxisBounds()
    )
    if axis_errors:
        for err in axis_errors:
            st.error(err)
        return

    # ── Build definition ────────────────────────────────────────
    plot_modes = _collect_report_plot_modes()
    unique_plot_modes = {mode for mode in plot_modes.values()}
    fallback_plot_mode = (
        next(iter(unique_plot_modes))
        if len(unique_plot_modes) == 1
        else _normalize_report_plot_mode(st.session_state.get("rb_mode", "lines+markers"))
    )
    defn = ReportDefinition(
        title=title,
        author=author,
        entry_ids=selected_entries,
        comparisons=selected_comparisons,
        notes=notes,
        bin_hz=plot_bin_hz,
        avg_bin_hz=avg_bin_hz,
        show_error_bars=show_err,
        show_individual_tests=show_indiv,
        show_raw_all_sweeps=show_raw_all_sweeps,
        plot_mode=fallback_plot_mode,
        plot_modes=plot_modes,
        marker_size=marker_size,
        opacity=opacity,
        max_raw_points=max_raw_points,
        mean_threshold_pct=best_mean_pct,
        std_threshold_pct=best_std_pct,
        sweep_axis=sweep_axis,
        relative_axis=relative_axis,
        time_axis=time_axis,
        variability_axis=variability_axis,
        selection_mode=selection_mode,
        overlay_entry_ids=overlay_selected_entries,
        auto_use_recommended_avg_bin=auto_use_recommended_avg_bin,
        overlay_include_sweep_drilldown=bool(
            st.session_state.get("rb_overlay_sweep_drilldown", False)
        ),
    )

    # ── Save template ───────────────────────────────────────────
    with st.expander("💾 Save as template", expanded=False):
        with st.form("rb_save_template", clear_on_submit=True):
            tmpl_name = st.text_input("Template name")
            if st.form_submit_button("Save"):
                if tmpl_name.strip():
                    save_report_definition(tmpl_name.strip(), defn)
                    st.success(f"Saved template **{tmpl_name.strip()}**")
                    st.rerun()
                else:
                    st.error("Name required.")

    _render_report_warmup_controls(
        defn=defn,
        selected_entries=selected_entries,
        selected_folders=selected_folders,
        data_folder_str=data_folder_str,
    )

    st.divider()

    # ── Check for unknown tests ──────────────────────────────────
    _all_report_folders: list[str] = []
    for entry_id in selected_entries:
        resolved = _resolve_report_entry(entry_id, reg, run_names)
        if resolved is None:
            continue
        _, folders, _ = resolved
        _all_report_folders.extend(folders)

    if _all_report_folders and run_dirs:
        _, _unknowns = classify_tests_quick(
            _all_report_folders, run_dirs, run_names,
        )
        if _unknowns:
            st.subheader("❓ Unknown Tests")
            render_unknown_test_prompt(
                _unknowns, run_dirs, run_names,
                key_prefix="rb_utp",
            )
            st.info(
                "⏸️ Classify all unknown tests above before "
                "generating a report."
            )
            return  # block generation until classified

    # ── Generate & preview ──────────────────────────────────────
    if not selected_comparisons:
        st.info("Select at least one comparison type above.")
        return

    render_preview = st.checkbox(
        "Render inline preview after generation",
        value=bool(st.session_state.get("rb_render_preview", False)),
        key="rb_render_preview",
        help=(
            "Disable this for faster report generation when you only need the "
            "exported HTML."
        ),
    )
    if set(selected_comparisons) & {"individual_sweeps", "constant_time_series", "raw_points"}:
        st.caption(
            "Performance note: these selections create one or more plots per test. "
            "Inline preview can become slow when many tests are included."
        )

    generate_requested = st.button("🔨 Generate Report", type="primary", key="rb_generate")
    if generate_requested:
        _generate_and_preview(
            defn,
            reg,
            run_names,
            run_dirs,
            render_preview=render_preview,
        )

    # ── If already generated, show preview + export ─────────────
    if "rb_html" in st.session_state and st.session_state.get("rb_html"):
        st.divider()
        st.subheader("📤 Export")

        col_fn, col_dl = st.columns([3, 1])
        with col_fn:
            now_str = datetime.now().strftime("%Y-%m-%d_%H%M")
            safe_title = "".join(
                c if c.isalnum() or c in " _-" else "_"
                for c in defn.title
            ).strip().replace(" ", "_")
            default_fn = f"{safe_title}_{now_str}.html"
            filename = st.text_input("Filename", value=default_fn, key="rb_fn")

        with col_dl:
            st.markdown("&nbsp;")
            st.download_button(
                "⬇️ Download HTML",
                data=st.session_state["rb_html"],
                file_name=filename or default_fn,
                mime="text/html",
                key="rb_download",
            )

        # Also save to Downloads folder
        if st.button("💾 Save to ~/Downloads", key="rb_save_dl"):
            path = save_report(
                st.session_state["rb_html"],
                filename or default_fn,
            )
            st.success(f"✅ Saved to `{path}`")

    if not generate_requested:
        warmup_job = load_warmup_job(REPORT_WARMUP_JOB_KEY)
        warmup_summary = summarize_warmup_job(warmup_job)
        worker_summary = summarize_warmup_worker_state(
            load_warmup_worker_state(REPORT_WARMUP_JOB_KEY)
        )
        if (
            warmup_job
            and warmup_job.get("auto_resume", True)
            and not warmup_summary.get("paused", False)
            and not worker_summary.get("running", False)
            and warmup_summary["pending"] > 0
        ):
            updated_job = run_report_warmup_job(
                run_names,
                run_dirs,
                max_tasks=1,
                time_budget_s=1.25,
                job_key=REPORT_WARMUP_JOB_KEY,
            )
            updated_summary = summarize_warmup_job(updated_job)
            if updated_summary["pending"] > 0 and updated_job:
                st.rerun()


def _generate_and_preview(
    defn: ReportDefinition,
    reg: PumpRegistry,
    run_names: list[str],
    run_dirs: list,
    *,
    render_preview: bool = False,
) -> None:
    """Generate report sections, store HTML in session state, show preview."""
    ap = _get_analysis_plots()
    bcp = _get_bar_comparison_plots()

    sections: list[ReportSection] = []
    entry_metadata: dict[str, dict] = {}

    # ── Collect test folders from selected report targets ──────
    entry_tests: dict[str, list[str]] = {}
    entry_display_names: dict[str, str] = {}
    for eid in defn.entry_ids:
        resolved = _resolve_report_entry(eid, reg, run_names)
        if resolved is None:
            continue
        display_name, folders, metadata = resolved
        entry_tests[eid] = folders
        entry_display_names[eid] = display_name
        entry_metadata[eid] = metadata

    all_folders = [f for folders in entry_tests.values() for f in folders]
    if not all_folders:
        st.warning(
            "⚠️ None of the selected report targets have test folders "
            "available in the current data folder."
        )
        return

    effective_defn = defn
    if (
        defn.auto_use_recommended_avg_bin
        and _comparisons_use_average_bin(defn.comparisons)
    ):
        recommendation = _ensure_report_bin_recommendation(
            selected_folders=all_folders,
            run_names=run_names,
            run_dirs=run_dirs,
        )
        if recommendation and int(recommendation.get("average_series_count", 0)) >= 2:
            recommended_avg_bin = float(recommendation["average_bin_hz"])
            effective_defn = replace(defn, avg_bin_hz=recommended_avg_bin)
            st.session_state["rb_avg_bin"] = recommended_avg_bin

    normalized_overlay_entry_ids = [
        entry_id for entry_id in effective_defn.overlay_entry_ids if entry_id in entry_tests
    ]
    if not normalized_overlay_entry_ids and entry_tests:
        normalized_overlay_entry_ids = list(entry_tests.keys())[:1]
    effective_defn = replace(
        effective_defn,
        overlay_entry_ids=normalized_overlay_entry_ids,
    )

    # ── Load data ───────────────────────────────────────────────
    with st.spinner(f"Loading {len(all_folders)} test(s)…"):
        loaded = _load_all_test_data(
            all_folders,
            run_names,
            run_dirs,
            effective_defn.bin_hz,
            max_raw_points=effective_defn.max_raw_points,
        )

    if loaded is None:
        st.error("No data could be loaded.")
        return

    (all_raw, sweep_raw, binned_data, const_raw,
     signal_col, errors) = loaded

    if errors:
        with st.expander(f"⚠️ {len(errors)} load issue(s)"):
            for e in errors:
                st.warning(e)

    # ── Build display-name mappings for nicer labels ────────────
    folder_to_display: dict[str, str] = {}
    plot_cache_contexts: dict[str, dict[str, object]] = {}
    for eid, folders in entry_tests.items():
        display_name = entry_display_names[eid]
        for f in folders:
            display_label = f"{display_name} / {f[:30]}"
            folder_to_display[f] = display_label
            run_dir = next((path for name, path in zip(run_names, run_dirs) if name == f), None)
            if run_dir is not None:
                plot_cache_contexts[display_label] = get_test_cache_context(f, run_dir)

    def _rename_keys(d: dict, mapping: dict) -> dict:
        return {mapping.get(k, k): v for k, v in d.items()}

    # Rename for display
    display_binned = _rename_keys(binned_data, folder_to_display)
    display_raw = _rename_keys(all_raw, folder_to_display)
    display_sweep = _rename_keys(sweep_raw, folder_to_display)
    display_const = _rename_keys(const_raw, folder_to_display)

    avg_binned_data: dict[str, pd.DataFrame] = {}
    display_avg_binned: dict[str, pd.DataFrame] = {}
    if _comparisons_use_average_bin(effective_defn.comparisons) and sweep_raw:
        avg_binned_data = _build_common_grid_binned_sweeps(
            sweep_raw,
            signal_col=signal_col,
            bin_hz=effective_defn.avg_bin_hz,
        )
        display_avg_binned = _rename_keys(avg_binned_data, folder_to_display)

    # Build bar-level grouping for bar comparison plots
    bar_binned: dict[str, dict[str, pd.DataFrame]] = {}
    bar_avg_binned: dict[str, dict[str, pd.DataFrame]] = {}
    bar_sweep: dict[str, dict[str, pd.DataFrame]] = {}
    bar_const: dict[str, dict[str, pd.DataFrame]] = {}
    for eid, folders in entry_tests.items():
        dname = entry_display_names[eid]
        bar_binned[dname] = {}
        bar_avg_binned[dname] = {}
        bar_sweep[dname] = {}
        bar_const[dname] = {}
        for f in folders:
            if f in binned_data:
                bar_binned[dname][f] = binned_data[f]
            if f in avg_binned_data:
                bar_avg_binned[dname][f] = avg_binned_data[f]
            if f in sweep_raw:
                bar_sweep[dname][folder_to_display.get(f, f)] = sweep_raw[f]
            if f in const_raw:
                bar_const[dname][f] = const_raw[f]
        if not bar_binned[dname]:
            del bar_binned[dname]
        if not bar_avg_binned[dname]:
            del bar_avg_binned[dname]
        if not bar_sweep[dname]:
            del bar_sweep[dname]
        if not bar_const[dname]:
            del bar_const[dname]

    overlay_folders = [
        folder
        for entry_id in effective_defn.overlay_entry_ids
        for folder in entry_tests.get(entry_id, [])
    ]
    overlay_display_binned = _rename_keys(
        {
            folder: binned_data[folder]
            for folder in overlay_folders
            if folder in binned_data
        },
        folder_to_display,
    )
    overlay_bar_sweep = {
        entry_display_names[eid]: bar_sweep[entry_display_names[eid]]
        for eid in effective_defn.overlay_entry_ids
        if entry_display_names.get(eid) in bar_sweep
    }

    # ── Generate requested sections ─────────────────────────────
    n_sweep = len(display_binned)
    n_const = len(display_const)

    # Info section
    sections.append(ReportSection(
        kind="text",
        title="Data Summary",
        content=(
            f"Loaded {len(all_folders)} test(s) from "
            f"{len(entry_tests)} report target(s): "
            f"{n_sweep} frequency-sweep, {n_const} constant-frequency."
        ),
    ))
    sections.append(ReportSection(kind="divider"))

    for comp in defn.comparisons:
        try:
            _add_comparison_section(
                comp, sections, effective_defn,
                display_binned, display_avg_binned, overlay_display_binned,
                display_raw, display_sweep, display_const,
                bar_binned, bar_avg_binned, overlay_bar_sweep, bar_const,
                plot_cache_contexts,
                signal_col,
                ap, bcp,
            )
            sections.append(ReportSection(kind="divider"))
        except Exception as exc:
            sections.append(ReportSection(
                kind="text",
                title=f"Error: {COMPARISON_OPTIONS.get(comp, comp)}",
                content=f"Could not generate: {exc}",
            ))

    for sec in sections:
        if sec.kind == "plot":
            _apply_report_plot_layout(sec.content)

    # ── Build HTML ──────────────────────────────────────────────
    with st.spinner("Building HTML report…"):
        html = build_report_html(effective_defn, sections, entry_metadata)
        st.session_state["rb_html"] = html

    st.success(
        f"✅ Report generated — {len(sections)} sections, "
        f"{len(html) / 1024:.0f} KB"
    )

    if not render_preview:
        st.info(
            "Inline preview was skipped to keep generation responsive. "
            "Use the export controls below, or regenerate with preview enabled."
        )
        return

    # ── Preview ─────────────────────────────────────────────────
    with st.expander("👁️ Preview (charts are shown inline)", expanded=True):
        for sec in sections:
            if sec.kind == "heading":
                st.subheader(sec.title)
            elif sec.kind == "text":
                if sec.title:
                    st.markdown(f"**{sec.title}**")
                st.markdown(str(sec.content))
            elif sec.kind == "plot":
                if sec.title:
                    st.markdown(f"**{sec.title}**")
                if sec.description:
                    subtitle, details = _split_report_section_description(sec.description)
                    if subtitle:
                        st.caption(subtitle)
                    if details:
                        with st.expander("More detail", expanded=False):
                            st.caption(details)
                st.plotly_chart(sec.content, use_container_width=True)
            elif sec.kind == "table":
                if sec.title:
                    st.markdown(f"**{sec.title}**")
                if sec.description:
                    subtitle, details = _split_report_section_description(sec.description)
                    if subtitle:
                        st.caption(subtitle)
                    if details:
                        with st.expander("More detail", expanded=False):
                            st.caption(details)
                if sec.collapsible:
                    with st.expander("Show table", expanded=not sec.collapsed):
                        st.dataframe(sec.content, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(sec.content, use_container_width=True, hide_index=True)
            elif sec.kind == "divider":
                st.divider()


def _add_comparison_section(
    comp: str,
    sections: list[ReportSection],
    defn: ReportDefinition,
    display_binned: dict,
    display_avg_binned: dict,
    overlay_display_binned: dict,
    display_raw: dict,
    display_sweep: dict,
    display_const: dict,
    bar_binned: dict,
    bar_avg_binned: dict,
    overlay_bar_sweep: dict,
    bar_const: dict,
    plot_cache_contexts: dict[str, dict[str, object]] | None,
    signal_col: str,
    ap,
    bcp,
) -> None:
    """Add section(s) for one comparison type."""
    bin_reco = st.session_state.get("_rb_bin_recommendation")

    if comp == "sweep_overlay":
        if bar_avg_binned:
            target_mode = _resolve_report_plot_mode(defn, "sweep_overlay_target")
            fig = bcp.plot_bar_sweep_overlay(
                bar_avg_binned,
                bin_hz=defn.avg_bin_hz,
                show_error_bars=defn.show_error_bars,
                show_individual=defn.show_individual_tests,
                mode=target_mode,
                marker_size=defn.marker_size,
            )
            _apply_axis_bounds(fig, defn.sweep_axis)
            fig.update_layout(
                title=(
                    "Frequency Sweep — Report Target Comparison "
                    f"({format_bin_choice_label(float(defn.avg_bin_hz), bin_reco, use_average_bin=True)})"
                )
            )
            sections.append(ReportSection(
                kind="plot",
                title=(
                    "Frequency Sweep — Report Target Comparison "
                    f"({format_bin_choice_label(float(defn.avg_bin_hz), bin_reco, use_average_bin=True)})"
                ),
                content=fig,
                description=_build_guided_section_description(
                    "sweep_overlay",
                    "What you are seeing: each thick curve is the average frequency-response for one selected report target.",
                    (
                        f"Method: raw sweep points were binned directly onto one shared {defn.avg_bin_hz:g} Hz averaging grid for every test, "
                        "then the tests inside each selected target were averaged on that same grid. "
                        + (
                            "The error bars show ±1 standard deviation between tests inside the same target."
                            if defn.show_error_bars
                            else "The curves are shown without ±1 standard deviation error bars."
                        )
                    ),
                    (
                        "Interpretation note: a comb-like or wavy appearance usually comes from discrete sweep steps, "
                        "up/down sweep hysteresis, and repeated visits to nearby frequencies. Once the data are binned by frequency, "
                        "this pattern is usually not driven mainly by small differences in when each sweep started."
                    ),
                    _format_sweep_plot_context(
                        defn,
                        use_average_bin=True,
                        include_test_bin=False,
                        include_individual_tests=defn.show_individual_tests,
                        axis_bounds=defn.sweep_axis,
                    ),
                    "Average-plot note: the per-test display bin is not reused here, so this view is driven directly by the raw sweep points on the shared averaging grid.",
                    _format_render_context(
                        defn,
                        plot_mode=target_mode,
                        axis_bounds=defn.sweep_axis,
                        x_name="frequency",
                        y_name="flow",
                    ),
                    _format_nested_test_listing(bar_avg_binned),
                ),
            ))
        focused_overlay_targets = list(overlay_bar_sweep.keys())
        overlay_scope_suffix = _format_overlay_scope_suffix(
            focused_overlay_targets,
            total_target_count=max(len(bar_binned), 1),
        )
        if overlay_display_binned:
            tests_mode = _resolve_report_plot_mode(defn, "sweep_overlay_tests")
            overlay_title_base = f"Frequency Sweep — All Tests Overlay{overlay_scope_suffix}"
            overlay_title = (
                f"{overlay_title_base} "
                f"({format_bin_choice_label(float(defn.bin_hz), bin_reco)})"
            )
            fig = ap.plot_combined_overlay(
                overlay_display_binned,
                show_error_bars=defn.show_error_bars,
                mode=tests_mode,
                marker_size=defn.marker_size,
                title=overlay_title,
            )
            _apply_axis_bounds(fig, defn.sweep_axis)
            sections.append(ReportSection(
                kind="plot",
                title=overlay_title,
                content=fig,
                description=_build_guided_section_description(
                    "sweep_overlay",
                    "What you are seeing: each curve is one individual test after per-test frequency binning.",
                    (
                        f"Method: raw sweep points were grouped into {defn.bin_hz:g} Hz bins and then plotted directly per test. "
                        "The overlay therefore compares both response magnitude and where peaks or troughs occur."
                    ),
                    (
                        "Focus note: this figure only includes the overlay targets chosen in Plot settings so the legend stays readable."
                        if focused_overlay_targets
                        else ""
                    ),
                    (
                        "Interpretation note: repeating comb-like teeth usually reflect discrete setpoints and sweep hysteresis more than small differences "
                        "in where each sweep happened to begin."
                    ),
                    _format_sweep_plot_context(defn, axis_bounds=defn.sweep_axis),
                    _format_render_context(
                        defn,
                        plot_mode=tests_mode,
                        axis_bounds=defn.sweep_axis,
                        x_name="frequency",
                        y_name="flow",
                    ),
                    _format_flat_test_listing(list(overlay_display_binned.keys())),
                ),
            ))
        if defn.overlay_include_sweep_drilldown and overlay_bar_sweep:
            for target_name, target_sweeps in overlay_bar_sweep.items():
                if not target_sweeps:
                    continue
                fig = ap.plot_target_sweep_drilldown(
                    target_sweeps,
                    signal_col=signal_col,
                    bin_hz=defn.bin_hz,
                    title=f"Frequency Sweep — All Sweeps Drill-Down — {target_name}",
                    mode=_resolve_report_plot_mode(defn, "individual_per_sweep"),
                    marker_size=max(3, defn.marker_size - 1),
                )
                _apply_axis_bounds(fig, defn.sweep_axis)
                sections.append(ReportSection(
                    kind="plot",
                    title=f"Frequency Sweep — All Sweeps Drill-Down — {target_name}",
                    content=fig,
                    description=_build_guided_section_description(
                        "individual_sweeps",
                        "What you are seeing: every sweep from every focused test inside this target is overlaid together.",
                        (
                            f"Method: raw sweep points were split by sweep index, each sweep was binned at {defn.bin_hz:g} Hz, "
                            "and the resulting sweep curves were overlaid without averaging them together."
                        ),
                        "Use this after the focused all-tests overlay when you need to see whether the spread comes from one unstable test or from sweep-to-sweep variation inside several tests.",
                        _format_sweep_plot_context(defn, axis_bounds=defn.sweep_axis),
                        _format_flat_test_listing(list(target_sweeps.keys())),
                    ),
                ))

    elif comp == "sweep_relative":
        if bar_avg_binned:
            target_mode = _resolve_report_plot_mode(defn, "sweep_relative_target")
            fig = bcp.plot_bar_sweep_relative(
                bar_avg_binned,
                bin_hz=defn.avg_bin_hz,
                mode=target_mode,
                marker_size=defn.marker_size,
            )
            _apply_axis_bounds(fig, defn.relative_axis)
            fig.update_layout(
                title=(
                    "Relative Sweep (0–100 %) — Report Target Comparison "
                    f"({format_bin_choice_label(float(defn.avg_bin_hz), bin_reco, use_average_bin=True)})"
                )
            )
            sections.append(ReportSection(
                kind="plot",
                title=(
                    "Relative Sweep (0–100 %) — Report Target Comparison "
                    f"({format_bin_choice_label(float(defn.avg_bin_hz), bin_reco, use_average_bin=True)})"
                ),
                content=fig,
                description=_build_guided_section_description(
                    "sweep_relative",
                    "What you are seeing: each target’s average sweep has been rescaled so its own minimum becomes 0% and its own maximum becomes 100%.",
                    (
                        f"Method: the same raw-sweep averaging path used in the target comparison was built on a shared {defn.avg_bin_hz:g} Hz grid and then normalized independently to remove absolute flow level "
                        "and emphasize shape differences across frequency."
                    ),
                    "Average-plot note: the normalization is applied after the raw sweep data have been rebinned directly on the shared averaging grid.",
                    _format_sweep_plot_context(
                        defn,
                        use_average_bin=True,
                        include_test_bin=False,
                        include_individual_tests=defn.show_individual_tests,
                    ),
                    _format_render_context(
                        defn,
                        plot_mode=target_mode,
                        axis_bounds=defn.relative_axis,
                        x_name="frequency",
                        y_name="relative flow",
                        include_opacity=False,
                    ),
                    _format_nested_test_listing(bar_avg_binned),
                ),
            ))
        if len(display_binned) > 1:
            tests_mode = _resolve_report_plot_mode(defn, "sweep_relative_tests")
            fig = ap.plot_relative_comparison(
                display_binned,
                mode=tests_mode,
                marker_size=defn.marker_size,
                title=(
                    "Relative Sweep (0–100 %) — All Tests "
                    f"({format_bin_choice_label(float(defn.bin_hz), bin_reco)})"
                ),
            )
            _apply_axis_bounds(fig, defn.relative_axis)
            sections.append(ReportSection(
                kind="plot",
                title=(
                    "Relative Sweep (0–100 %) — All Tests "
                    f"({format_bin_choice_label(float(defn.bin_hz), bin_reco)})"
                ),
                content=fig,
                description=_build_guided_section_description(
                    "sweep_relative",
                    "What you are seeing: each test is normalized to its own 0–100% range so only the profile shape remains.",
                    "Method: per-test binned mean curves are rescaled independently before plotting, so only shape differences remain and absolute flow levels are intentionally removed.",
                    _format_sweep_plot_context(defn),
                    _format_render_context(
                        defn,
                        plot_mode=tests_mode,
                        axis_bounds=defn.relative_axis,
                        x_name="frequency",
                        y_name="relative flow",
                        include_opacity=False,
                    ),
                    _format_flat_test_listing(list(display_binned.keys())),
                ),
            ))

    elif comp == "individual_sweeps":
        from ..plots.plot_generator import (
            downsample_sweep_points,
            plot_sweep_all_points,
            plot_sweep_binned,
        )
        for name, binned in display_binned.items():
            binned_mode = _resolve_report_plot_mode(defn, "individual_binned")
            cache_context = (plot_cache_contexts or {}).get(name)
            fig = _get_cached_test_figure(
                cache_context=cache_context,
                plot_kind="sweep_binned",
                settings={
                    "bin_hz": float(defn.bin_hz),
                    "mode": binned_mode,
                    "marker_size": int(defn.marker_size),
                    "show_error_bars": bool(defn.show_error_bars),
                },
                builder=lambda: plot_sweep_binned(
                    binned,
                    title="",
                    mode=binned_mode,
                    marker_size=defn.marker_size,
                    show_error_bars=defn.show_error_bars,
                ),
            )
            fig.update_layout(
                title=(
                    f"{name} — Binned Mean "
                    f"({format_bin_choice_label(float(defn.bin_hz), bin_reco)})"
                )
            )
            _apply_axis_bounds(fig, defn.sweep_axis)
            sections.append(ReportSection(
                kind="plot",
                title=(
                    f"{name} — Binned Mean "
                    f"({format_bin_choice_label(float(defn.bin_hz), bin_reco)})"
                ),
                content=fig,
                description=_build_guided_section_description(
                    "individual_sweeps",
                    "What you are seeing: the overall mean flow-frequency curve for this single test after binning.",
                    (
                        f"Method: all sweep points from this test were grouped into {defn.bin_hz:g} Hz frequency bins and summarized as mean flow, "
                        + (
                            "with ±1 standard deviation error bars."
                            if defn.show_error_bars
                            else "without ±1 standard deviation error bars."
                        )
                    ),
                    _format_sweep_plot_context(defn, axis_bounds=defn.sweep_axis),
                    _format_render_context(
                        defn,
                        plot_mode=binned_mode,
                        axis_bounds=defn.sweep_axis,
                        x_name="frequency",
                        y_name="flow",
                    ),
                    _format_flat_test_listing([name]),
                ),
            ))
            sweep_df = display_sweep.get(name)
            if sweep_df is not None and not sweep_df.empty:
                per_sweep_mode = _resolve_report_plot_mode(defn, "individual_per_sweep")
                fig_per_sweep = _get_cached_test_figure(
                    cache_context=cache_context,
                    plot_kind="sweep_per_sweep_average",
                    settings={
                        "bin_hz": float(defn.bin_hz),
                        "signal_col": signal_col,
                        "mode": per_sweep_mode,
                        "marker_size": int(defn.marker_size),
                    },
                    builder=lambda: ap.plot_per_test_sweeps(
                        sweep_df,
                        signal_col=signal_col,
                        bin_hz=defn.bin_hz,
                        title="",
                        mode=per_sweep_mode,
                        marker_size=defn.marker_size,
                    ),
                )
                fig_per_sweep.update_layout(title=f"{name} — Per-Sweep Breakdown")
                _apply_axis_bounds(fig_per_sweep, defn.sweep_axis)
                sections.append(ReportSection(
                    kind="plot",
                    title=f"{name} — Per-Sweep Breakdown",
                    content=fig_per_sweep,
                    description=_build_guided_section_description(
                        "individual_sweeps",
                        "What you are seeing: each line is one sweep cycle from the same test, binned separately.",
                        (
                            f"Method: the raw data were first split by sweep index and then each sweep was binned with width {defn.bin_hz:g} Hz. "
                            "This exposes whether one sweep repeats another or whether there is cycle-to-cycle hysteresis."
                        ),
                        _build_sweep_start_alignment_summary(
                            sweep_df,
                            bin_hz=defn.bin_hz,
                        ),
                        _format_sweep_plot_context(defn, axis_bounds=defn.sweep_axis),
                        _format_render_context(
                            defn,
                            plot_mode=per_sweep_mode,
                            axis_bounds=defn.sweep_axis,
                            x_name="frequency",
                            y_name="flow",
                        ),
                        _format_flat_test_listing([name]),
                    ),
                ))

                if defn.show_raw_all_sweeps:
                    avg_overlay = _prepare_average_overlay(binned)
                    raw_sweep_df = downsample_sweep_points(
                        sweep_df,
                        max_points=defn.max_raw_points,
                    )
                    raw_sweeps_mode = _resolve_report_plot_mode(
                        defn,
                        "individual_raw_all_sweeps",
                    )
                    fig_all_sweeps = _get_cached_test_figure(
                        cache_context=cache_context,
                        plot_kind="sweep_all_points",
                        settings={
                            "max_raw_points": int(defn.max_raw_points),
                            "signal_col": signal_col,
                            "mode": raw_sweeps_mode,
                            "marker_size": int(defn.marker_size),
                            "opacity": float(defn.opacity),
                            "show_average_error_bars": bool(defn.show_error_bars),
                            "has_average_overlay": bool(not avg_overlay.empty),
                        },
                        builder=lambda: plot_sweep_all_points(
                            raw_sweep_df,
                            x_col="Frequency",
                            y_col=signal_col,
                            color_col="Sweep" if "Sweep" in sweep_df.columns else None,
                            title="",
                            mode=raw_sweeps_mode,
                            marker_size=defn.marker_size,
                            opacity=defn.opacity,
                            average_df=avg_overlay if not avg_overlay.empty else None,
                            show_average_error_bars=defn.show_error_bars,
                        ),
                    )
                    fig_all_sweeps.update_layout(title=f"{name} — Raw All-Sweeps Layer")
                    _apply_axis_bounds(fig_all_sweeps, defn.sweep_axis)
                    sections.append(ReportSection(
                        kind="plot",
                        title=f"{name} — Raw All-Sweeps Layer",
                        content=fig_all_sweeps,
                        description=_build_guided_section_description(
                            "raw_points",
                            "What you are seeing: every raw sweep point is shown, with sweep cycles separated by color and the overall average overlaid in black.",
                            (
                                "Method: no cross-sweep averaging is applied to the colored traces. This is the best view for checking whether all sweeps are present "
                                "and whether the same frequencies are revisited consistently."
                            ),
                            _build_sweep_start_alignment_summary(
                                sweep_df,
                                bin_hz=defn.bin_hz,
                            ),
                            (
                                "Interpretation note: repeated vertical or comb-like bands usually indicate that different sweep cycles hit similar frequency values "
                                "with different flow responses. That is more often a sweep-cycle or hysteresis effect than a simple horizontal offset from where each sweep started."
                            ),
                            _format_raw_sweep_context(
                                defn,
                                axis_bounds=defn.sweep_axis,
                                max_raw_points=defn.max_raw_points,
                                average_overlay_bin_hz=defn.bin_hz,
                            ),
                            _format_render_context(
                                defn,
                                plot_mode=raw_sweeps_mode,
                                axis_bounds=defn.sweep_axis,
                                x_name="frequency",
                                y_name="flow",
                            ),
                            _format_flat_test_listing([name]),
                        ),
                    ))

    elif comp == "global_average":
        if len(display_avg_binned) >= 2:
            global_average_mode = _resolve_report_plot_mode(defn, "global_average")
            fig, avg_df = ap.plot_global_average(
                display_avg_binned,
                bin_hz=defn.avg_bin_hz,
                title=(
                    "Global Average Across Tests "
                    f"({format_bin_choice_label(float(defn.avg_bin_hz), bin_reco, use_average_bin=True)})"
                ),
                mode=global_average_mode,
                marker_size=defn.marker_size,
                show_error_bars=defn.show_error_bars,
            )
            _apply_axis_bounds(fig, defn.sweep_axis)
            sections.append(ReportSection(
                kind="plot",
                title=(
                    "Global Average Across Tests "
                    f"({format_bin_choice_label(float(defn.avg_bin_hz), bin_reco, use_average_bin=True)})"
                ),
                content=fig,
                description=_build_guided_section_description(
                    "global_average",
                    "What you are seeing: a single mean response computed across all selected tests.",
                    (
                        f"Method: raw sweep points from each test were binned directly onto one shared {defn.avg_bin_hz:g} Hz grid before calculating the cross-test average. "
                        + (
                            "The error bars show inter-test ±1 standard deviation."
                            if defn.show_error_bars
                            else "No inter-test standard-deviation error bars are shown."
                        )
                    ),
                    "Average-plot note: this plot does not reuse the display-bin curves from the all-tests overlay, so the result is not a second average of already-averaged 5 Hz curves.",
                    _format_sweep_plot_context(
                        defn,
                        use_average_bin=True,
                        include_test_bin=False,
                        axis_bounds=defn.sweep_axis,
                    ),
                    _format_render_context(
                        defn,
                        plot_mode=global_average_mode,
                        axis_bounds=defn.sweep_axis,
                        x_name="frequency",
                        y_name="flow",
                    ),
                    _format_flat_test_listing(list(display_avg_binned.keys())),
                ),
            ))
            if not avg_df.empty:
                sections.append(ReportSection(
                    kind="table",
                    title="Average Curve Data",
                    content=avg_df,
                    description=_build_guided_section_description(
                        "global_average",
                        "Tabulated frequency-grid values behind the global-average plot.",
                        f"Method: the same shared {defn.avg_bin_hz:g} Hz raw-sweep averaging grid used in the plot above.",
                        _format_flat_test_listing(list(display_avg_binned.keys())),
                    ),
                    collapsible=True,
                    collapsed=True,
                ))

    elif comp == "boxplots":
        if bar_const:
            fig = bcp.plot_bar_constant_boxplots(
                bar_const, signal_col=signal_col,
            )
            sections.append(ReportSection(
                kind="plot",
                title="Constant-Frequency Boxplots — per Report Target",
                content=fig,
                description=_build_guided_section_description(
                    "boxplots",
                    "What you are seeing: one boxplot per constant-frequency test, grouped by selected report target.",
                    (
                        "Method: each box summarizes the distribution of raw flow measurements for one constant-frequency test. "
                        "The center line is the median, the box spans the interquartile range, and the whiskers/outliers show spread."
                    ),
                    _format_nested_test_listing(bar_const),
                ),
            ))
            fig2 = bcp.plot_bar_constant_aggregated(
                bar_const, signal_col=signal_col,
            )
            sections.append(ReportSection(
                kind="plot",
                title="Aggregated Boxplots per Report Target",
                content=fig2,
                description=_build_guided_section_description(
                    "boxplots",
                    "What you are seeing: all constant-frequency points are pooled within each selected target and summarized in one boxplot.",
                    "Method: raw flow points from all constant tests inside the same target are pooled before calculating the distribution statistics.",
                    _format_nested_test_listing(bar_const),
                ),
            ))
        if display_sweep or display_const:
            data = {**display_sweep, **display_const}
            data = {k: v for k, v in data.items() if signal_col in v.columns}
            if data:
                fig = ap.plot_combined_boxplots(data, signal_col=signal_col)
                sections.append(ReportSection(
                    kind="plot",
                    title="Flow Distribution Boxplots — All Tests",
                    content=fig,
                    description=_build_guided_section_description(
                        "boxplots",
                        "What you are seeing: one boxplot per individual test, combining all selected test types.",
                        "Method: each box summarizes the raw flow distribution for one test without pooling across targets.",
                        _format_flat_test_listing(list(data.keys())),
                    ),
                ))

    elif comp == "histograms":
        if bar_const:
            fig = bcp.plot_bar_constant_histograms(
                bar_const, signal_col=signal_col,
            )
            sections.append(ReportSection(
                kind="plot",
                title="Constant-Frequency Histograms — per Report Target",
                content=fig,
                description=_build_guided_section_description(
                    "histograms",
                    "What you are seeing: pooled flow histograms for constant-frequency tests, one color per selected target.",
                    "Method: all raw constant-frequency points inside a target are pooled and displayed as an overlaid histogram.",
                    _format_nested_test_listing(bar_const),
                ),
            ))
        if display_sweep or display_const:
            data = {**display_sweep, **display_const}
            data = {k: v for k, v in data.items() if signal_col in v.columns}
            if data:
                fig = ap.plot_combined_histograms(data, signal_col=signal_col)
                sections.append(ReportSection(
                    kind="plot",
                    title="Flow Histograms — All Tests",
                    content=fig,
                    description=_build_guided_section_description(
                        "histograms",
                        "What you are seeing: the flow-value distribution of each test as an overlaid histogram.",
                        "Method: raw points are binned by flow magnitude rather than frequency or time, which highlights skewness, multimodality, and spread.",
                        _format_flat_test_listing(list(data.keys())),
                    ),
                ))

    elif comp == "summary_table":
        if bar_binned:
            for test_type_label, data_dict in [
                ("Sweep", bar_binned), ("Constant", bar_const),
            ]:
                if data_dict:
                    # Collect raw data for summary
                    bar_raw_for_summary: dict[str, dict[str, pd.DataFrame]] = {}
                    for bname, tests in data_dict.items():
                        bar_raw_for_summary[bname] = {}
                        for tname, tdf in tests.items():
                            bar_raw_for_summary[bname][tname] = tdf
                    tbl = bcp.build_bar_summary_table(
                        bar_raw_for_summary,
                        signal_col=signal_col if test_type_label == "Constant" else "mean",
                        test_type=test_type_label,
                    )
                    if not tbl.empty:
                        sections.append(ReportSection(
                            kind="table",
                            title=f"Summary Statistics — {test_type_label} Tests (per Entry)",
                            content=tbl,
                            description=_build_guided_section_description(
                                "summary_table",
                                f"Summary statistics pooled by selected target for {test_type_label.lower()} tests.",
                                (
                                    "Method note: frequency-binned mean values are pooled for sweep tests."
                                    if test_type_label == "Sweep"
                                    else "Method note: raw cleaned time-series points are pooled directly for constant tests."
                                ),
                                _format_nested_test_listing(data_dict),
                            ),
                        ))

        # Per-test summary
        data = {**display_sweep, **display_const}
        data = {k: v for k, v in data.items() if signal_col in v.columns}
        if data:
            tbl = ap.build_summary_table(data, signal_col=signal_col)
            if not tbl.empty:
                sections.append(ReportSection(
                    kind="table",
                    title="Summary Statistics — All Tests",
                    content=tbl,
                    description=_build_guided_section_description(
                        "summary_table",
                        "Per-test summary statistics using the raw points supplied to the report.",
                        "Method note: sweep tests contribute prepared sweep points, while constant tests contribute cleaned time-series points.",
                        _format_flat_test_listing(list(data.keys())),
                    ),
                ))

    elif comp == "constant_time_series":
        from ..plots.plot_generator import plot_time_series

        if not display_const:
            return

        for name, df in display_const.items():
            if df.empty or signal_col not in df.columns:
                continue
            plot_df = _downsample_frame_evenly(df, defn.max_raw_points)
            x_col = plot_df.columns[0]
            time_series_mode = _resolve_report_plot_mode(defn, "constant_time_series")
            cache_context = (plot_cache_contexts or {}).get(name)
            fig = _get_cached_test_figure(
                cache_context=cache_context,
                plot_kind="time_series",
                settings={
                    "max_raw_points": int(defn.max_raw_points),
                    "signal_col": signal_col,
                    "mode": time_series_mode,
                    "marker_size": int(defn.marker_size),
                    "opacity": float(defn.opacity),
                    "x_col": x_col,
                },
                builder=lambda: plot_time_series(
                    plot_df,
                    x_col=x_col,
                    y_col=signal_col,
                    title="",
                    mode=time_series_mode,
                    marker_size=defn.marker_size,
                    opacity=defn.opacity,
                ),
            )
            fig.update_layout(title=f"{name} — Flow vs Time")
            _apply_axis_bounds(fig, defn.time_axis)
            sections.append(ReportSection(
                kind="plot",
                title=f"{name} — Flow vs Time",
                content=fig,
                description=_build_guided_section_description(
                    "constant_time_series",
                    "What you are seeing: raw flow versus time for one constant-frequency test.",
                    "Method: the cleaned time-series data are plotted directly without frequency binning so drift, warm-up, settling, or decay remain visible.",
                    _build_time_effect_summary(df, signal_col),
                    (
                        "Plot context: no frequency binning is applied in this view."
                        if len(plot_df) == len(df)
                        else (
                            "Plot context: no frequency binning is applied in this view. "
                            f"For responsiveness, the plotted trace was downsampled to {len(plot_df):,} "
                            f"evenly spaced points from {len(df):,} raw points."
                        )
                    ),
                    _format_render_context(
                        defn,
                        plot_mode=time_series_mode,
                        axis_bounds=defn.time_axis,
                        x_name="time",
                        y_name="flow",
                    ),
                    _format_flat_test_listing([name]),
                ),
            ))
        time_effect_df = _build_time_effect_table(display_const, signal_col)
        if not time_effect_df.empty:
            sections.append(ReportSection(
                kind="table",
                title="Constant-Frequency Time-Effect Summary",
                content=time_effect_df,
                description=_build_guided_section_description(
                    "constant_time_series",
                    "Tabulated correlation between time and flow for each constant-frequency test included in the report.",
                    "Method: Pearson correlation and a first-order linear-fit slope are computed from the cleaned time-series points of each constant test.",
                    _format_flat_test_listing(list(display_const.keys())),
                ),
            ))

    elif comp == "raw_points":
        if display_sweep:
            from ..plots.plot_generator import downsample_sweep_points

            capped = {}
            for n, d in display_sweep.items():
                capped[n] = downsample_sweep_points(
                    d,
                    max_points=defn.max_raw_points,
                )
            raw_points_mode = _resolve_report_plot_mode(defn, "raw_points")
            fig = ap.plot_all_raw_points(
                capped,
                freq_col="Frequency",
                signal_col=signal_col,
                mode=raw_points_mode,
                marker_size=defn.marker_size,
                opacity=defn.opacity,
            )
            _apply_axis_bounds(fig, defn.sweep_axis)
            sections.append(ReportSection(
                kind="plot",
                title="All Raw Sweep Points",
                content=fig,
                description=_build_guided_section_description(
                    "raw_points",
                    "What you are seeing: every raw sweep point that was kept for the report, colored by test.",
                    f"Method: sweep data are plotted directly against frequency without averaging. For report size control, each test is capped at {defn.max_raw_points:,} points before plotting.",
                    _format_raw_sweep_context(
                        defn,
                        axis_bounds=defn.sweep_axis,
                        max_raw_points=defn.max_raw_points,
                    ),
                    _format_render_context(
                        defn,
                        plot_mode=raw_points_mode,
                        axis_bounds=defn.sweep_axis,
                        x_name="frequency",
                        y_name="flow",
                    ),
                    _format_flat_test_listing(list(capped.keys())),
                ),
            ))

    elif comp == "std_vs_mean":
        if display_binned:
            fig = ap.plot_std_vs_mean(
                display_binned,
                marker_size=defn.marker_size,
            )
            _apply_axis_bounds(fig, defn.variability_axis)
            sections.append(ReportSection(
                kind="plot",
                title="Std vs Mean — Variability Analysis",
                content=fig,
                description=_build_guided_section_description(
                    "std_vs_mean",
                    "What you are seeing: each point is one frequency bin from one test, positioned by mean flow and within-bin standard deviation.",
                    "Method: after frequency binning, variability is summarized per bin and compared to its mean to assess whether noise scales with flow.",
                    _format_sweep_plot_context(defn, axis_bounds=defn.sweep_axis),
                    _format_variability_context(defn, axis_bounds=defn.variability_axis),
                    _format_render_context(defn, axis_bounds=defn.variability_axis, x_name="mean flow", y_name="std", include_opacity=False),
                    _format_flat_test_listing(list(display_binned.keys())),
                ),
            ))

    elif comp == "best_region":
        if display_binned:
            fig, best_df = ap.plot_stability_cloud(
                display_binned,
                mean_threshold_pct=float(defn.mean_threshold_pct),
                std_threshold_pct=float(defn.std_threshold_pct),
                marker_size=defn.marker_size,
            )
            _apply_axis_bounds(fig, defn.variability_axis)
            sections.append(ReportSection(
                kind="plot",
                title="Best Operating Region",
                content=fig,
                description=_build_guided_section_description(
                    "best_region",
                    "What you are seeing: candidate operating points that combine high mean flow with low variability.",
                    (
                        f"Method: frequency bins are first filtered to the top {defn.mean_threshold_pct}% by mean flow, "
                        f"then the lowest {defn.std_threshold_pct}% by standard deviation are retained within that high-flow subset."
                    ),
                    _format_sweep_plot_context(defn, axis_bounds=defn.sweep_axis),
                    _format_variability_context(
                        defn,
                        axis_bounds=defn.variability_axis,
                        include_thresholds=True,
                    ),
                    _format_render_context(defn, axis_bounds=defn.variability_axis, x_name="mean flow", y_name="std", include_opacity=False),
                    _format_flat_test_listing(list(display_binned.keys())),
                ),
            ))
            if not best_df.empty:
                best_table = (
                    best_df[["test", "freq_center", "mean", "std"]]
                    .rename(
                        columns={
                            "test": "Test",
                            "freq_center": "Frequency (Hz)",
                            "mean": "Mean (µL/min)",
                            "std": "Std (µL/min)",
                        }
                    )
                    .sort_values(
                        ["Std (µL/min)", "Mean (µL/min)"],
                        ascending=[True, False],
                    )
                )
                sections.append(ReportSection(
                    kind="table",
                    title="Top Operating Points",
                    content=best_table,
                    description=_build_guided_section_description(
                        "best_region",
                        "Ranked table of the operating points selected by the best-region method.",
                        "Ranking note: rows are ordered first by lower standard deviation and then by higher mean flow, so stability is prioritized after the high-flow filter.",
                        _format_flat_test_listing(list(display_binned.keys())),
                    ),
                ))

    elif comp == "correlation":
        if len(display_binned) >= 2:
            fig = ap.plot_correlation_heatmap(display_binned)
            sections.append(ReportSection(
                kind="plot",
                title="Inter-Test Correlation Heatmap",
                content=fig,
                description=_build_guided_section_description(
                    "correlation",
                    "What you are seeing: the pairwise Pearson correlation between binned mean-flow curves.",
                    "Method: each test is represented by its binned mean curve across frequency; correlations close to 1 indicate very similar shapes across the shared frequency range.",
                    _format_sweep_plot_context(defn, axis_bounds=defn.sweep_axis),
                    _format_flat_test_listing(list(display_binned.keys())),
                ),
            ))


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════

def _load_all_test_data(
    folder_names: list[str],
    run_names: list[str],
    run_dirs: list,
    bin_hz: float,
    *,
    max_raw_points: int | None = None,
) -> tuple | None:
    """Load and classify all test folders.

    Returns
    -------
    (all_raw, sweep_raw, binned_data, const_raw, signal_col, errors)
    or None.
    """
    result = load_and_classify_tests(
        folder_names,
        run_dirs,
        run_names,
        bin_hz=float(bin_hz),
        max_raw_points=max_raw_points,
    )

    if result.is_empty:
        return None

    return (
        result.all_data,
        result.sweep_data,
        result.binned_data,
        result.const_data,
        result.signal_col,
        result.errors,
    )


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — AUDIT LOG
# ════════════════════════════════════════════════════════════════════════

def _render_audit_tab() -> None:
    """Display the full audit trail of registry changes."""
    st.subheader("📜 Audit Log")
    st.caption(
        "Complete history of all registry changes — entries created, "
        "renamed, deleted; tests linked / unlinked; bulk imports."
    )

    records = load_audit_log()
    if not records:
        st.info("No audit records yet.")
        return

    # Show most recent first
    records = list(reversed(records))

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        action_filter = st.multiselect(
            "Filter by action",
            sorted({r.action for r in records}),
            key="rb_audit_action",
        )
    with col_f2:
        target_filter = st.text_input(
            "Filter by target (entry ID)",
            key="rb_audit_target",
        )

    filtered = records
    if action_filter:
        filtered = [r for r in filtered if r.action in action_filter]
    if target_filter.strip():
        q = target_filter.strip().lower()
        filtered = [r for r in filtered if q in r.target.lower()]

    st.markdown(f"Showing **{len(filtered)}** of {len(records)} records:")

    # Render as a table
    rows = []
    for r in filtered:
        details_str = ""
        if r.details:
            detail_parts = []
            for k, v in r.details.items():
                if isinstance(v, list):
                    detail_parts.append(f"{k}: [{len(v)} items]")
                else:
                    detail_parts.append(f"{k}: {v}")
            details_str = "; ".join(detail_parts)

        rows.append({
            "Timestamp": r.timestamp,
            "Action": r.action,
            "Target": r.target,
            "Details": details_str,
            "Author": r.author,
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Details": st.column_config.TextColumn(width="large"),
            },
        )
