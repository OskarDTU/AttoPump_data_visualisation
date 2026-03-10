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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus, unquote_plus

import numpy as np
import pandas as pd
import streamlit as st

from ..data.io_local import pick_best_csv
from ..data.config import PLOT_BIN_WIDTH_HZ, PLOT_HEIGHT
from ..data.data_processor import (
    bin_by_frequency,
    detect_constant_frequency,
    detect_test_type,
    detect_time_format,
    guess_signal_column,
    guess_time_column,
    parse_sweep_spec_from_name,
    prepare_sweep_data,
    prepare_time_series_data,
)
from ..data.loader import load_csv_cached, resolve_data_path
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
    save_registry,
    unlink_test,
    upsert_sub_group,
)
from ..reports.generator import (
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
    if key not in st.session_state:
        st.session_state[key] = migrate_legacy_files()
    return st.session_state[key]


def _persist_registry() -> None:
    """Flush the registry to disk."""
    save_registry(_get_registry())


_SELECTION_MODE_OPTIONS = {
    "pumps": "Whole pumps",
    "sub_groups": "Saved pump sub-groups",
}
_SUB_GROUP_ENTRY_PREFIX = "__pump_sub_group__"


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


def _prepare_average_overlay(binned_df: pd.DataFrame) -> pd.DataFrame:
    """Adapt a binned sweep DataFrame for the raw all-sweeps overlay."""
    overlay = binned_df.rename(columns={"freq_center": "freq"})
    overlay = overlay[[col for col in ["freq", "mean", "std"] if col in overlay.columns]].copy()
    return overlay


def _time_axis_seconds(series: pd.Series) -> tuple[pd.Series, str]:
    """Convert a time-like axis to elapsed seconds for correlation analysis."""
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        return (parsed - parsed.min()).dt.total_seconds(), "elapsed seconds"

    numeric = pd.to_numeric(series, errors="coerce")
    return numeric, str(series.name or "time")


def _build_time_effect_summary(df: pd.DataFrame, signal_col: str) -> str:
    """Summarize the strength of any time-effect in a constant test."""
    if df.empty or signal_col not in df.columns:
        return "Time-effect analysis: unavailable because the signal column is missing."

    x_col = df.columns[0]
    x_values, x_label = _time_axis_seconds(df[x_col])
    y_values = pd.to_numeric(df[signal_col], errors="coerce")
    mask = x_values.notna() & y_values.notna()
    if mask.sum() < 3:
        return "Time-effect analysis: not enough valid points for a correlation estimate."

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

    return (
        f"Time-effect analysis: Pearson correlation between {x_label} and flow = "
        f"{corr:.3f} ({strength}). Estimated slope = {slope:.4f} µL/min per {x_label}. "
        f"{interpretation}"
    )


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
            _render_build_tab(run_names, run_dirs)

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

def _render_build_tab(run_names: list[str], run_dirs: list) -> None:
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
        if load_choice != "(new report)" and st.button("🗑️", key="rb_del_saved"):
            delete_saved_report(load_choice)
            st.rerun()

    loaded_defn: ReportDefinition | None = None
    if load_choice != "(new report)":
        loaded_defn = load_report_definition(load_choice)

    st.divider()

    # ── Report metadata ─────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input(
            "Report title",
            value=loaded_defn.title if loaded_defn else "AttoPump Test Report",
            key="rb_title",
        )
    with col2:
        author = st.text_input(
            "Author",
            value=loaded_defn.author if loaded_defn else "",
            key="rb_author",
        )
    notes = st.text_area(
        "Report notes / description",
        value=loaded_defn.notes if loaded_defn else "",
        key="rb_notes",
        height=80,
    )

    # ── Select report targets ───────────────────────────────────
    st.subheader("📦 Select Report Targets")
    default_mode = (
        loaded_defn.selection_mode
        if loaded_defn and loaded_defn.selection_mode in _SELECTION_MODE_OPTIONS
        else "pumps"
    )
    load_marker = load_choice if loaded_defn else "(new report)"
    if st.session_state.get("_rb_loaded_template") != load_marker:
        st.session_state["_rb_loaded_template"] = load_marker
        st.session_state["rb_selection_mode"] = default_mode
        st.session_state["rb_entries_pumps"] = []
        st.session_state["rb_entries_sub_groups"] = []
        if loaded_defn:
            target_key = (
                "rb_entries_sub_groups"
                if default_mode == "sub_groups"
                else "rb_entries_pumps"
            )
            st.session_state[target_key] = list(loaded_defn.entry_ids)

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

    # ── Comparison types ────────────────────────────────────────
    st.subheader("📊 Comparisons")
    default_comps = (
        loaded_defn.comparisons if loaded_defn
        else ["sweep_overlay", "summary_table", "boxplots", "constant_time_series"]
    )
    selected_comparisons = st.multiselect(
        "Choose what to include in the report",
        list(COMPARISON_OPTIONS.keys()),
        default=[c for c in default_comps if c in COMPARISON_OPTIONS],
        format_func=lambda k: COMPARISON_OPTIONS[k],
        key="rb_comparisons",
    )

    # ── Plot settings ───────────────────────────────────────────
    with st.expander("⚙️ Plot settings", expanded=False):
        bin_hz = st.slider(
            "Frequency bin width (Hz)", 0.5, 100.0, 5.0, 0.5,
            key="rb_bin",
        )
        show_err = st.checkbox("Show ±1 std bands", True, key="rb_err")
        show_indiv = st.checkbox(
            "Show individual test lines behind averages",
            False, key="rb_indiv",
        )

    # ── Build definition ────────────────────────────────────────
    defn = ReportDefinition(
        title=title,
        author=author,
        entry_ids=selected_entries,
        comparisons=selected_comparisons,
        notes=notes,
        bin_hz=bin_hz,
        show_error_bars=show_err,
        show_individual_tests=show_indiv,
        selection_mode=selection_mode,
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

    if st.button("🔨 Generate Report Preview", type="primary", key="rb_generate"):
        _generate_and_preview(defn, reg, run_names, run_dirs)

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


def _generate_and_preview(
    defn: ReportDefinition,
    reg: PumpRegistry,
    run_names: list[str],
    run_dirs: list,
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

    # ── Load data ───────────────────────────────────────────────
    with st.spinner(f"Loading {len(all_folders)} test(s)…"):
        loaded = _load_all_test_data(
            all_folders, run_names, run_dirs, defn.bin_hz,
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
    for eid, folders in entry_tests.items():
        display_name = entry_display_names[eid]
        for f in folders:
            folder_to_display[f] = f"{display_name} / {f[:30]}"

    def _rename_keys(d: dict, mapping: dict) -> dict:
        return {mapping.get(k, k): v for k, v in d.items()}

    # Rename for display
    display_binned = _rename_keys(binned_data, folder_to_display)
    display_raw = _rename_keys(all_raw, folder_to_display)
    display_sweep = _rename_keys(sweep_raw, folder_to_display)
    display_const = _rename_keys(const_raw, folder_to_display)

    # Build bar-level grouping for bar comparison plots
    bar_binned: dict[str, dict[str, pd.DataFrame]] = {}
    bar_const: dict[str, dict[str, pd.DataFrame]] = {}
    for eid, folders in entry_tests.items():
        dname = entry_display_names[eid]
        bar_binned[dname] = {}
        bar_const[dname] = {}
        for f in folders:
            if f in binned_data:
                bar_binned[dname][f] = binned_data[f]
            if f in const_raw:
                bar_const[dname][f] = const_raw[f]
        if not bar_binned[dname]:
            del bar_binned[dname]
        if not bar_const[dname]:
            del bar_const[dname]

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
                comp, sections, defn,
                display_binned, display_raw, display_sweep, display_const,
                bar_binned, bar_const,
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

    # ── Build HTML ──────────────────────────────────────────────
    with st.spinner("Building HTML report…"):
        html = build_report_html(defn, sections, entry_metadata)
        st.session_state["rb_html"] = html

    st.success(
        f"✅ Report generated — {len(sections)} sections, "
        f"{len(html) / 1024:.0f} KB"
    )

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
                    st.caption(sec.description)
                st.plotly_chart(sec.content, use_container_width=True)
            elif sec.kind == "table":
                if sec.title:
                    st.markdown(f"**{sec.title}**")
                if sec.description:
                    st.caption(sec.description)
                st.dataframe(sec.content, use_container_width=True, hide_index=True)
            elif sec.kind == "divider":
                st.divider()


def _add_comparison_section(
    comp: str,
    sections: list[ReportSection],
    defn: ReportDefinition,
    display_binned: dict,
    display_raw: dict,
    display_sweep: dict,
    display_const: dict,
    bar_binned: dict,
    bar_const: dict,
    signal_col: str,
    ap,
    bcp,
) -> None:
    """Add section(s) for one comparison type."""

    if comp == "sweep_overlay":
        if bar_binned:
            fig = bcp.plot_bar_sweep_overlay(
                bar_binned,
                show_error_bars=defn.show_error_bars,
                show_individual=defn.show_individual_tests,
            )
            sections.append(ReportSection(
                kind="plot",
                title="Frequency Sweep — Report Target Comparison",
                content=fig,
                description=_build_section_description(
                    "What you are seeing: each thick curve is the average frequency-response for one selected report target.",
                    (
                        f"Method: raw sweep points were grouped into {defn.bin_hz:g} Hz frequency bins inside each test. "
                        "Those test-level mean curves were then averaged within each selected target. "
                        + (
                            "The shaded band shows ±1 standard deviation between tests inside the same target."
                            if defn.show_error_bars
                            else "The curves are shown without ±1 standard deviation shading."
                        )
                    ),
                    (
                        "Interpretation note: a comb-like or wavy appearance usually comes from discrete sweep steps, "
                        "up/down sweep hysteresis, and repeated visits to nearby frequencies. Once the data are binned by frequency, "
                        "this pattern is usually not driven mainly by small differences in when each sweep started."
                    ),
                    _format_nested_test_listing(bar_binned),
                ),
            ))
        if len(display_binned) > 1:
            fig = ap.plot_combined_overlay(
                display_binned,
                show_error_bars=defn.show_error_bars,
            )
            sections.append(ReportSection(
                kind="plot",
                title="Frequency Sweep — All Tests Overlay",
                content=fig,
                description=_build_section_description(
                    "What you are seeing: each curve is one individual test after frequency binning.",
                    (
                        f"Method: raw sweep points were grouped into {defn.bin_hz:g} Hz bins and then plotted directly per test."
                    ),
                    _format_flat_test_listing(list(display_binned.keys())),
                ),
            ))

    elif comp == "sweep_relative":
        if bar_binned:
            fig = bcp.plot_bar_sweep_relative(bar_binned)
            sections.append(ReportSection(
                kind="plot",
                title="Relative Sweep (0–100 %) — Report Target Comparison",
                content=fig,
                description=_build_section_description(
                    "What you are seeing: each target’s average sweep has been rescaled so its own minimum becomes 0% and its own maximum becomes 100%.",
                    (
                        "Method: the same target-level average curves used in the sweep overlay were normalized independently to remove absolute flow level "
                        "and emphasize shape differences across frequency."
                    ),
                    _format_nested_test_listing(bar_binned),
                ),
            ))
        if len(display_binned) > 1:
            fig = ap.plot_relative_comparison(display_binned)
            sections.append(ReportSection(
                kind="plot",
                title="Relative Sweep (0–100 %) — All Tests",
                content=fig,
                description=_build_section_description(
                    "What you are seeing: each test is normalized to its own 0–100% range so only the profile shape remains.",
                    "Method: per-test binned mean curves are rescaled independently before plotting.",
                    _format_flat_test_listing(list(display_binned.keys())),
                ),
            ))

    elif comp == "individual_sweeps":
        from ..plots.plot_generator import plot_sweep_all_points, plot_sweep_binned
        for name, binned in display_binned.items():
            fig = plot_sweep_binned(
                binned,
                title=f"{name} — Binned Mean",
                show_error_bars=defn.show_error_bars,
            )
            sections.append(ReportSection(
                kind="plot",
                title=f"{name} — Binned Mean",
                content=fig,
                description=_build_section_description(
                    "What you are seeing: the overall mean flow-frequency curve for this single test after binning.",
                    (
                        f"Method: all sweep points from this test were grouped into {defn.bin_hz:g} Hz frequency bins and summarized as mean flow, "
                        + (
                            "with a ±1 standard deviation band."
                            if defn.show_error_bars
                            else "without a ±1 standard deviation band."
                        )
                    ),
                    _format_flat_test_listing([name]),
                ),
            ))
            sweep_df = display_sweep.get(name)
            if sweep_df is not None and not sweep_df.empty:
                fig_per_sweep = ap.plot_per_test_sweeps(
                    sweep_df,
                    signal_col=signal_col,
                    bin_hz=defn.bin_hz,
                    title=f"{name} — Per-Sweep Breakdown",
                )
                sections.append(ReportSection(
                    kind="plot",
                    title=f"{name} — Per-Sweep Breakdown",
                    content=fig_per_sweep,
                    description=_build_section_description(
                        "What you are seeing: each line is one sweep cycle from the same test, binned separately.",
                        (
                            f"Method: the raw data were first split by sweep index and then each sweep was binned with width {defn.bin_hz:g} Hz. "
                            "This exposes whether one sweep repeats another or whether there is cycle-to-cycle hysteresis."
                        ),
                        _format_flat_test_listing([name]),
                    ),
                ))

                avg_overlay = _prepare_average_overlay(binned)
                fig_all_sweeps = plot_sweep_all_points(
                    sweep_df,
                    x_col="Frequency",
                    y_col=signal_col,
                    color_col="Sweep" if "Sweep" in sweep_df.columns else None,
                    title=f"{name} — Raw All-Sweeps Layer",
                    average_df=avg_overlay if not avg_overlay.empty else None,
                    show_average_error_bars=defn.show_error_bars,
                )
                sections.append(ReportSection(
                    kind="plot",
                    title=f"{name} — Raw All-Sweeps Layer",
                    content=fig_all_sweeps,
                    description=_build_section_description(
                        "What you are seeing: every raw sweep point is shown, with sweep cycles separated by color and the overall average overlaid in black.",
                        (
                            "Method: no cross-sweep averaging is applied to the colored traces. This is the best view for checking whether all sweeps are present "
                            "and whether the same frequencies are revisited consistently."
                        ),
                        (
                            "Interpretation note: repeated vertical or comb-like bands usually indicate that different sweep cycles hit similar frequency values "
                            "with different flow responses. That is more often a sweep-cycle or hysteresis effect than a simple horizontal offset from where each sweep started."
                        ),
                        _format_flat_test_listing([name]),
                    ),
                ))

    elif comp == "global_average":
        if len(display_binned) >= 2:
            fig, avg_df = ap.plot_global_average(
                display_binned,
                bin_hz=defn.bin_hz,
                show_error_bars=defn.show_error_bars,
            )
            sections.append(ReportSection(
                kind="plot",
                title="Global Average Across Tests",
                content=fig,
                description=_build_section_description(
                    "What you are seeing: a single mean response computed across all selected tests.",
                    (
                        f"Method: each test was aligned onto a common {defn.bin_hz:g} Hz frequency grid and then averaged across tests at each grid point. "
                        + (
                            "The band shows inter-test ±1 standard deviation."
                            if defn.show_error_bars
                            else "No inter-test standard-deviation band is shown."
                        )
                    ),
                    _format_flat_test_listing(list(display_binned.keys())),
                ),
            ))
            if not avg_df.empty:
                sections.append(ReportSection(
                    kind="table",
                    title="Average Curve Data",
                    content=avg_df,
                    description=_build_section_description(
                        "Tabulated frequency-grid values behind the global-average plot.",
                        f"Method: same common-grid averaging as the plot above, using {defn.bin_hz:g} Hz bins.",
                    ),
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
                description=_build_section_description(
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
                description=_build_section_description(
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
                    description=_build_section_description(
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
                description=_build_section_description(
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
                    description=_build_section_description(
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
                            description=_build_section_description(
                                f"Summary statistics pooled by selected target for {test_type_label.lower()} tests.",
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
                    description=_build_section_description(
                        "Per-test summary statistics using the raw points supplied to the report.",
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
            x_col = df.columns[0]
            fig = plot_time_series(
                df,
                x_col=x_col,
                y_col=signal_col,
                title=f"{name} — Flow vs Time",
                mode="lines",
            )
            sections.append(ReportSection(
                kind="plot",
                title=f"{name} — Flow vs Time",
                content=fig,
                description=_build_section_description(
                    "What you are seeing: raw flow versus time for one constant-frequency test.",
                    "Method: the cleaned time-series data are plotted directly without frequency binning so drift, warm-up, settling, or decay remain visible.",
                    _build_time_effect_summary(df, signal_col),
                    _format_flat_test_listing([name]),
                ),
            ))

    elif comp == "raw_points":
        if display_sweep:
            capped = {}
            for n, d in display_sweep.items():
                capped[n] = d.sample(n=50000, random_state=42) if len(d) > 50000 else d
            fig = ap.plot_all_raw_points(
                capped,
                freq_col="Frequency",
                signal_col=signal_col,
            )
            sections.append(ReportSection(
                kind="plot",
                title="All Raw Data Points",
                content=fig,
                description=_build_section_description(
                    "What you are seeing: every raw sweep point that was kept for the report, colored by test.",
                    "Method: sweep data are plotted directly against frequency without averaging. For report size control, each test is capped at 50,000 points before plotting.",
                    _format_flat_test_listing(list(capped.keys())),
                ),
            ))

    elif comp == "std_vs_mean":
        if display_binned:
            fig = ap.plot_std_vs_mean(display_binned)
            sections.append(ReportSection(
                kind="plot",
                title="Std vs Mean — Variability Analysis",
                content=fig,
                description=_build_section_description(
                    "What you are seeing: each point is one frequency bin from one test, positioned by mean flow and within-bin standard deviation.",
                    "Method: after frequency binning, variability is summarized per bin and compared to its mean to assess whether noise scales with flow.",
                    _format_flat_test_listing(list(display_binned.keys())),
                ),
            ))

    elif comp == "best_region":
        if display_binned:
            fig, best_df = ap.plot_stability_cloud(display_binned)
            sections.append(ReportSection(
                kind="plot",
                title="Best Operating Region",
                content=fig,
                description=_build_section_description(
                    "What you are seeing: candidate operating points that combine high mean flow with low variability.",
                    "Method: frequency bins are filtered first by mean-flow percentile and then by low-standard-deviation percentile to identify the most stable high-output region.",
                    _format_flat_test_listing(list(display_binned.keys())),
                ),
            ))
            if not best_df.empty:
                sections.append(ReportSection(
                    kind="table",
                    title="Top Operating Points",
                    content=best_df[["test", "freq_center", "mean", "std"]].rename(
                        columns={
                            "test": "Test",
                            "freq_center": "Frequency (Hz)",
                            "mean": "Mean (µL/min)",
                            "std": "Std (µL/min)",
                        }
                    ).sort_values("Mean (µL/min)", ascending=False),
                    description="Ranked table of the strongest operating points selected by the stability-cloud method.",
                ))

    elif comp == "correlation":
        if len(display_binned) >= 2:
            fig = ap.plot_correlation_heatmap(display_binned)
            sections.append(ReportSection(
                kind="plot",
                title="Inter-Test Correlation Heatmap",
                content=fig,
                description=_build_section_description(
                    "What you are seeing: the pairwise Pearson correlation between binned mean-flow curves.",
                    "Method: each test is represented by its binned mean curve across frequency; correlations close to 1 indicate very similar shapes across the shared frequency range.",
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
) -> tuple | None:
    """Load and classify all test folders.

    Returns
    -------
    (all_raw, sweep_raw, binned_data, const_raw, signal_col, errors)
    or None.
    """
    all_raw: dict[str, pd.DataFrame] = {}
    sweep_raw: dict[str, pd.DataFrame] = {}
    binned_data: dict[str, pd.DataFrame] = {}
    const_raw: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    signal_col: str | None = None

    for name in folder_names:
        idx = run_names.index(name) if name in run_names else -1
        if idx < 0:
            errors.append(f"{name}: not found in data folder")
            continue
        run_dir = run_dirs[idx]
        try:
            pick = pick_best_csv(run_dir)
            df = load_csv_cached(str(pick.csv_path))
            if df.empty:
                errors.append(f"{name}: empty CSV")
                continue

            time_col = guess_time_column(df)
            sig_col = guess_signal_column(df, "flow")
            if not time_col or not sig_col:
                errors.append(f"{name}: cannot detect columns")
                continue
            if signal_col is None:
                signal_col = sig_col

            time_fmt = detect_time_format(df, time_col)
            ts_df = prepare_time_series_data(
                df, time_col, sig_col,
                parse_time=(time_fmt == "absolute_timestamp"),
            )
            all_raw[name] = ts_df

            ttype, _, _ = detect_test_type(name, df, data_root=run_dir.parent)
            has_freq = "freq_set_hz" in df.columns

            if ttype == "sweep" or (has_freq and df["freq_set_hz"].dropna().nunique() > 1):
                spec = parse_sweep_spec_from_name(name)
                if has_freq or (spec and spec.duration_s > 0):
                    sdf = prepare_sweep_data(
                        ts_df, time_col, sig_col,
                        spec=spec,
                        parse_time=(time_fmt == "absolute_timestamp"),
                        full_df=df if has_freq else None,
                    )
                    sweep_raw[name] = sdf
                    try:
                        binned = bin_by_frequency(
                            sdf, value_col=sig_col,
                            freq_col="Frequency", bin_hz=bin_hz,
                        )
                        binned_data[name] = binned
                    except Exception as be:
                        errors.append(f"{name}: binning — {be}")
                else:
                    const_raw[name] = ts_df
            else:
                const_raw[name] = ts_df

        except Exception as e:
            errors.append(f"{name}: {e}")

    if not all_raw:
        return None

    if signal_col is None:
        signal_col = "flow"

    return (all_raw, sweep_raw, binned_data, const_raw, signal_col, errors)


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
