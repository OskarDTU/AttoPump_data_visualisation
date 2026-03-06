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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ..data.io_local import (
    list_run_dirs,
    normalize_root,
    pick_best_csv,
    read_csv_full,
)
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
from ..data.bar_registry import (
    BarRegistry,
    RegistryEntry,
    TestLink,
    add_entry,
    all_test_folders,
    get_display_name,
    import_from_dataframe,
    link_test,
    load_audit_log,
    load_registry,
    remove_entry,
    rename_entry,
    save_registry,
    unlink_test,
    update_entry_notes,
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

def _get_registry() -> BarRegistry:
    """Return the in-memory pump registry (loads from disk on first access)."""
    key = "_bar_registry"
    if key not in st.session_state:
        st.session_state[key] = load_registry()
    return st.session_state[key]


def _persist_registry() -> None:
    """Flush the registry to disk."""
    save_registry(_get_registry())


@st.cache_data(ttl=300, show_spinner=False)
def _load_csv_cached(csv_path_str: str) -> pd.DataFrame:
    """Load a CSV with Streamlit caching (5-min TTL)."""
    return read_csv_full(csv_path_str)


# ════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Entry point for the Report Builder page."""
    try:
        st.title("📑 Report Builder")

        # ── Sidebar — Data Source ───────────────────────────────
        with st.sidebar:
            st.header("📁 Data Source")
            if "last_data_path" not in st.session_state:
                st.session_state.last_data_path = ""
            data_folder_str = st.text_input(
                "Path to test data folder",
                value=st.session_state.last_data_path,
                placeholder="/Users/.../All_tests",
                key="rb_data_path",
            )

        # ── Resolve data path ──────────────────────────────────
        run_dirs: list = []
        run_names: list[str] = []
        if data_folder_str.strip():
            try:
                root = normalize_root(data_folder_str)
                if data_folder_str != st.session_state.last_data_path:
                    st.session_state.last_data_path = data_folder_str
                run_dirs = list(list_run_dirs(root))
                run_names = [p.name for p in run_dirs]
            except Exception as e:
                st.sidebar.error(f"❌ Invalid path: {e}")

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
    with st.expander("📥 Import from spreadsheet", expanded=not bool(reg.entries)):
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
    with st.expander("➕ Add new entry manually"):
        with st.form("rb_add_entry", clear_on_submit=True):
            new_id = st.text_input(
                "Entry ID (e.g. 'Pump 270226-2')"
            )
            new_display = st.text_input(
                "Display name (optional, defaults to entry ID)"
            )
            new_notes = st.text_area("Notes (optional)")
            if st.form_submit_button("Add Entry"):
                if new_id.strip():
                    entry = RegistryEntry(
                        entry_id=new_id.strip(),
                        display_name=new_display.strip() or new_id.strip(),
                        notes=new_notes.strip(),
                    )
                    add_entry(reg, entry)
                    _persist_registry()
                    st.success(f"✅ Added **{new_id.strip()}**")
                    st.rerun()
                else:
                    st.error("Entry ID is required.")

    # ── Display existing entries ────────────────────────────────
    st.divider()
    if not reg.entries:
        st.info(
            "📭 Registry is empty.  Import a spreadsheet or add entries "
            "manually above."
        )
        return

    st.markdown(f"**{len(reg.entries)} entries** in registry:")

    for eid, entry in sorted(reg.entries.items()):
        with st.expander(
            f"{'🔵' if entry.tests else '⚪'} {entry.display_name}  "
            f"({len(entry.tests)} tests)",
            expanded=False,
        ):
            # ── Edit display name ───────────────────────────────
            col_name, col_del = st.columns([4, 1])
            with col_name:
                new_name = st.text_input(
                    "Display name",
                    value=entry.display_name,
                    key=f"rb_name_{eid}",
                )
                if new_name != entry.display_name:
                    if st.button("Save name", key=f"rb_savename_{eid}"):
                        rename_entry(reg, eid, new_name)
                        _persist_registry()
                        st.rerun()
            with col_del:
                st.markdown("&nbsp;")  # spacing
                if st.button("🗑️ Delete entry", key=f"rb_del_{eid}"):
                    remove_entry(reg, eid)
                    _persist_registry()
                    st.rerun()

            # ── Notes ───────────────────────────────────────────
            notes_val = st.text_area(
                "Notes",
                value=entry.notes,
                key=f"rb_notes_{eid}",
            )
            if notes_val != entry.notes:
                if st.button("Save notes", key=f"rb_savenotes_{eid}"):
                    update_entry_notes(reg, eid, notes_val)
                    _persist_registry()
                    st.rerun()

            # ── Linked tests ────────────────────────────────────
            st.markdown("**Linked tests:**")
            if entry.tests:
                for i, tl in enumerate(entry.tests):
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
                            "Unlink", key=f"rb_unlink_{eid}_{i}",
                        ):
                            unlink_test(reg, eid, tl.folder)
                            _persist_registry()
                            st.rerun()
            else:
                st.caption("No tests linked yet.")

            # ── Link a test ─────────────────────────────────────
            if run_names:
                already_linked = {t.folder for t in entry.tests}
                available = [n for n in run_names if n not in already_linked]
                if available:
                    with st.form(f"rb_link_{eid}", clear_on_submit=True):
                        sel_folder = st.selectbox(
                            "Link a test folder",
                            available,
                            key=f"rb_linksel_{eid}",
                        )
                        sel_type = st.selectbox(
                            "Test type",
                            ["", "sweep", "constant"],
                            key=f"rb_linktype_{eid}",
                        )
                        sel_desc = st.text_input(
                            "Description (optional)",
                            key=f"rb_linkdesc_{eid}",
                        )
                        if st.form_submit_button("Link test"):
                            link_test(
                                reg, eid,
                                TestLink(
                                    folder=sel_folder,
                                    test_type=sel_type,
                                    description=sel_desc,
                                ),
                            )
                            _persist_registry()
                            st.success(f"Linked `{sel_folder}` to {entry.display_name}")
                            st.rerun()


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — BUILD REPORT
# ════════════════════════════════════════════════════════════════════════

def _render_build_tab(run_names: list[str], run_dirs: list) -> None:
    """Report composition: select entries, choose charts, preview, export."""
    reg = _get_registry()

    if not reg.entries:
        st.info(
            "📭 Registry is empty.  Go to the **Registry** tab to add "
            "pump entries first."
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

    # ── Select entries ──────────────────────────────────────────
    st.subheader("📦 Select Entries")
    entry_ids = list(reg.entries.keys())
    entry_labels = {
        eid: f"{reg.entries[eid].display_name} ({len(reg.entries[eid].tests)} tests)"
        for eid in entry_ids
    }

    default_sel = []
    if loaded_defn:
        default_sel = [e for e in loaded_defn.entry_ids if e in entry_ids]

    selected_entries = st.multiselect(
        "Choose pump entries to include",
        entry_ids,
        default=default_sel,
        format_func=lambda eid: entry_labels.get(eid, eid),
        key="rb_entries",
    )

    # Quick select buttons
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Select all", key="rb_sel_all"):
            st.session_state["rb_entries"] = entry_ids
            st.rerun()
    with c2:
        if st.button("Clear", key="rb_sel_clr"):
            st.session_state["rb_entries"] = []
            st.rerun()

    if not selected_entries:
        st.info("👆 Select at least one entry to continue.")
        return

    # ── Comparison types ────────────────────────────────────────
    st.subheader("📊 Comparisons")
    default_comps = (
        loaded_defn.comparisons if loaded_defn
        else ["sweep_overlay", "summary_table", "boxplots"]
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
    # Collect all folders linked to the selected entries
    _all_report_folders: list[str] = []
    for _eid in selected_entries:
        _entry = reg.entries.get(_eid)
        if _entry:
            _all_report_folders.extend(
                t.folder for t in _entry.tests if t.folder in run_names
            )

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
    reg: BarRegistry,
    run_names: list[str],
    run_dirs: list,
) -> None:
    """Generate report sections, store HTML in session state, show preview."""
    ap = _get_analysis_plots()
    bcp = _get_bar_comparison_plots()

    sections: list[ReportSection] = []
    entry_metadata: dict[str, dict] = {}

    # ── Collect test folders from selected entries ──────────────
    # Build: {entry_id: [folder_names]}
    entry_tests: dict[str, list[str]] = {}
    for eid in defn.entry_ids:
        entry = reg.entries.get(eid)
        if entry is None:
            continue
        folders = [
            t.folder for t in entry.tests
            if t.folder in run_names
        ]
        entry_tests[eid] = folders
        entry_metadata[eid] = {
            "display_name": entry.display_name,
            "notes": entry.notes,
            "tests": [asdict(t) for t in entry.tests],
        }

    all_folders = [f for folders in entry_tests.values() for f in folders]
    if not all_folders:
        st.warning(
            "⚠️ None of the selected entries have test folders "
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
        entry = reg.entries[eid]
        for f in folders:
            folder_to_display[f] = f"{entry.display_name} / {f[:30]}"

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
        dname = reg.entries[eid].display_name
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
            f"{len(defn.entry_ids)} entries: "
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
                title="Frequency Sweep — Entry Comparison",
                content=fig,
                description=(
                    "Overlay of per-entry average binned sweep curves "
                    f"(bin width = {defn.bin_hz} Hz)."
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
                description="Individual test binned curves overlaid.",
            ))

    elif comp == "sweep_relative":
        if bar_binned:
            fig = bcp.plot_bar_sweep_relative(bar_binned)
            sections.append(ReportSection(
                kind="plot",
                title="Relative Sweep (0–100 %) — Entry Comparison",
                content=fig,
                description="Each entry's average normalised to 0–100%.",
            ))
        if len(display_binned) > 1:
            fig = ap.plot_relative_comparison(display_binned)
            sections.append(ReportSection(
                kind="plot",
                title="Relative Sweep (0–100 %) — All Tests",
                content=fig,
            ))

    elif comp == "individual_sweeps":
        from ..plots.plot_generator import plot_sweep_binned
        for name, binned in display_binned.items():
            fig = plot_sweep_binned(
                binned,
                title=f"{name}",
                show_error_bars=defn.show_error_bars,
            )
            sections.append(ReportSection(
                kind="plot",
                title=name,
                content=fig,
            ))

    elif comp == "global_average":
        if len(display_binned) >= 2:
            fig, avg_df = ap.plot_global_average(
                display_binned,
                show_error_bars=defn.show_error_bars,
            )
            sections.append(ReportSection(
                kind="plot",
                title="Global Average Across Tests",
                content=fig,
                description=(
                    "Inter-test average curve with ±1 std band."
                ),
            ))
            if not avg_df.empty:
                sections.append(ReportSection(
                    kind="table",
                    title="Average Curve Data",
                    content=avg_df,
                ))

    elif comp == "boxplots":
        if bar_const:
            fig = bcp.plot_bar_constant_boxplots(
                bar_const, signal_col=signal_col,
            )
            sections.append(ReportSection(
                kind="plot",
                title="Constant-Frequency Boxplots — per Entry",
                content=fig,
            ))
            fig2 = bcp.plot_bar_constant_aggregated(
                bar_const, signal_col=signal_col,
            )
            sections.append(ReportSection(
                kind="plot",
                title="Aggregated Boxplots per Entry",
                content=fig2,
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
                ))

    elif comp == "histograms":
        if bar_const:
            fig = bcp.plot_bar_constant_histograms(
                bar_const, signal_col=signal_col,
            )
            sections.append(ReportSection(
                kind="plot",
                title="Constant-Frequency Histograms — per Entry",
                content=fig,
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
                description="Raw sweep data points (capped at 50k per test).",
            ))

    elif comp == "std_vs_mean":
        if display_binned:
            fig = ap.plot_std_vs_mean(display_binned)
            sections.append(ReportSection(
                kind="plot",
                title="Std vs Mean — Variability Analysis",
                content=fig,
                description=(
                    "Each point is one frequency bin. Linear fit shows "
                    "whether noise scales with flow rate."
                ),
            ))

    elif comp == "best_region":
        if display_binned:
            fig, best_df = ap.plot_stability_cloud(display_binned)
            sections.append(ReportSection(
                kind="plot",
                title="Best Operating Region",
                content=fig,
                description=(
                    "Green stars = frequency bins with highest flow AND "
                    "lowest variability."
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
                ))

    elif comp == "correlation":
        if len(display_binned) >= 2:
            fig = ap.plot_correlation_heatmap(display_binned)
            sections.append(ReportSection(
                kind="plot",
                title="Inter-Test Correlation Heatmap",
                content=fig,
                description=(
                    "Pearson correlation between binned mean-flow curves."
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
            df = _load_csv_cached(str(pick.csv_path))
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
