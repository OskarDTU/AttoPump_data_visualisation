"""Overview page for resolved test classifications and missing metadata."""

from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from ..data.experiment_log import find_experiment_log
from ..data.io_local import list_run_dirs, normalize_root
from ..data.loader import resolve_data_path
from ..data.pump_registry import (
    PumpRegistry,
    link_test,
    migrate_legacy_files,
    save_registry,
)
from ..data.test_catalog import (
    build_test_catalog_dataframe,
    format_classification_summary,
    format_detection_method,
    rank_test_names,
    resolve_test_record,
)


@st.cache_data(ttl=300, show_spinner=False)
def _build_catalog_cached(root_str: str, run_names: tuple[str, ...]) -> pd.DataFrame:
    run_dirs = [Path(root_str) / name for name in run_names]
    return build_test_catalog_dataframe(root_str, run_dirs)


def _filter_catalog(df: pd.DataFrame, query: str, mode: str) -> pd.DataFrame:
    out = df.copy()
    if mode == "Incomplete only":
        out = out[~out["is_complete"]]
    elif mode == "Unknown type only":
        out = out[out["test_type"] == "unknown"]

    if query.strip():
        needle = query.lower().strip()
        mask = (
            out["run_name"].str.lower().str.contains(needle, na=False)
            | out["pump_bar_id"].str.lower().str.contains(needle, na=False)
            | out["classification"].str.lower().str.contains(needle, na=False)
            | out["note"].str.lower().str.contains(needle, na=False)
        )
        out = out[mask]

    return out.reset_index(drop=True)


def _get_registry() -> PumpRegistry:
    if "_pump_registry" not in st.session_state:
        st.session_state["_pump_registry"] = migrate_legacy_files()
    return st.session_state["_pump_registry"]


def _render_detail_panel(root: Path, run_map: dict[str, Path], available_names: list[str]) -> None:
    st.divider()
    st.subheader("Inspect One Test")

    query = st.text_input(
        "Search for a test to inspect",
        key="overview_detail_query",
        placeholder="Pump 3, 500Hz, 20260224...",
    )
    ranked_names = rank_test_names(available_names, query, limit=50)
    if not ranked_names:
        st.info("No matching tests.")
        return

    selected_name = st.selectbox(
        "Matching tests",
        options=ranked_names,
        key="overview_detail_select",
    )
    record = resolve_test_record(
        selected_name,
        root,
        run_dir=run_map.get(selected_name),
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Test folder:** `{record.run_name}`")
        st.markdown(f"**Classification:** {format_classification_summary(record)}")
        st.markdown(f"**Detection source:** {format_detection_method(record.detection_method)}")
        st.markdown(f"**Pump/BAR:** {record.pump_bar_id or '—'}")
        st.markdown(f"**CSV:** {record.csv_name or '—'}")
        st.markdown(f"**Issues:** {', '.join(record.issues) if record.issues else 'None'}")

    with c2:
        st.markdown(f"**Date:** {record.date or '—'}")
        st.markdown(f"**Time:** {record.time or '—'}")
        st.markdown(f"**Author:** {record.author or '—'}")
        st.markdown(f"**Voltage:** {record.voltage or '—'}")
        st.markdown(f"**Result:** {record.success or '—'}")
        if record.duration_s:
            st.markdown(f"**Duration:** {record.duration_s:g} s")
        else:
            st.markdown("**Duration:** —")

    if record.test_type == "constant" and record.frequency_hz is not None:
        st.caption(f"Constant frequency: {record.frequency_hz:g} Hz")
    elif record.test_type == "sweep":
        st.caption(
            "Sweep range: "
            f"{record.sweep_start_hz if record.sweep_start_hz is not None else '—'}"
            f" → {record.sweep_end_hz if record.sweep_end_hz is not None else '—'} Hz"
        )

    if record.note:
        with st.expander("Notes", expanded=False):
            st.write(record.note)

    # ── Action buttons ──────────────────────────────────────────
    st.divider()
    st.subheader("Actions")
    reg = _get_registry()

    # -- Assign to pump --
    pump_names = list(reg.pumps.keys())
    if pump_names:
        acol1, acol2 = st.columns([2, 1])
        with acol1:
            assign_pump = st.selectbox(
                "Assign this test to a pump",
                options=pump_names,
                index=None,
                placeholder="Select a pump…",
                key="overview_assign_pump",
            )
        with acol2:
            st.write("")  # spacer
            st.write("")
            if st.button("Assign", key="overview_assign_btn", disabled=assign_pump is None):
                from ..data.pump_registry import TestLink

                link_test(reg, assign_pump, TestLink(folder=selected_name))
                save_registry(reg)
                st.success(f"Linked **{selected_name}** → **{assign_pump}**")
                st.rerun()
    else:
        st.info("No pumps registered yet. Add pumps on the **Manage Groups** page.")

    # -- Set / override test type --
    from ..data.test_configs import TestConfig, save_test_config

    type_options = ["sweep", "constant", "unknown"]
    current_type = record.test_type or "unknown"
    new_type = st.selectbox(
        "Override test type",
        options=type_options,
        index=type_options.index(current_type) if current_type in type_options else 2,
        key="overview_set_type",
    )
    if new_type != "unknown" and st.button("Save test type", key="overview_save_type_btn"):
        save_test_config(selected_name, TestConfig(test_type=new_type))
        st.success(f"Saved **{selected_name}** as **{new_type}**")
        _build_catalog_cached.clear()
        st.rerun()

    # -- Quick navigation --
    st.caption("Quick links")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if st.button("🔍 Open in Explorer", key="overview_goto_explorer"):
            st.session_state["explorer_preselect"] = selected_name
            st.switch_page("app/pages/explorer.py")
    with bcol2:
        if st.button("📊 Open in Analysis", key="overview_goto_analysis"):
            st.session_state["analysis_preselect"] = selected_name
            st.switch_page("app/pages/analysis.py")


def main() -> None:
    """Entry point for the Test Overview page."""
    try:
        st.title("🗂️ Test Overview")
        st.caption(
            "Resolved classifications and metadata across every discovered "
            "test folder, including gaps that still need attention."
        )

        data_folder_str, run_dirs, run_names_list = resolve_data_path(
            key_suffix="to",
            render_widget=False,
        )

        if not data_folder_str.strip():
            st.info("Enter the All_tests folder path in the sidebar.")
            return

        root = normalize_root(data_folder_str)

        if not run_dirs:
            st.warning("No test folders were found under the selected path.")
            return

        run_names = tuple(p.name for p in run_dirs)
        run_map = {p.name: p for p in run_dirs}
        log_path = find_experiment_log(root)
        if log_path is None:
            st.warning(
                "Experiment log not found relative to this data folder. "
                "The overview will still use saved configs, metadata, and folder-name parsing."
            )
        else:
            st.caption(f"Experiment log: `{log_path}`")

        with st.spinner(f"Resolving metadata for {len(run_dirs)} test folders…"):
            catalog_df = _build_catalog_cached(str(root), run_names)

        if catalog_df.empty:
            st.warning("No tests could be resolved.")
            return

        total_tests = int(len(catalog_df))
        incomplete_tests = int((~catalog_df["is_complete"]).sum())
        unknown_type_tests = int((catalog_df["test_type"] == "unknown").sum())
        log_matched_tests = int(catalog_df["log_found"].sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tests", total_tests)
        m2.metric("Complete", total_tests - incomplete_tests)
        m3.metric("Needs Attention", incomplete_tests)
        m4.metric("Log Matches", log_matched_tests)

        if unknown_type_tests:
            st.warning(f"{unknown_type_tests} test(s) still have an unknown type.")

        filter_col, mode_col = st.columns([2, 1])
        with filter_col:
            query = st.text_input(
                "Filter tests",
                placeholder="Search by folder name, pump, classification, or notes",
            )
        with mode_col:
            mode = st.selectbox(
                "View",
                ["All tests", "Incomplete only", "Unknown type only"],
                index=0,
            )

        filtered_df = _filter_catalog(catalog_df, query, mode)

        tab_all, tab_attention = st.tabs(["All Tests", "Needs Attention"])

        with tab_all:
            st.dataframe(
                filtered_df[
                    [
                        "run_name",
                        "pump_bar_id",
                        "classification",
                        "detection_source",
                        "date",
                        "voltage",
                        "success",
                        "issues",
                    ]
                ].rename(
                    columns={
                        "run_name": "Test folder",
                        "pump_bar_id": "Pump/BAR",
                        "classification": "Classification",
                        "detection_source": "Detected via",
                        "date": "Date",
                        "voltage": "Voltage",
                        "success": "Result",
                        "issues": "Missing / issues",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=500,
            )

        with tab_attention:
            attention_df = catalog_df[~catalog_df["is_complete"]].copy()
            if attention_df.empty:
                st.success("No unresolved metadata gaps.")
            else:
                issue_counts = attention_df["issues"].value_counts().reset_index()
                issue_counts.columns = ["Issue", "Count"]
                top1, top2 = st.columns([2, 3])
                with top1:
                    st.dataframe(issue_counts, use_container_width=True, hide_index=True)
                with top2:
                    st.dataframe(
                        attention_df[
                            [
                                "run_name",
                                "pump_bar_id",
                                "classification",
                                "issues",
                            ]
                        ].rename(
                            columns={
                                "run_name": "Test folder",
                                "pump_bar_id": "Pump/BAR",
                                "classification": "Classification",
                                "issues": "Missing / issues",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                        height=400,
                    )

        _render_detail_panel(root, run_map, list(run_map))

    except Exception as exc:
        st.error(f"❌ **CRITICAL ERROR:** {exc}")
        with st.expander("🔍 Debug"):
            st.code(traceback.format_exc())
