"""Manage Groups page — CRUD for pumps, sub-groups, shipments, test groups & analysis configs.

Uses the unified ``PumpRegistry`` (``pump_registry.py``) as its single
data store, replacing the previous split between bar_groups and bar_registry.
"""

from __future__ import annotations

import traceback

import streamlit as st

from ..data.io_local import normalize_root
from ..data.loader import resolve_data_path
from ..data.pump_registry import (
    AnalysisConfig,
    AVAILABLE_PLOTS,
    Pump,
    PumpRegistry,
    PumpSubGroup,
    Shipment,
    TestGroup,
    TestLink,
    add_analysis_config,
    add_pump,
    add_shipment,
    add_sub_group,
    add_test_group,
    migrate_legacy_files,
    remove_analysis_config,
    remove_pump,
    remove_shipment,
    remove_sub_group,
    remove_test_group,
    save_registry,
    sync_pumps_from_experiment_log,
    upsert_analysis_config,
    upsert_sub_group,
)


# ── Session helpers ────────────────────────────────────────────────────

def _get_registry() -> PumpRegistry:
    key = "_pump_registry"
    if key not in st.session_state:
        st.session_state[key] = migrate_legacy_files()
    return st.session_state[key]


def _persist() -> None:
    save_registry(_get_registry())


def _auto_sync_registry(
    reg: PumpRegistry,
    data_folder_str: str,
    run_names: list[str],
) -> dict[str, list[str]]:
    """Auto-sync pump mappings from the experiment log once per data root."""
    if not data_folder_str.strip():
        return {}

    try:
        root = normalize_root(data_folder_str)
    except Exception:
        return {}

    sync_key = f"_manage_groups_auto_sync::{root}"
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


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Entry point for the Manage Groups page."""
    try:
        st.title("🛠️ Manage Groups")

        data_folder_str, run_dirs, run_names = resolve_data_path(
            key_suffix="mg",
            render_widget=False,
        )

        # ── Tabs ────────────────────────────────────────────────
        tab_pumps, tab_ships, tab_tgroups, tab_configs = st.tabs([
            "🔧 Pumps",
            "🚚 Shipments",
            "📋 Test Groups",
            "⚙️ Analysis Configs",
        ])

        reg = _get_registry()
        auto_sync_changes = _auto_sync_registry(reg, data_folder_str, run_names)

        with tab_pumps:
            _render_pump_sync(reg, data_folder_str, run_names, auto_sync_changes)
            _render_pumps(reg, run_names)

        with tab_ships:
            _render_shipments(reg)

        with tab_tgroups:
            _render_test_groups(reg, run_names)

        with tab_configs:
            _render_analysis_configs(reg)

    except Exception as e:
        st.error(f"❌ **CRITICAL ERROR:** {e}")
        with st.expander("🔍 Debug"):
            st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════
# PUMP SYNC FROM EXPERIMENT LOG
# ════════════════════════════════════════════════════════════════════════

def _render_pump_sync(
    reg: PumpRegistry,
    data_folder_str: str,
    run_names: list[str],
    auto_sync_changes: dict[str, list[str]],
) -> None:
    """Render the 'Sync from Experiment Log' section."""
    st.subheader("🔄 Sync Pumps from Experiment Log")
    st.caption(
        "Automatically link test folders to pumps using the "
        "**Pump/BAR ID** column in the experiment log spreadsheet. "
        "This page auto-syncs once when a data folder is loaded, and you can "
        "re-run it manually below."
    )

    if auto_sync_changes:
        st.success(
            "Auto-synced pump links for: "
            + ", ".join(
                f"{pump} ({len(folders)} test(s))"
                for pump, folders in auto_sync_changes.items()
            )
        )

    if not data_folder_str.strip():
        st.info("Enter a data folder path in the sidebar to enable sync.")
        return

    try:
        root = normalize_root(data_folder_str)
    except Exception as e:
        st.error(f"Invalid data path: {e}")
        return

    if st.button("🔄 Sync from Experiment Log", key="sync_exp_log"):
        try:
            _, changes = sync_pumps_from_experiment_log(
                reg,
                root,
                available_folders=run_names if run_names else None,
            )
            _persist()
            if changes:
                parts = [
                    f"**{pump}**: {len(folders)} test(s) linked"
                    for pump, folders in changes.items()
                ]
                st.success(
                    "Synced from experiment log:\n\n"
                    + "\n\n".join(parts)
                )
            else:
                st.info("Already up-to-date — no new associations found.")
            st.rerun()
        except Exception as e:
            st.error(f"Sync failed: {e}")
            with st.expander("Debug"):
                st.code(traceback.format_exc())

    st.markdown("---")


# ════════════════════════════════════════════════════════════════════════
# PUMPS
# ════════════════════════════════════════════════════════════════════════

def _render_pumps(reg: PumpRegistry, run_names: list[str]) -> None:
    st.subheader("🔧 Pumps")
    st.caption("A **pump** groups test folders belonging to the same physical pump.")

    if reg.pumps:
        for pname, pump in list(reg.pumps.items()):
            with st.expander(
                f"**{pname}** — {len(pump.tests)} test(s), "
                f"{len(pump.sub_groups)} sub-group(s)",
                expanded=False,
            ):
                st.markdown(
                    f"*{pump.description}*"
                    if pump.description
                    else "_No description._"
                )
                if pump.notes:
                    st.caption(pump.notes)

                # ── Linked tests ────────────────────────────────
                for t in pump.tests:
                    badge = ""
                    if t.test_type == "sweep":
                        badge = " 🔵"
                    elif t.test_type == "constant":
                        badge = " 🟡"
                    if t.success is True:
                        badge += " ✅"
                    elif t.success is False:
                        badge += " ❌"
                    avail = " ✓" if t.folder in run_names else " ⚠️"
                    st.markdown(f"- `{t.folder}`{badge}{avail}")

                if run_names:
                    # Edit tests
                    current_folders = [t.folder for t in pump.tests]
                    new_tests = st.multiselect(
                        "Tests",
                        run_names,
                        default=[f for f in current_folders if f in run_names],
                        key=f"pe_{pname}",
                    )
                    new_desc = st.text_input(
                        "Description", pump.description, key=f"pd_{pname}",
                    )
                    new_notes = st.text_area(
                        "Notes", pump.notes, key=f"pn_{pname}", height=68,
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("💾 Save", key=f"ps_{pname}"):
                            # Preserve TestLink metadata for existing tests
                            folder_to_link = {t.folder: t for t in pump.tests}
                            pump.tests = [
                                folder_to_link.get(f, TestLink(folder=f))
                                for f in new_tests
                            ]
                            pump.description = new_desc
                            pump.notes = new_notes
                            _persist()
                            st.success(f"Updated **{pname}**")
                            st.rerun()
                    with c2:
                        if st.button("🗑️ Delete", key=f"px_{pname}", type="secondary"):
                            remove_pump(reg, pname)
                            _persist()
                            st.rerun()
                else:
                    st.info("Enter a data folder path in the sidebar to enable test selection.")

                # ── Sub-groups ──────────────────────────────────
                st.markdown("---")
                st.markdown("**Sub-groups** (named subsets of this pump's tests)")

                if pump.sub_groups:
                    for sg_name, sg in list(pump.sub_groups.items()):
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            new_sg_name = st.text_input(
                                "Group name",
                                sg.name,
                                key=f"sgn_edit_{pname}_{sg_name}",
                            )
                            new_sg_desc = st.text_input(
                                "Description",
                                sg.description,
                                key=f"sgd_edit_{pname}_{sg_name}",
                            )
                            available_for_sg = [t.folder for t in pump.tests]
                            new_sg_tests = st.multiselect(
                                f"📂 {sg_name}",
                                available_for_sg,
                                default=[t for t in sg.tests if t in available_for_sg],
                                key=f"sg_{pname}_{sg_name}",
                            )
                        with sc2:
                            save_col, delete_col = st.columns(2)
                        with save_col:
                            if st.button("💾 Save", key=f"sgs_{pname}_{sg_name}"):
                                proposed_name = new_sg_name.strip()
                                if not proposed_name:
                                    st.error("Sub-group name required.")
                                elif (
                                    proposed_name != sg_name
                                    and proposed_name in pump.sub_groups
                                ):
                                    st.error(
                                        f"A sub-group named **{proposed_name}** already exists."
                                    )
                                else:
                                    upsert_sub_group(
                                        reg,
                                        pname,
                                        PumpSubGroup(
                                            proposed_name,
                                            new_sg_tests,
                                            new_sg_desc,
                                        ),
                                        previous_name=sg_name,
                                    )
                                    _persist()
                                    st.rerun()
                        with delete_col:
                            if st.button("🗑️", key=f"sgx_{pname}_{sg_name}"):
                                remove_sub_group(reg, pname, sg_name)
                                _persist()
                                st.rerun()
                else:
                    st.caption("No sub-groups yet.")

                # Add new sub-group
                with st.form(f"new_sg_{pname}", clear_on_submit=True):
                    sg_name_new = st.text_input(
                        "New sub-group name",
                        placeholder="e.g. Sweep tests",
                        key=f"sgn_{pname}",
                    )
                    available_for_new_sg = [t.folder for t in pump.tests]
                    sg_tests_new = st.multiselect(
                        "Tests in group",
                        available_for_new_sg,
                        key=f"sgt_{pname}",
                    )
                    sg_desc_new = st.text_input(
                        "Description (optional)", key=f"sgd_{pname}",
                    )
                    if st.form_submit_button("➕ Create Sub-Group"):
                        if sg_name_new.strip():
                            add_sub_group(
                                reg, pname,
                                PumpSubGroup(sg_name_new.strip(), sg_tests_new, sg_desc_new),
                            )
                            _persist()
                            st.success(f"Created sub-group **{sg_name_new}**")
                            st.rerun()
                        else:
                            st.error("Name required.")
    else:
        st.info("No pumps yet. Sync from the experiment log or create one below.")

    st.markdown("---")
    st.markdown("#### ➕ New Pump")
    with st.form("new_pump", clear_on_submit=True):
        np_name = st.text_input("Name", placeholder="e.g. Pump 270226-1")
        np_desc = st.text_input("Description", placeholder="e.g. First prototype")
        np_notes = st.text_area("Notes (optional)", height=68)
        np_tests = (
            st.multiselect("Tests", run_names, key="np_tests")
            if run_names
            else []
        )
        if not run_names:
            st.info("Enter a data folder path in the sidebar to assign tests.")
        if st.form_submit_button("Create"):
            if not np_name.strip():
                st.error("Name required.")
            elif np_name in reg.pumps:
                st.error(f"A pump named **{np_name}** already exists.")
            else:
                add_pump(
                    reg,
                    Pump(np_name, [TestLink(folder=f) for f in np_tests], np_desc, np_notes),
                )
                _persist()
                st.success(f"Created **{np_name}** with {len(np_tests)} test(s).")
                st.rerun()


# ════════════════════════════════════════════════════════════════════════
# SHIPMENTS
# ════════════════════════════════════════════════════════════════════════

def _render_shipments(reg: PumpRegistry) -> None:
    st.subheader("🚚 Shipments")
    st.caption("A **shipment** groups pumps under a recipient label.")
    pump_list = list(reg.pumps.keys())

    if reg.shipments:
        for sname, ship in list(reg.shipments.items()):
            with st.expander(
                f"**{sname}** — {ship.recipient or '—'} "
                f"— {len(ship.pumps)} pump(s)",
                expanded=False,
            ):
                st.markdown(f"**Recipient:** {ship.recipient or '—'}")
                st.markdown(f"**Description:** {ship.description or '—'}")
                for p in ship.pumps:
                    n = len(reg.pumps[p].tests) if p in reg.pumps else "?"
                    st.markdown(f"- **{p}** ({n} tests)")

                if pump_list:
                    new_pumps = st.multiselect(
                        "Pumps",
                        pump_list,
                        default=[p for p in ship.pumps if p in pump_list],
                        key=f"se_{sname}",
                    )
                    new_recip = st.text_input(
                        "Recipient", ship.recipient, key=f"sr_{sname}",
                    )
                    new_desc = st.text_input(
                        "Description", ship.description, key=f"sd_{sname}",
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("💾 Save", key=f"ss_{sname}"):
                            ship.pumps = new_pumps
                            ship.recipient = new_recip
                            ship.description = new_desc
                            _persist()
                            st.success(f"Updated **{sname}**")
                            st.rerun()
                    with c2:
                        if st.button(
                            "🗑️ Delete", key=f"sx_{sname}", type="secondary",
                        ):
                            remove_shipment(reg, sname)
                            _persist()
                            st.rerun()
    else:
        st.info("No shipments yet.")

    st.markdown("---")
    st.markdown("#### ➕ New Shipment")
    with st.form("new_ship", clear_on_submit=True):
        ns_name = st.text_input("Name", placeholder="e.g. Niels' pumps")
        ns_recip = st.text_input("Recipient", placeholder="e.g. Niels")
        ns_desc = st.text_input("Description")
        ns_pumps = (
            st.multiselect("Pumps", pump_list, key="ns_pumps")
            if pump_list
            else []
        )
        if not pump_list:
            st.info("Create pumps first before making a shipment.")
        if st.form_submit_button("Create"):
            if not ns_name.strip():
                st.error("Name required.")
            elif ns_name in reg.shipments:
                st.error(f"Already exists: **{ns_name}**")
            else:
                add_shipment(
                    reg, Shipment(ns_name, ns_pumps, ns_recip, ns_desc),
                )
                _persist()
                st.success(f"Created shipment **{ns_name}**.")
                st.rerun()


# ════════════════════════════════════════════════════════════════════════
# TEST GROUPS
# ════════════════════════════════════════════════════════════════════════

def _render_test_groups(reg: PumpRegistry, run_names: list[str]) -> None:
    st.subheader("📋 Test Groups")
    st.caption(
        "A **test group** is a saved collection of test folders "
        "for quick re-comparison."
    )

    if reg.test_groups:
        for gname, grp in list(reg.test_groups.items()):
            with st.expander(
                f"**{gname}** — {len(grp.tests)} test(s)",
                expanded=False,
            ):
                st.markdown(
                    f"*{grp.description}*"
                    if grp.description
                    else "_No description._"
                )
                for t in grp.tests:
                    st.markdown(f"- `{t}`")

                if run_names:
                    new_tests = st.multiselect(
                        "Tests",
                        run_names,
                        default=[t for t in grp.tests if t in run_names],
                        key=f"ge_{gname}",
                    )
                    new_desc = st.text_input(
                        "Description", grp.description, key=f"gd_{gname}",
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("💾 Save", key=f"gs_{gname}"):
                            grp.tests = new_tests
                            grp.description = new_desc
                            _persist()
                            st.success(f"Updated **{gname}**")
                            st.rerun()
                    with c2:
                        if st.button(
                            "🗑️ Delete", key=f"gx_{gname}", type="secondary",
                        ):
                            remove_test_group(reg, gname)
                            _persist()
                            st.rerun()
                else:
                    st.info(
                        "Enter a data folder path in the sidebar "
                        "to edit test assignments."
                    )
    else:
        st.info("No test groups yet.")

    st.markdown("---")
    st.markdown("#### ➕ New Test Group")
    with st.form("new_tgrp", clear_on_submit=True):
        ng_name = st.text_input("Name", placeholder="e.g. March sweep tests")
        ng_desc = st.text_input("Description")
        ng_tests = (
            st.multiselect("Tests", run_names, key="ng_tests")
            if run_names
            else []
        )
        if not run_names:
            st.info("Enter a data folder path in the sidebar to assign tests.")
        if st.form_submit_button("Create"):
            if not ng_name.strip():
                st.error("Name required.")
            elif ng_name in reg.test_groups:
                st.error(f"Already exists: **{ng_name}**")
            else:
                add_test_group(
                    reg, TestGroup(ng_name, ng_tests, ng_desc),
                )
                _persist()
                st.success(
                    f"Created **{ng_name}** with {len(ng_tests)} test(s)."
                )
                st.rerun()


# ════════════════════════════════════════════════════════════════════════
# ANALYSIS CONFIGS
# ════════════════════════════════════════════════════════════════════════

def _render_analysis_configs(reg: PumpRegistry) -> None:
    st.subheader("⚙️ Analysis Configurations")
    st.caption(
        "Save reusable analysis presets — choose which plots to generate, "
        "bin widths, appearance settings, etc."
    )

    if reg.analysis_configs:
        for cname, cfg in list(reg.analysis_configs.items()):
            with st.expander(
                f"**{cname}** — {len(cfg.plots)} plot(s)", expanded=False,
            ):
                st.markdown(
                    f"*{cfg.description}*" if cfg.description else "_No description._"
                )
                st.markdown(
                    f"Bin: {cfg.bin_hz} Hz · "
                    f"Avg bin: {cfg.avg_bin_hz} Hz · "
                    f"Freq tol: {cfg.freq_tol} Hz · "
                    f"Error bars: {'✓' if cfg.show_error_bars else '✗'} · "
                    f"All points: {'✓' if cfg.show_all_data_points else '✗'} · "
                    f"Mode: {cfg.plot_mode}"
                )
                for p in cfg.plots:
                    st.markdown(f"- {AVAILABLE_PLOTS.get(p, p)}")

                edit_name = st.text_input(
                    "Config name",
                    cfg.name,
                    key=f"cfg_name_{cname}",
                )
                edit_desc = st.text_input(
                    "Description",
                    cfg.description,
                    key=f"cfg_desc_{cname}",
                )
                edit_plots = st.multiselect(
                    "Plots",
                    list(AVAILABLE_PLOTS.keys()),
                    default=[p for p in cfg.plots if p in AVAILABLE_PLOTS],
                    format_func=lambda k: AVAILABLE_PLOTS[k],
                    key=f"cfg_plots_{cname}",
                )
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    edit_bin = st.number_input(
                        "Bin width (Hz)",
                        value=float(cfg.bin_hz),
                        min_value=0.5,
                        step=0.5,
                        key=f"cfg_bin_{cname}",
                    )
                with ec2:
                    edit_err = st.checkbox(
                        "Error bars",
                        value=cfg.show_error_bars,
                        key=f"cfg_err_{cname}",
                    )
                with ec3:
                    edit_all = st.checkbox(
                        "All data points",
                        value=cfg.show_all_data_points,
                        key=f"cfg_all_{cname}",
                    )
                ec6, ec7 = st.columns(2)
                with ec6:
                    edit_avg_bin = st.number_input(
                        "Average-curve bin width (Hz)",
                        value=float(cfg.avg_bin_hz),
                        min_value=0.5,
                        step=0.5,
                        key=f"cfg_avg_bin_{cname}",
                    )
                with ec7:
                    edit_freq_tol = st.number_input(
                        "Freq-match tolerance (Hz)",
                        value=float(cfg.freq_tol),
                        min_value=0.5,
                        step=0.5,
                        key=f"cfg_freq_tol_{cname}",
                    )
                edit_max_raw = st.number_input(
                    "Max raw points when capped",
                    value=int(cfg.max_raw_points),
                    min_value=1000,
                    step=1000,
                    key=f"cfg_maxraw_{cname}",
                )
                edit_mode = st.selectbox(
                    "Plot mode",
                    ["lines+markers", "lines", "markers"],
                    index=["lines+markers", "lines", "markers"].index(cfg.plot_mode)
                    if cfg.plot_mode in ("lines+markers", "lines", "markers")
                    else 0,
                    key=f"cfg_mode_{cname}",
                )
                ec4, ec5 = st.columns(2)
                with ec4:
                    edit_marker = st.slider(
                        "Marker size",
                        1,
                        20,
                        int(cfg.marker_size),
                        key=f"cfg_marker_{cname}",
                    )
                with ec5:
                    edit_opacity = st.slider(
                        "Opacity",
                        0.1,
                        1.0,
                        float(cfg.opacity),
                        0.05,
                        key=f"cfg_opacity_{cname}",
                    )
                ec8, ec9 = st.columns(2)
                with ec8:
                    edit_mean_pct = st.slider(
                        "High-flow %",
                        50,
                        99,
                        int(cfg.mean_threshold_pct),
                        key=f"cfg_mean_pct_{cname}",
                    )
                with ec9:
                    edit_std_pct = st.slider(
                        "Stability %",
                        1,
                        50,
                        int(cfg.std_threshold_pct),
                        key=f"cfg_std_pct_{cname}",
                    )

                save_col, delete_col = st.columns(2)
                with save_col:
                    if st.button("💾 Save", key=f"cs_{cname}"):
                        proposed_name = edit_name.strip()
                        if not proposed_name:
                            st.error("Config name required.")
                        elif (
                            proposed_name != cname
                            and proposed_name in reg.analysis_configs
                        ):
                            st.error(
                                f"A config named **{proposed_name}** already exists."
                            )
                        else:
                            upsert_analysis_config(
                                reg,
                                AnalysisConfig(
                                    name=proposed_name,
                                    description=edit_desc,
                                    plots=edit_plots,
                                    bin_hz=float(edit_bin),
                                    avg_bin_hz=float(edit_avg_bin),
                                    freq_tol=float(edit_freq_tol),
                                    show_error_bars=edit_err,
                                    show_all_data_points=edit_all,
                                    max_raw_points=int(edit_max_raw),
                                    plot_mode=edit_mode,
                                    marker_size=int(edit_marker),
                                    opacity=float(edit_opacity),
                                    mean_threshold_pct=int(edit_mean_pct),
                                    std_threshold_pct=int(edit_std_pct),
                                ),
                                previous_name=cname,
                            )
                            _persist()
                            st.rerun()
                with delete_col:
                    if st.button("🗑️ Delete", key=f"cx_{cname}", type="secondary"):
                        remove_analysis_config(reg, cname)
                        _persist()
                        st.rerun()
    else:
        st.info("No saved analysis configs yet.")

    st.markdown("---")
    st.markdown("#### ➕ New Analysis Config")
    with st.form("new_config", clear_on_submit=True):
        nc_name = st.text_input("Config name", placeholder="e.g. Standard sweep analysis")
        nc_desc = st.text_input("Description")
        nc_plots = st.multiselect(
            "Plots to include",
            list(AVAILABLE_PLOTS.keys()),
            default=["sweep_overlay", "boxplots", "summary_table"],
            format_func=lambda k: AVAILABLE_PLOTS[k],
            key="nc_plots",
        )
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            nc_bin = st.number_input("Bin width (Hz)", value=5.0, min_value=0.5, step=0.5)
        with nc2:
            nc_err = st.checkbox("Error bars", value=True)
        with nc3:
            nc_all = st.checkbox("All data points", value=True)
        nc6, nc7 = st.columns(2)
        with nc6:
            nc_avg_bin = st.number_input(
                "Average-curve bin width (Hz)",
                value=3.0,
                min_value=0.5,
                step=0.5,
            )
        with nc7:
            nc_freq_tol = st.number_input(
                "Freq-match tolerance (Hz)",
                value=5.0,
                min_value=0.5,
                step=0.5,
            )
        nc_mode = st.selectbox(
            "Plot mode",
            ["lines+markers", "lines", "markers"],
            key="nc_mode",
        )
        nc4, nc5 = st.columns(2)
        with nc4:
            nc_msz = st.slider("Marker size", 1, 20, 6, key="nc_msz")
        with nc5:
            nc_opa = st.slider("Opacity", 0.1, 1.0, 0.8, 0.05, key="nc_opa")
        nc_max_raw = st.number_input(
            "Max raw points when capped",
            value=500000,
            min_value=1000,
            step=1000,
            key="nc_max_raw",
        )
        nc8, nc9 = st.columns(2)
        with nc8:
            nc_mean_pct = st.slider("High-flow %", 50, 99, 75, key="nc_mean_pct")
        with nc9:
            nc_std_pct = st.slider("Stability %", 1, 50, 10, key="nc_std_pct")

        if st.form_submit_button("Create"):
            if not nc_name.strip():
                st.error("Name required.")
            elif nc_name in reg.analysis_configs:
                st.error(f"Already exists: **{nc_name}**")
            else:
                add_analysis_config(
                    reg,
                    AnalysisConfig(
                        name=nc_name,
                        description=nc_desc,
                        plots=nc_plots,
                        bin_hz=nc_bin,
                        avg_bin_hz=nc_avg_bin,
                        freq_tol=nc_freq_tol,
                        show_error_bars=nc_err,
                        show_all_data_points=nc_all,
                        max_raw_points=int(nc_max_raw),
                        plot_mode=nc_mode,
                        marker_size=nc_msz,
                        opacity=nc_opa,
                        mean_threshold_pct=int(nc_mean_pct),
                        std_threshold_pct=int(nc_std_pct),
                    ),
                )
                _persist()
                st.success(f"Created config **{nc_name}**.")
                st.rerun()
