"""Manage Groups page — CRUD for pumps, shipments, and test groups.

This standalone page lets the user create, edit, and delete:
- **Pumps** — a named group of test folders belonging to one physical pump.
- **Shipments** — groups of pumps under a recipient label.
- **Test Groups** — arbitrary saved collections of test folders for quick
  re-comparison.

Persistence is handled by ``bar_groups.json`` (via ``app.data.bar_groups``).
"""

from __future__ import annotations

import traceback

import streamlit as st

from ..data.io_local import list_run_dirs, normalize_root
from ..data.bar_groups import (
    Bar,
    BarGroupsStore,
    Shipment,
    TestGroup,
    add_bar,
    add_shipment,
    add_test_group,
    load_bar_groups,
    remove_bar,
    remove_shipment,
    remove_test_group,
    save_bar_groups,
)


# ── Session helpers ────────────────────────────────────────────────────

def _get_store() -> BarGroupsStore:
    key = "_bar_groups_store"
    if key not in st.session_state:
        st.session_state[key] = load_bar_groups()
    return st.session_state[key]


def _persist() -> None:
    save_bar_groups(_get_store())


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Entry point for the Manage Groups page."""
    try:
        st.title("🛠️ Manage Groups")

        # ── Sidebar — Data Source ───────────────────────────────
        with st.sidebar:
            st.header("📁 Data Source")
            if "last_data_path" not in st.session_state:
                st.session_state.last_data_path = ""
            data_folder_str = st.text_input(
                "Path to test data folder",
                value=st.session_state.last_data_path,
                placeholder="/Users/.../All_tests",
                key="mg_data_path",
            )

        # ── Resolve data path ──────────────────────────────────
        run_names: list[str] = []
        if data_folder_str.strip():
            try:
                root = normalize_root(data_folder_str)
                run_dirs = list(list_run_dirs(root))
                run_names = [p.name for p in run_dirs]
            except Exception as e:
                st.sidebar.error(f"❌ Invalid path: {e}")

        # ── Tabs ────────────────────────────────────────────────
        tab_pumps, tab_ships, tab_tgroups = st.tabs([
            "🔧 Pumps",
            "🚚 Shipments",
            "📋 Test Groups",
        ])

        store = _get_store()

        with tab_pumps:
            _render_pumps(store, run_names)

        with tab_ships:
            _render_shipments(store)

        with tab_tgroups:
            _render_test_groups(store, run_names)

    except Exception as e:
        st.error(f"❌ **CRITICAL ERROR:** {e}")
        with st.expander("🔍 Debug"):
            st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════
# PUMPS
# ════════════════════════════════════════════════════════════════════════

def _render_pumps(store: BarGroupsStore, run_names: list[str]) -> None:
    st.subheader("🔧 Pumps")
    st.caption(
        "A **pump** groups test folders belonging to the same physical pump."
    )

    if store.bars:
        for bname, bar in list(store.bars.items()):
            with st.expander(
                f"**{bname}** — {len(bar.tests)} test(s)",
                expanded=False,
            ):
                st.markdown(
                    f"*{bar.description}*"
                    if bar.description
                    else "_No description._"
                )
                for t in bar.tests:
                    st.markdown(f"- `{t}`")

                if run_names:
                    new_tests = st.multiselect(
                        "Tests",
                        run_names,
                        default=[t for t in bar.tests if t in run_names],
                        key=f"be_{bname}",
                    )
                    new_desc = st.text_input(
                        "Description",
                        bar.description,
                        key=f"bd_{bname}",
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("💾 Save", key=f"bs_{bname}"):
                            bar.tests = new_tests
                            bar.description = new_desc
                            _persist()
                            st.success(f"Updated **{bname}**")
                            st.rerun()
                    with c2:
                        if st.button(
                            "🗑️ Delete",
                            key=f"bx_{bname}",
                            type="secondary",
                        ):
                            remove_bar(store, bname)
                            _persist()
                            st.rerun()
                else:
                    st.info(
                        "Enter a data folder path in the sidebar "
                        "to enable test selection."
                    )
    else:
        st.info("No pumps yet.")

    st.markdown("---")
    st.markdown("#### ➕ New Pump")
    with st.form("new_bar", clear_on_submit=True):
        nb_name = st.text_input("Name", placeholder="e.g. Pump 1")
        nb_desc = st.text_input(
            "Description", placeholder="e.g. Serial #1234",
        )
        nb_tests = (
            st.multiselect("Tests", run_names, key="nb_tests")
            if run_names
            else []
        )
        if not run_names:
            st.info(
                "Enter a data folder path in the sidebar to assign tests."
            )
        if st.form_submit_button("Create"):
            if not nb_name.strip():
                st.error("Name required.")
            elif nb_name in store.bars:
                st.error(f"A pump named **{nb_name}** already exists.")
            else:
                add_bar(store, Bar(nb_name, nb_tests, nb_desc))
                _persist()
                st.success(
                    f"Created **{nb_name}** with {len(nb_tests)} test(s)."
                )
                st.rerun()


# ════════════════════════════════════════════════════════════════════════
# SHIPMENTS
# ════════════════════════════════════════════════════════════════════════

def _render_shipments(store: BarGroupsStore) -> None:
    st.subheader("🚚 Shipments")
    st.caption(
        "A **shipment** groups pumps under a recipient label."
    )
    pump_list = list(store.bars.keys())

    if store.shipments:
        for sname, ship in list(store.shipments.items()):
            with st.expander(
                f"**{sname}** — {ship.recipient or '—'} "
                f"— {len(ship.bars)} pump(s)",
                expanded=False,
            ):
                st.markdown(f"**Recipient:** {ship.recipient or '—'}")
                st.markdown(
                    f"**Description:** {ship.description or '—'}"
                )
                for b in ship.bars:
                    n = (
                        len(store.bars[b].tests)
                        if b in store.bars
                        else "?"
                    )
                    st.markdown(f"- **{b}** ({n} tests)")

                if pump_list:
                    new_pumps = st.multiselect(
                        "Pumps",
                        pump_list,
                        default=[
                            b for b in ship.bars if b in pump_list
                        ],
                        key=f"se_{sname}",
                    )
                    new_recip = st.text_input(
                        "Recipient",
                        ship.recipient,
                        key=f"sr_{sname}",
                    )
                    new_desc = st.text_input(
                        "Description",
                        ship.description,
                        key=f"sd_{sname}",
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("💾 Save", key=f"ss_{sname}"):
                            ship.bars = new_pumps
                            ship.recipient = new_recip
                            ship.description = new_desc
                            _persist()
                            st.success(f"Updated **{sname}**")
                            st.rerun()
                    with c2:
                        if st.button(
                            "🗑️ Delete",
                            key=f"sx_{sname}",
                            type="secondary",
                        ):
                            remove_shipment(store, sname)
                            _persist()
                            st.rerun()
    else:
        st.info("No shipments yet.")

    st.markdown("---")
    st.markdown("#### ➕ New Shipment")
    with st.form("new_ship", clear_on_submit=True):
        ns_name = st.text_input(
            "Name", placeholder="e.g. Niels' pumps",
        )
        ns_recip = st.text_input(
            "Recipient", placeholder="e.g. Niels",
        )
        ns_desc = st.text_input("Description")
        ns_pumps = (
            st.multiselect("Pumps", pump_list, key="ns_bars")
            if pump_list
            else []
        )
        if not pump_list:
            st.info("Create pumps first before making a shipment.")
        if st.form_submit_button("Create"):
            if not ns_name.strip():
                st.error("Name required.")
            elif ns_name in store.shipments:
                st.error(
                    f"A shipment named **{ns_name}** already exists."
                )
            else:
                add_shipment(
                    store,
                    Shipment(ns_name, ns_pumps, ns_recip, ns_desc),
                )
                _persist()
                st.success(f"Created shipment **{ns_name}**.")
                st.rerun()


# ════════════════════════════════════════════════════════════════════════
# TEST GROUPS
# ════════════════════════════════════════════════════════════════════════

def _render_test_groups(store: BarGroupsStore, run_names: list[str]) -> None:
    st.subheader("📋 Test Groups")
    st.caption(
        "A **test group** is a saved collection of test folders "
        "for quick re-comparison."
    )

    if store.test_groups:
        for gname, grp in list(store.test_groups.items()):
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
                        default=[
                            t for t in grp.tests if t in run_names
                        ],
                        key=f"ge_{gname}",
                    )
                    new_desc = st.text_input(
                        "Description",
                        grp.description,
                        key=f"gd_{gname}",
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
                            "🗑️ Delete",
                            key=f"gx_{gname}",
                            type="secondary",
                        ):
                            remove_test_group(store, gname)
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
        ng_name = st.text_input(
            "Name", placeholder="e.g. March sweep tests",
        )
        ng_desc = st.text_input("Description")
        ng_tests = (
            st.multiselect("Tests", run_names, key="ng_tests")
            if run_names
            else []
        )
        if not run_names:
            st.info(
                "Enter a data folder path in the sidebar to assign tests."
            )
        if st.form_submit_button("Create"):
            if not ng_name.strip():
                st.error("Name required.")
            elif ng_name in store.test_groups:
                st.error(
                    f"A test group named **{ng_name}** already exists."
                )
            else:
                add_test_group(
                    store, TestGroup(ng_name, ng_tests, ng_desc),
                )
                _persist()
                st.success(
                    f"Created **{ng_name}** "
                    f"with {len(ng_tests)} test(s)."
                )
                st.rerun()
