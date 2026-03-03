"""Bar Comparison page — define bars, group them into shipments, compare.

A **bar** is a named group of test folders belonging to the same physical
pump.  A **shipment** bundles several bars together under a recipient
label (e.g. “Niels’ bars”).  Both are persisted to ``bar_groups.json``.

User workflow
-------------
1. **Manage tab** — CRUD operations for bars and shipments.
   - Create a bar, assign test folders to it, add a description.
   - Create a shipment, pick bars, name the recipient.
   - Edit or delete existing bars / shipments at any time.
2. **Compare tab** — pick a saved shipment (or individual bars),
   then see comparison plots split by test type:
   - Sweep tests  →  overlay of binned mean±std curves, relative (0–100 %).
   - Constant-freq tests → per-test boxplots, aggregated boxplots,
     overlaid histograms.
   - Summary statistics table for every bar.

Inputs
------
- Local folder path (same OneDrive root as other pages).
- ``bar_groups.json`` for persisted bar/shipment definitions.
- CSV files for each test folder referenced by bars.

Outputs
-------
- Interactive Plotly comparison charts.
- ``pd.DataFrame`` summary table.
- Updated ``bar_groups.json`` on every create/edit/delete.
"""

from __future__ import annotations

import traceback

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
    detect_test_type,
    detect_time_format,
    guess_signal_column,
    guess_time_column,
    parse_sweep_spec_from_name,
    prepare_sweep_data,
    prepare_time_series_data,
)
from ..data.bar_groups import (
    Bar,
    BarGroupsStore,
    Shipment,
    add_bar,
    add_shipment,
    load_bar_groups,
    remove_bar,
    remove_shipment,
    save_bar_groups,
)
from ..plots.bar_comparison_plots import (
    build_bar_summary_table,
    plot_bar_constant_aggregated,
    plot_bar_constant_boxplots,
    plot_bar_constant_histograms,
    plot_bar_sweep_overlay,
    plot_bar_sweep_relative,
)
from ..plots.plot_generator import export_html


# ────────────────────────────────────────────────────────────────────────────
# Cached CSV loader (shared with analysis page via key)
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _load_csv_cached(csv_path_str: str) -> pd.DataFrame:
    """Load a CSV with Streamlit caching (5-min TTL, keyed by path string)."""
    return read_csv_full(csv_path_str)


# ────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ────────────────────────────────────────────────────────────────────────────

def _store_key() -> str:
    """Return the Streamlit session-state key for the bar-groups store."""
    return "_bar_groups_store"


def _get_store() -> BarGroupsStore:
    """Return the in-memory store, loading from disk on first access."""
    if _store_key() not in st.session_state:
        st.session_state[_store_key()] = load_bar_groups()
    return st.session_state[_store_key()]


def _persist() -> None:
    """Flush session store to JSON file."""
    save_bar_groups(_get_store())


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: C901
    """Entry point for the Bar Comparison page."""
    try:
        st.title("📦 Bar Comparison")
        st.markdown(
            "Define **bars** (groups of tests for the same pump), "
            "organise them into **shipments** (recipient + bars), "
            "and compare results across bars."
        )

        # ════════════════════════════════════════════════════════════════
        # SIDEBAR — data path + plot settings
        # ════════════════════════════════════════════════════════════════
        with st.sidebar:
            st.header("📁 Data Source")
            if "last_data_path" not in st.session_state:
                st.session_state.last_data_path = ""

            data_folder_str = st.text_input(
                "Path to test data folder",
                value=st.session_state.last_data_path,
                placeholder="/Users/.../All_tests",
                help="Same root folder used on other pages.",
                key="bar_data_path",
            )

            st.divider()
            st.header("⚙️ Settings")
            bin_hz = st.slider(
                "Frequency bin width (Hz)",
                min_value=0.5, max_value=100.0,
                value=float(PLOT_BIN_WIDTH_HZ), step=0.5,
                key="bar_bin_hz",
            )
            st.divider()
            st.header("🎨 Plot Appearance")
            plot_mode = st.selectbox(
                "Plot type",
                ["lines+markers", "lines", "markers"],
                key="bar_plot_mode",
            )
            marker_size = st.slider("Marker size", 1, 20, 6, key="bar_marker_size")
            show_error_bars = st.checkbox("Show ±1 std bands", value=True, key="bar_error_bars")
            show_individual = st.checkbox("Show individual test traces", value=False, key="bar_show_individual")
            export_html_toggle = st.checkbox("Export plots as HTML", value=False, key="bar_export")

        # ════════════════════════════════════════════════════════════════
        # Resolve data path (needed for test-folder discovery)
        # ════════════════════════════════════════════════════════════════
        root = None
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
                st.warning(f"⚠️ Could not read data folder: {e}")

        # ════════════════════════════════════════════════════════════════
        # TOP-LEVEL TABS — Manage / Compare
        # ════════════════════════════════════════════════════════════════
        tab_manage, tab_compare = st.tabs(["🛠️ Manage Bars & Shipments", "📊 Compare"])

        # ────────────────────────────────────────────────────────────
        # TAB 1 — MANAGE
        # ────────────────────────────────────────────────────────────
        with tab_manage:
            _render_manage_tab(run_names)

        # ────────────────────────────────────────────────────────────
        # TAB 2 — COMPARE
        # ────────────────────────────────────────────────────────────
        with tab_compare:
            _render_compare_tab(
                run_dirs=run_dirs,
                run_names=run_names,
                bin_hz=float(bin_hz),
                plot_mode=plot_mode,
                marker_size=marker_size,
                show_error_bars=show_error_bars,
                show_individual=show_individual,
                export_html_toggle=export_html_toggle,
            )

    except Exception as e:
        st.error(f"❌ **CRITICAL ERROR:** {e}")
        with st.expander("🔍 Debug Information"):
            st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════════
# MANAGE TAB
# ════════════════════════════════════════════════════════════════════════════

def _render_manage_tab(run_names: list[str]) -> None:
    """Render the Manage Bars & Shipments tab.

    Displays existing bars and shipments with edit/delete controls,
    plus forms to create new ones.  All changes are immediately
    persisted to ``bar_groups.json``.

    Parameters
    ----------
    run_names : list[str]
        Available test-folder names (used to populate multiselect widgets).
    """
    store = _get_store()

    # ── Section: Bars ───────────────────────────────────────────────────
    st.subheader("📦 Bars")
    st.caption("A bar is a group of test folders that belong to the **same pump**.")

    # Existing bars
    if store.bars:
        for bar_name, bar in list(store.bars.items()):
            with st.expander(f"**{bar_name}**  —  {len(bar.tests)} test(s)", expanded=False):
                st.markdown(f"*{bar.description}*" if bar.description else "_No description._")
                if bar.tests:
                    for t in bar.tests:
                        st.markdown(f"- `{t}`")
                else:
                    st.info("No tests assigned yet.")

                # Edit tests
                if run_names:
                    new_tests = st.multiselect(
                        "Edit tests",
                        options=run_names,
                        default=[t for t in bar.tests if t in run_names],
                        key=f"bar_edit_{bar_name}",
                    )
                    new_desc = st.text_input(
                        "Description",
                        value=bar.description,
                        key=f"bar_desc_{bar_name}",
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("💾 Save changes", key=f"bar_save_{bar_name}"):
                            bar.tests = new_tests
                            bar.description = new_desc
                            _persist()
                            st.success(f"Updated **{bar_name}**")
                            st.rerun()
                    with c2:
                        if st.button("🗑️ Delete bar", key=f"bar_del_{bar_name}", type="secondary"):
                            remove_bar(store, bar_name)
                            _persist()
                            st.success(f"Deleted **{bar_name}**")
                            st.rerun()
                else:
                    st.info("Enter a data folder path in the sidebar to enable test selection.")
    else:
        st.info("No bars defined yet. Create one below.")

    # Create new bar
    st.markdown("---")
    st.markdown("#### ➕ Create New Bar")
    with st.form("new_bar_form", clear_on_submit=True):
        new_bar_name = st.text_input("Bar name", placeholder="e.g. Bar 1")
        new_bar_desc = st.text_input("Description (optional)", placeholder="e.g. Pump serial #1234")
        if run_names:
            new_bar_tests = st.multiselect(
                "Assign tests",
                options=run_names,
                key="new_bar_tests",
            )
        else:
            new_bar_tests = []
            st.info("Enter a data folder path in the sidebar to assign tests.")

        submitted_bar = st.form_submit_button("Create Bar")
        if submitted_bar:
            if not new_bar_name.strip():
                st.error("Bar name is required.")
            elif new_bar_name in store.bars:
                st.error(f"A bar named **{new_bar_name}** already exists.")
            else:
                add_bar(store, Bar(name=new_bar_name, tests=new_bar_tests, description=new_bar_desc))
                _persist()
                st.success(f"Created **{new_bar_name}** with {len(new_bar_tests)} test(s).")
                st.rerun()

    # ── Section: Shipments ──────────────────────────────────────────────
    st.divider()
    st.subheader("🚚 Shipments")
    st.caption(
        "A shipment groups bars together under a **recipient** label "
        "(e.g. *Niels' bars*)."
    )

    bar_name_list = list(store.bars.keys())

    if store.shipments:
        for ship_name, ship in list(store.shipments.items()):
            with st.expander(
                f"**{ship_name}** — {ship.recipient or 'no recipient'} — {len(ship.bars)} bar(s)",
                expanded=False,
            ):
                st.markdown(f"**Recipient:** {ship.recipient or '—'}")
                st.markdown(f"**Description:** {ship.description or '—'}")
                if ship.bars:
                    for b in ship.bars:
                        n_tests = len(store.bars[b].tests) if b in store.bars else "?"
                        st.markdown(f"- **{b}** ({n_tests} tests)")
                else:
                    st.info("No bars assigned.")

                # Edit
                if bar_name_list:
                    new_ship_bars = st.multiselect(
                        "Edit bars",
                        options=bar_name_list,
                        default=[b for b in ship.bars if b in bar_name_list],
                        key=f"ship_bars_{ship_name}",
                    )
                    new_ship_recip = st.text_input(
                        "Recipient", value=ship.recipient, key=f"ship_recip_{ship_name}"
                    )
                    new_ship_desc = st.text_input(
                        "Description", value=ship.description, key=f"ship_desc_{ship_name}"
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("💾 Save changes", key=f"ship_save_{ship_name}"):
                            ship.bars = new_ship_bars
                            ship.recipient = new_ship_recip
                            ship.description = new_ship_desc
                            _persist()
                            st.success(f"Updated **{ship_name}**")
                            st.rerun()
                    with c2:
                        if st.button("🗑️ Delete shipment", key=f"ship_del_{ship_name}", type="secondary"):
                            remove_shipment(store, ship_name)
                            _persist()
                            st.success(f"Deleted **{ship_name}**")
                            st.rerun()
    else:
        st.info("No shipments defined yet. Create one below.")

    # Create new shipment
    st.markdown("---")
    st.markdown("#### ➕ Create New Shipment")
    with st.form("new_shipment_form", clear_on_submit=True):
        new_ship_name = st.text_input("Shipment name", placeholder="e.g. Niels' bars")
        new_ship_recip = st.text_input("Recipient", placeholder="e.g. Niels")
        new_ship_desc = st.text_input("Description (optional)", placeholder="e.g. Shipped 2026-03-01")
        if bar_name_list:
            new_ship_bars = st.multiselect(
                "Assign bars",
                options=bar_name_list,
                key="new_ship_bars",
            )
        else:
            new_ship_bars = []
            st.info("Create bars first before making a shipment.")

        submitted_ship = st.form_submit_button("Create Shipment")
        if submitted_ship:
            if not new_ship_name.strip():
                st.error("Shipment name is required.")
            elif new_ship_name in store.shipments:
                st.error(f"A shipment named **{new_ship_name}** already exists.")
            else:
                add_shipment(
                    store,
                    Shipment(
                        name=new_ship_name,
                        bars=new_ship_bars,
                        recipient=new_ship_recip,
                        description=new_ship_desc,
                    ),
                )
                _persist()
                st.success(f"Created shipment **{new_ship_name}**.")
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# COMPARE TAB
# ════════════════════════════════════════════════════════════════════════════

def _render_compare_tab(
    *,
    run_dirs: list,
    run_names: list[str],
    bin_hz: float,
    plot_mode: str,
    marker_size: int,
    show_error_bars: bool,
    show_individual: bool,
    export_html_toggle: bool,
) -> None:
    """Render the Compare tab.

    Loads test data for all selected bars, classifies each test as
    sweep or constant-frequency, and then renders comparison tabs
    with the appropriate chart types.

    Parameters
    ----------
    run_dirs : list[Path]
        Resolved run-directory paths (one per test folder).
    run_names : list[str]
        Corresponding folder names.
    bin_hz : float
        Frequency bin width for sweep binning.
    plot_mode : str
        Plotly trace mode (``"lines"``, ``"markers"``, ``"lines+markers"``).
    marker_size : int
        Marker diameter in pixels.
    show_error_bars : bool
        Whether to draw ±1 std bands on sweep overlays.
    show_individual : bool
        Whether to draw faint per-test traces behind bar averages.
    export_html_toggle : bool
        Whether to save each chart as a standalone HTML file.
    """
    store = _get_store()

    if not store.bars:
        st.info(
            "👈 Switch to the **Manage** tab to create bars and shipments first."
        )
        return

    # ── Selection: shipment OR manual bars ──────────────────────────────
    select_mode = st.radio(
        "Select bars to compare",
        ["Pick a saved shipment", "Pick bars manually"],
        horizontal=True,
        key="bar_compare_mode",
    )

    selected_bar_names: list[str] = []

    if select_mode == "Pick a saved shipment":
        if not store.shipments:
            st.info("No shipments defined yet. Create one in the Manage tab, or pick bars manually.")
            return
        ship_options = list(store.shipments.keys())
        ship_labels = [
            f"{name}  ({store.shipments[name].recipient or '—'})"
            for name in ship_options
        ]
        chosen_idx = st.selectbox(
            "🚚 Shipment",
            range(len(ship_options)),
            format_func=lambda i: ship_labels[i],
            key="bar_compare_shipment",
        )
        chosen_ship = store.shipments[ship_options[chosen_idx]]
        selected_bar_names = [b for b in chosen_ship.bars if b in store.bars]

        st.markdown(
            f"**Shipment:** {chosen_ship.name}  ·  "
            f"**Recipient:** {chosen_ship.recipient or '—'}  ·  "
            f"**Bars:** {', '.join(selected_bar_names) or 'none'}"
        )
    else:
        bar_name_list = list(store.bars.keys())
        selected_bar_names = st.multiselect(
            "📦 Bars to compare",
            options=bar_name_list,
            key="bar_compare_manual",
        )

    if not selected_bar_names:
        st.info("👆 Select at least one bar to start the comparison.")
        return

    if len(selected_bar_names) < 2:
        st.warning("Select **≥ 2 bars** for meaningful comparison plots.")

    # ── Load data for every test in every selected bar ──────────────────
    # Classify tests into sweep vs constant
    bar_sweep_binned: dict[str, dict[str, pd.DataFrame]] = {}  # bar → {test → binned}
    bar_sweep_raw: dict[str, dict[str, pd.DataFrame]] = {}      # bar → {test → sweep_df}
    bar_const_data: dict[str, dict[str, pd.DataFrame]] = {}     # bar → {test → ts_df}
    load_errors: list[str] = []
    signal_col_used: str | None = None

    run_name_to_dir = {p.name: p for p in run_dirs}

    with st.spinner(f"Loading data for {len(selected_bar_names)} bar(s)…"):
        for bar_name in selected_bar_names:
            bar = store.bars[bar_name]
            bar_sweep_binned[bar_name] = {}
            bar_sweep_raw[bar_name] = {}
            bar_const_data[bar_name] = {}

            for test_name in bar.tests:
                if test_name not in run_name_to_dir:
                    load_errors.append(f"{bar_name}/{test_name}: folder not found in data path")
                    continue
                run_dir = run_name_to_dir[test_name]
                try:
                    pick = pick_best_csv(run_dir)
                    df = _load_csv_cached(str(pick.csv_path))
                    if df.empty:
                        load_errors.append(f"{bar_name}/{test_name}: empty CSV")
                        continue

                    time_col = guess_time_column(df)
                    sig_col = guess_signal_column(df, "flow")
                    if not time_col or not sig_col:
                        load_errors.append(f"{bar_name}/{test_name}: cannot detect columns")
                        continue
                    if signal_col_used is None:
                        signal_col_used = sig_col

                    time_fmt = detect_time_format(df, time_col)

                    # Classify
                    test_type, _, _ = detect_test_type(test_name, df)
                    is_sweep = test_type == "sweep"

                    ts_df = prepare_time_series_data(
                        df, time_col, sig_col,
                        parse_time=(time_fmt == "absolute_timestamp"),
                    )

                    if is_sweep:
                        # Frequency sweep path
                        has_freq = "freq_set_hz" in df.columns
                        spec = parse_sweep_spec_from_name(test_name)

                        if has_freq or (spec and spec.duration_s > 0):
                            sweep_df = prepare_sweep_data(
                                ts_df, time_col, sig_col,
                                spec=spec,
                                parse_time=(time_fmt == "absolute_timestamp"),
                                full_df=df if has_freq else None,
                            )
                            bar_sweep_raw[bar_name][test_name] = sweep_df
                            try:
                                binned = bin_by_frequency(
                                    sweep_df, value_col=sig_col,
                                    freq_col="Frequency", bin_hz=bin_hz,
                                )
                                bar_sweep_binned[bar_name][test_name] = binned
                            except Exception as be:
                                load_errors.append(f"{bar_name}/{test_name}: binning — {be}")
                        else:
                            # Sweep detected by metadata/regex but no usable freq data
                            bar_const_data[bar_name][test_name] = ts_df
                    else:
                        # Constant frequency or unknown → constant path
                        bar_const_data[bar_name][test_name] = ts_df

                except Exception as e:
                    load_errors.append(f"{bar_name}/{test_name}: {e}")

    # ── Load summary ────────────────────────────────────────────────────
    if load_errors:
        with st.expander(f"⚠️ {len(load_errors)} load issue(s)", expanded=False):
            for err in load_errors:
                st.warning(err)

    if signal_col_used is None:
        signal_col_used = "flow"

    n_sweep_bars = sum(1 for v in bar_sweep_binned.values() if v)
    n_const_bars = sum(1 for v in bar_const_data.values() if v)
    total_tests = sum(
        len(bar_sweep_binned.get(b, {})) + len(bar_const_data.get(b, {}))
        for b in selected_bar_names
    )

    if total_tests == 0:
        st.error("❌ No test data loaded for the selected bars. Check the data folder path and bar test assignments.")
        return

    st.success(
        f"✅ Loaded **{total_tests}** test(s) across **{len(selected_bar_names)}** bar(s)  ·  "
        f"**{n_sweep_bars}** bar(s) with sweep data  ·  "
        f"**{n_const_bars}** bar(s) with constant-freq data"
    )

    # ── Detailed bar inventory ──────────────────────────────────────────
    with st.expander("📋 Bar contents", expanded=False):
        for bar_name in selected_bar_names:
            sw_tests = list(bar_sweep_binned.get(bar_name, {}).keys())
            cf_tests = list(bar_const_data.get(bar_name, {}).keys())
            st.markdown(f"**{bar_name}**")
            if sw_tests:
                st.markdown(f"  - Sweep tests ({len(sw_tests)}): " + ", ".join(f"`{t}`" for t in sw_tests))
            if cf_tests:
                st.markdown(f"  - Constant-freq tests ({len(cf_tests)}): " + ", ".join(f"`{t}`" for t in cf_tests))
            if not sw_tests and not cf_tests:
                st.markdown("  - _no loaded tests_")

    # ════════════════════════════════════════════════════════════════════
    # COMPARISON TABS
    # ════════════════════════════════════════════════════════════════════
    tab_labels: list[str] = []
    if n_sweep_bars:
        tab_labels += ["🔀 Sweep Comparison", "📏 Sweep Relative"]
    if n_const_bars:
        tab_labels += ["📦 Constant-Freq Boxplots", "📊 Constant-Freq Histograms"]
    tab_labels.append("📋 Summary Table")

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # ── Sweep Comparison ────────────────────────────────────────────────
    if n_sweep_bars:
        with tabs[tab_idx]:
            st.subheader("Frequency Sweep — Bar Comparison")
            fig_sw = plot_bar_sweep_overlay(
                bar_sweep_binned,
                show_error_bars=show_error_bars,
                show_individual=show_individual,
                mode=plot_mode,
                marker_size=marker_size,
            )
            st.plotly_chart(fig_sw, use_container_width=True)
            if export_html_toggle:
                try:
                    p = export_html(fig_sw, "bar_sweep_overlay.html")
                    st.success(f"✅ Exported: {p.name}")
                except Exception:
                    pass
        tab_idx += 1

        with tabs[tab_idx]:
            st.subheader("Relative (0–100 %) — Bar Comparison")
            fig_rel = plot_bar_sweep_relative(
                bar_sweep_binned,
                mode=plot_mode,
                marker_size=marker_size,
            )
            st.plotly_chart(fig_rel, use_container_width=True)
            if export_html_toggle:
                try:
                    p = export_html(fig_rel, "bar_sweep_relative.html")
                    st.success(f"✅ Exported: {p.name}")
                except Exception:
                    pass
        tab_idx += 1

    # ── Constant-Freq Boxplots ──────────────────────────────────────────
    if n_const_bars:
        with tabs[tab_idx]:
            st.subheader("Constant-Frequency — Bar Comparison")

            view = st.radio(
                "View",
                ["Per-test (grouped by bar)", "Aggregated per bar"],
                horizontal=True,
                key="bar_const_view",
            )
            if view.startswith("Per"):
                fig_box = plot_bar_constant_boxplots(
                    bar_const_data, signal_col=signal_col_used,
                )
            else:
                fig_box = plot_bar_constant_aggregated(
                    bar_const_data, signal_col=signal_col_used,
                )
            st.plotly_chart(fig_box, use_container_width=True)
            if export_html_toggle:
                try:
                    p = export_html(fig_box, "bar_constant_boxplots.html")
                    st.success(f"✅ Exported: {p.name}")
                except Exception:
                    pass
        tab_idx += 1

        with tabs[tab_idx]:
            st.subheader("Constant-Frequency Histograms — Bar Comparison")
            nbins = st.slider("Bins", 10, 200, 50, key="bar_const_hist_bins")
            fig_hist = plot_bar_constant_histograms(
                bar_const_data, signal_col=signal_col_used, nbins=nbins,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            if export_html_toggle:
                try:
                    p = export_html(fig_hist, "bar_constant_histograms.html")
                    st.success(f"✅ Exported: {p.name}")
                except Exception:
                    pass
        tab_idx += 1

    # ── Summary Table ───────────────────────────────────────────────────
    with tabs[tab_idx]:
        st.subheader("Summary Statistics per Bar")

        dfs: list[pd.DataFrame] = []
        if n_sweep_bars:
            # Pool sweep raw data per bar for summary stats
            bar_sweep_pool: dict[str, dict[str, pd.DataFrame]] = {}
            for bar_name in selected_bar_names:
                if bar_sweep_raw.get(bar_name):
                    bar_sweep_pool[bar_name] = bar_sweep_raw[bar_name]
            if bar_sweep_pool:
                dfs.append(
                    build_bar_summary_table(
                        bar_sweep_pool, signal_col=signal_col_used, test_type="Sweep"
                    )
                )
        if n_const_bars:
            const_with_data = {
                b: t for b, t in bar_const_data.items() if t
            }
            if const_with_data:
                dfs.append(
                    build_bar_summary_table(
                        const_with_data, signal_col=signal_col_used, test_type="Constant"
                    )
                )

        if dfs:
            summary = pd.concat(dfs, ignore_index=True)
            st.dataframe(
                summary, use_container_width=True, hide_index=True,
                column_config={"CV (%)": st.column_config.NumberColumn(format="%.2f")},
            )
        else:
            st.info("No data available for summary.")
