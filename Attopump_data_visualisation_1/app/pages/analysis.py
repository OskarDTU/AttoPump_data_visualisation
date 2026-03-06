"""Unified Comprehensive Analysis page — compare tests and pumps.

This page consolidates all multi-test and multi-pump comparison into a
single interface with two modes:

Modes
-----
1. **Compare Tests** — select individual tests (ad-hoc or from a saved
   test group) and run cross-test analysis with 7+ chart types.  If the
   selection mixes constant-frequency and sweep tests, the page warns
   and optionally extracts sweep data at the matching frequency.

2. **Compare Pumps** — select pumps or a shipment and
   compare aggregated results across pumps.

Inputs
------
- Local folder path (OneDrive-synced root with test subfolders).
- ``bar_groups.json`` for persisted definitions.
- CSV files for each test folder.

Outputs
-------
- Interactive Plotly charts and summary tables.
- Updated ``bar_groups.json`` on create / edit / delete.
"""

from __future__ import annotations

import traceback
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

from ..data.io_local import (
    list_run_dirs,
    normalize_root,
    pick_best_csv,
    read_csv_full,
)
from ..data.config import (
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
from ..data.bar_groups import (
    BarGroupsStore,
    TestGroup,
    add_test_group,
    load_bar_groups,
    save_bar_groups,
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
# Cached loader & session-state helpers
# ────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _load_csv_cached(csv_path_str: str) -> pd.DataFrame:
    """Load a CSV with Streamlit caching (5-min TTL)."""
    return read_csv_full(csv_path_str)


def _get_store() -> BarGroupsStore:
    """Return the in-memory bar-groups store (loads from disk on first access)."""
    key = "_bar_groups_store"
    if key not in st.session_state:
        st.session_state[key] = load_bar_groups()
    return st.session_state[key]


def _persist() -> None:
    """Flush the bar-groups store to ``bar_groups.json``."""
    save_bar_groups(_get_store())


def _maybe_export(fig, filename: str, S: dict) -> None:
    """Export a figure to HTML if the toggle is on."""
    if S.get("export"):
        try:
            path = export_html(fig, filename)
            st.success(f"✅ Exported: {path.name}")
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def main() -> None:  # noqa: C901
    """Entry point for the Comprehensive Analysis page."""
    try:
        st.title("🔬 Comprehensive Analysis")

        # ── Sidebar ─────────────────────────────────────────────
        with st.sidebar:
            st.header("📁 Data Source")
            if "last_data_path" not in st.session_state:
                st.session_state.last_data_path = ""
            data_folder_str = st.text_input(
                "Path to test data folder",
                value=st.session_state.last_data_path,
                placeholder="/Users/.../All_tests",
                key="analysis_data_path",
            )

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
            max_raw = st.slider(
                "Max raw points per test", 1000, 500000,
                50000, 5000, key="a_maxpts",
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
                "Export plots as HTML", False, key="a_exp",
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
            freq_tol=freq_tol, plot_mode=plot_mode, marker_sz=marker_sz,
            opacity=opacity, err_bars=err_bars, export=export,
            mean_pct=mean_pct, std_pct=std_pct,
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
                st.error(f"❌ Invalid path: {e}")

        # ── Mode selector ──────────────────────────────────────
        mode = st.radio(
            "What would you like to do?",
            ["📊 Compare Tests", "📦 Compare Pumps"],
            horizontal=True,
            key="a_toplevel",
        )
        st.divider()

        if mode == "📊 Compare Tests":
            _render_compare_tests(S, run_dirs, run_names)
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
    store = _get_store()

    # ── Selection method ────────────────────────────────────────
    sel = st.radio(
        "How would you like to select tests?",
        ["Pick from available tests", "Load a saved test group"],
        horizontal=True,
        key="ct_sel",
    )

    selected_names: list[str] = []

    if sel == "Load a saved test group":
        if not store.test_groups:
            st.info(
                "No saved test groups yet.  Create one on the **Manage Groups** page, "
                "or pick tests manually."
            )
            return
        grp_names = list(store.test_groups.keys())
        chosen = st.selectbox("📋 Test group", grp_names, key="ct_grp")
        selected_names = [
            t for t in store.test_groups[chosen].tests if t in run_names
        ]
        st.caption(
            f"Group contains {len(store.test_groups[chosen].tests)} test(s), "
            f"{len(selected_names)} available in current data folder."
        )
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
                        store,
                        TestGroup(grp_name, list(selected_names), grp_desc),
                    )
                    _persist()
                    st.success(
                        f"Saved **{grp_name}** ({len(selected_names)} tests)"
                    )
                else:
                    st.error("Name required.")

    # ── Check for unclassified tests ───────────────────────────
    _, unknowns = classify_tests_quick(selected_names, run_dirs, run_names)
    if unknowns:
        all_classified = render_unknown_test_prompt(
            unknowns, run_dirs, run_names, key_prefix="ct_utp",
        )
        if not all_classified:
            st.info(
                "💡 Classify the unknown test(s) above, then the "
                "analysis will load automatically."
            )
            return

    # ── Load data ───────────────────────────────────────────────
    data = _load_tests(selected_names, run_dirs, run_names, S)
    if data is None:
        return

    (all_data, sweep_data, binned_data, const_data,
     signal_col, load_errors, test_types, const_freqs) = data

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

    # ── Mixed-type warning ──────────────────────────────────────
    mixed_mode = False
    matched_const_data: dict[str, pd.DataFrame] = {}
    target_freq: float | None = None

    if n_sweep > 0 and n_const > 0:
        # Auto-detect the constant frequency
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
            f"**{target_freq:.0f} Hz** (±{S['freq_tol']:.0f} Hz) "
            f"to match constant-frequency tests",
            key="ct_mixed",
        )

        if mixed_mode:
            target_freq_override = st.number_input(
                "Override target frequency (Hz)",
                value=float(target_freq),
                min_value=0.1,
                step=10.0,
                key="ct_target_freq",
            )
            target_freq = target_freq_override

            # Build matched data: const tests as-is, sweep tests sliced
            for name, df in const_data.items():
                matched_const_data[name] = df
            for name, sdf in sweep_data.items():
                sliced = extract_frequency_slice(
                    sdf, target_freq, S["freq_tol"], signal_col,
                )
                if not sliced.empty:
                    matched_const_data[f"{name} @{target_freq:.0f}Hz"] = sliced

    # ── Analysis tabs ───────────────────────────────────────────
    # Cross-type comparison (if user opted in)
    if mixed_mode and matched_const_data:
        _render_mixed_comparison(
            matched_const_data, signal_col, target_freq, S,
        )
        st.divider()

    # Sweep analysis (for sweep tests)
    if n_sweep > 0:
        if n_const > 0:
            st.subheader(f"📈 Frequency-Sweep Tests ({n_sweep})")
        _render_sweep_analysis(sweep_data, binned_data, signal_col, S)

    # Constant-freq analysis (if not already covered by mixed comparison)
    if n_const > 0 and not mixed_mode:
        if n_sweep > 0:
            st.subheader(f"📊 Constant-Frequency Tests ({n_const})")
        _render_constant_analysis(const_data, signal_col, S)


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
                df = _load_csv_cached(str(pick.csv_path))
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
) -> None:
    """Render the full set of sweep-analysis tabs."""
    n_binned = len(binned_data)

    tabs = st.tabs([
        "📊 Individual",
        "🔀 Combined",
        "📏 Average",
        "⚡ Raw Points",
        "📦 EDA",
        "📐 Std vs Mean",
        "🎯 Best Region",
    ])

    # ── TAB 1: Individual per-test ──────────────────────────────
    with tabs[0]:
        st.subheader("Individual Test Results")
        if not binned_data:
            st.info("No frequency-binned data available.")
        else:
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

                    # Quick metrics
                    if name in sweep_data and signal_col in sweep_data[name].columns:
                        s = sweep_data[name][signal_col].dropna()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Mean", f"{s.mean():.2f} µL/min")
                        c2.metric("Std", f"{s.std():.2f} µL/min")
                        c3.metric("Min", f"{s.min():.2f} µL/min")
                        c4.metric("Max", f"{s.max():.2f} µL/min")

                    _maybe_export(fig_b, f"{name}_binned.html", S)

    # ── TAB 2: Combined overlay ─────────────────────────────────
    with tabs[1]:
        st.subheader("Combined Frequency Sweep Comparison")
        if n_binned < 1:
            st.info("Need at least 1 test with frequency data.")
        else:
            fig_comb = plot_combined_overlay(
                binned_data,
                show_error_bars=S["err_bars"],
                mode=S["plot_mode"],
                marker_size=S["marker_sz"],
            )
            st.plotly_chart(fig_comb, use_container_width=True)
            _maybe_export(fig_comb, "combined_overlay.html", S)

            st.divider()
            st.subheader("Relative (0–100 %) Comparison")
            fig_rel = plot_relative_comparison(
                binned_data,
                mode=S["plot_mode"],
                marker_size=S["marker_sz"],
            )
            st.plotly_chart(fig_rel, use_container_width=True)

    # ── TAB 3: Global average ───────────────────────────────────
    with tabs[2]:
        st.subheader("Global Average Across Tests")
        if n_binned < 2:
            st.info("Need ≥ 2 tests for averaging.")
        else:
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
                st.dataframe(
                    avg_df, use_container_width=True, hide_index=True,
                )

    # ── TAB 4: Raw points ───────────────────────────────────────
    with tabs[3]:
        st.subheader("All Raw Data Points")
        if not sweep_data:
            st.info("No frequency data.")
        else:
            capped: dict[str, pd.DataFrame] = {}
            for n, d in sweep_data.items():
                capped[n] = (
                    d.sample(n=S["max_raw"], random_state=42)
                    if len(d) > S["max_raw"] else d
                )
            total_pts = sum(len(d) for d in capped.values())
            st.caption(
                f"Showing {total_pts:,} points "
                f"(capped at {S['max_raw']:,}/test)"
            )
            fig_raw = plot_all_raw_points(
                capped,
                freq_col="Frequency",
                signal_col=signal_col,
                marker_size=S["marker_sz"],
                opacity=S["opacity"],
            )
            st.plotly_chart(fig_raw, use_container_width=True)
            _maybe_export(fig_raw, "all_raw_points.html", S)

    # ── TAB 5: EDA ──────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Exploratory Data Analysis")

        eda_data = {
            n: d for n, d in sweep_data.items()
            if signal_col in d.columns
        }
        if not eda_data:
            eda_data = dict(sweep_data)

        # Summary table
        st.markdown("### 📋 Summary Statistics")
        stats = build_summary_table(eda_data, signal_col=signal_col)
        if not stats.empty:
            st.dataframe(
                stats, use_container_width=True, hide_index=True,
                column_config={
                    "CV (%)": st.column_config.NumberColumn(format="%.2f"),
                },
            )

        # Boxplots + Histograms
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("### 📦 Boxplots")
            st.plotly_chart(
                plot_combined_boxplots(eda_data, signal_col=signal_col),
                use_container_width=True,
            )
        with col_r:
            st.markdown("### 📊 Histograms")
            nbins = st.slider("Bins", 10, 200, 50, key="eda_bins")
            st.plotly_chart(
                plot_combined_histograms(
                    eda_data, signal_col=signal_col, nbins=nbins,
                ),
                use_container_width=True,
            )

        # Correlation heatmap
        if n_binned >= 2:
            st.markdown("### 🔗 Inter-Test Correlation")
            st.plotly_chart(
                plot_correlation_heatmap(binned_data),
                use_container_width=True,
            )

    # ── TAB 6: Std vs Mean ──────────────────────────────────────
    with tabs[5]:
        st.subheader("Variability: Std vs Mean")
        if not binned_data:
            st.info("No binned data.")
        else:
            fig_svm = plot_std_vs_mean(
                binned_data, marker_size=S["marker_sz"],
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

    # ── TAB 7: Best region ──────────────────────────────────────
    with tabs[6]:
        st.subheader("Best Operating Region")
        if not binned_data:
            st.info("No binned data.")
        else:
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
) -> None:
    """Render comparison tabs for constant-frequency tests."""
    tabs = st.tabs(["📦 Boxplots", "📊 Histograms", "📋 Summary"])

    with tabs[0]:
        fig = plot_combined_boxplots(
            const_data, signal_col=signal_col,
            title="Constant-Frequency Flow Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        nbins = st.slider("Bins", 10, 200, 50, key="const_bins")
        fig = plot_combined_histograms(
            const_data, signal_col=signal_col, nbins=nbins,
            title="Constant-Frequency Flow Histograms",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        stats = build_summary_table(const_data, signal_col=signal_col)
        if not stats.empty:
            st.dataframe(
                stats, use_container_width=True, hide_index=True,
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
) -> None:
    """Render cross-type comparison at the matched frequency."""
    st.subheader(f"🔀 Cross-Type Comparison at {target_freq:.0f} Hz")
    st.caption(
        f"Constant-frequency tests shown as-is.  Sweep tests filtered to "
        f"{target_freq:.0f} ± {S['freq_tol']:.0f} Hz."
    )

    tabs = st.tabs(["📦 Boxplots", "📊 Histograms", "📋 Summary"])

    with tabs[0]:
        fig = plot_combined_boxplots(
            matched_data, signal_col=signal_col,
            title=f"Flow at {target_freq:.0f} Hz — All Tests",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        nbins = st.slider("Bins", 10, 200, 50, key="mixed_bins")
        fig = plot_combined_histograms(
            matched_data, signal_col=signal_col, nbins=nbins,
            title=f"Flow Distribution at {target_freq:.0f} Hz",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        stats = build_summary_table(matched_data, signal_col=signal_col)
        if not stats.empty:
            st.dataframe(
                stats, use_container_width=True, hide_index=True,
                column_config={
                    "CV (%)": st.column_config.NumberColumn(format="%.2f"),
                },
            )


# ════════════════════════════════════════════════════════════════════════
# MODE 2 — COMPARE PUMPS
# ════════════════════════════════════════════════════════════════════════

def _render_compare_bars(
    S: dict, run_dirs: list, run_names: list[str],
) -> None:
    """Compare Pumps mode — select pumps or shipment, load, compare."""
    store = _get_store()

    if not store.bars:
        st.info(
            "No pumps defined yet.  Go to the **Manage Groups** page to create pumps."
        )
        return

    # ── Selection ───────────────────────────────────────────────
    sel = st.radio(
        "How would you like to select pumps?",
        ["Pick pumps manually", "Load a saved shipment"],
        horizontal=True,
        key="cb_sel",
    )

    selected_bar_names: list[str] = []

    if sel == "Load a saved shipment":
        if not store.shipments:
            st.info(
                "No shipments saved.  Create one on the **Manage Groups** page, "
                "or pick pumps manually."
            )
            return
        opts = list(store.shipments.keys())
        labels = [
            f"{n}  ({store.shipments[n].recipient or '—'})"
            for n in opts
        ]
        chosen_idx = st.selectbox(
            "🚚 Shipment",
            range(len(opts)),
            format_func=lambda i: labels[i],
            key="cb_ship",
        )
        ship = store.shipments[opts[chosen_idx]]
        selected_bar_names = [b for b in ship.bars if b in store.bars]
        st.markdown(
            f"**Shipment:** {ship.name}  ·  "
            f"**Recipient:** {ship.recipient or '—'}  ·  "
            f"**Pumps:** {', '.join(selected_bar_names) or 'none'}"
        )
    else:
        selected_bar_names = st.multiselect(
            "� Pumps to compare",
            list(store.bars.keys()),
            key="cb_bars",
        )

    if not selected_bar_names:
        st.info("👆 Select at least one pump.")
        return
    if len(selected_bar_names) < 2:
        st.warning("Select **≥ 2 pumps** for meaningful comparison.")

    # ── Check for unclassified tests across bars ───────────────
    all_bar_tests = []
    for bn in selected_bar_names:
        all_bar_tests.extend(
            t for t in store.bars[bn].tests if t in run_names
        )
    if all_bar_tests:
        _, unknowns = classify_tests_quick(all_bar_tests, run_dirs, run_names)
        if unknowns:
            all_classified = render_unknown_test_prompt(
                unknowns, run_dirs, run_names, key_prefix="cb_utp",
            )
            if not all_classified:
                st.info(
                    "💡 Classify the unknown test(s) above, then the "
                    "comparison will load automatically."
                )
                return

    # ── Load bar data ───────────────────────────────────────────
    run_map = {p.name: p for p in run_dirs}
    bar_sweep_binned: dict[str, dict[str, pd.DataFrame]] = {}
    bar_sweep_raw: dict[str, dict[str, pd.DataFrame]] = {}
    bar_const_data: dict[str, dict[str, pd.DataFrame]] = {}
    load_errors: list[str] = []
    signal_col: str | None = None

    with st.spinner(f"Loading {len(selected_bar_names)} pump(s)…"):
        for bar_name in selected_bar_names:
            bar = store.bars[bar_name]
            bar_sweep_binned[bar_name] = {}
            bar_sweep_raw[bar_name] = {}
            bar_const_data[bar_name] = {}

            for test_name in bar.tests:
                if test_name not in run_map:
                    load_errors.append(
                        f"{bar_name}/{test_name}: not in data folder"
                    )
                    continue

                try:
                    pick = pick_best_csv(run_map[test_name])
                    df = _load_csv_cached(str(pick.csv_path))
                    if df.empty:
                        load_errors.append(f"{bar_name}/{test_name}: empty")
                        continue

                    tc = guess_time_column(df)
                    sc = guess_signal_column(df, "flow")
                    if not tc or not sc:
                        load_errors.append(
                            f"{bar_name}/{test_name}: columns?"
                        )
                        continue
                    if signal_col is None:
                        signal_col = sc

                    tfmt = detect_time_format(df, tc)
                    ts = prepare_time_series_data(
                        df, tc, sc,
                        parse_time=(tfmt == "absolute_timestamp"),
                    )

                    ttype, _, _ = detect_test_type(
                        test_name,
                        df,
                        data_root=run_map[test_name].parent,
                    )
                    if ttype == "sweep":
                        has_freq = "freq_set_hz" in df.columns
                        spec = parse_sweep_spec_from_name(test_name)
                        if has_freq or (spec and spec.duration_s > 0):
                            sdf = prepare_sweep_data(
                                ts, tc, sc,
                                spec=spec,
                                parse_time=(tfmt == "absolute_timestamp"),
                                full_df=df if has_freq else None,
                            )
                            bar_sweep_raw[bar_name][test_name] = sdf
                            try:
                                b = bin_by_frequency(
                                    sdf, value_col=sc,
                                    freq_col="Frequency",
                                    bin_hz=S["bin_hz"],
                                )
                                bar_sweep_binned[bar_name][test_name] = b
                            except Exception as be:
                                load_errors.append(
                                    f"{bar_name}/{test_name}: bin — {be}"
                                )
                        else:
                            bar_const_data[bar_name][test_name] = ts
                    else:
                        bar_const_data[bar_name][test_name] = ts

                except Exception as e:
                    load_errors.append(f"{bar_name}/{test_name}: {e}")

    if load_errors:
        with st.expander(f"⚠️ {len(load_errors)} issue(s)", expanded=False):
            for e in load_errors:
                st.warning(e)

    if signal_col is None:
        signal_col = "flow"

    n_sw = sum(1 for v in bar_sweep_binned.values() if v)
    n_cf = sum(1 for v in bar_const_data.values() if v)
    total = sum(
        len(bar_sweep_binned.get(b, {})) + len(bar_const_data.get(b, {}))
        for b in selected_bar_names
    )

    if total == 0:
        st.error(
            "❌ No data loaded.  Check the data folder path "
            "and pump test assignments."
        )
        return

    st.success(
        f"✅ **{total}** tests across **{len(selected_bar_names)}** pumps — "
        f"**{n_sw}** sweep, **{n_cf}** constant"
    )

    # ── Mixed-type warning for pumps ─────────────────────────────
    if n_sw > 0 and n_cf > 0:
        st.warning(
            "⚠️ Some pumps contain a mix of sweep and constant-frequency "
            "tests.  Results are shown separately by test type below."
        )

    # ── Pump inventory ───────────────────────────────────────────
    with st.expander("📋 Pump contents", expanded=False):
        for bar_name in selected_bar_names:
            sw = list(bar_sweep_binned.get(bar_name, {}).keys())
            cf = list(bar_const_data.get(bar_name, {}).keys())
            st.markdown(f"**{bar_name}**")
            if sw:
                st.markdown(
                    f"  - Sweep ({len(sw)}): "
                    + ", ".join(f"`{t}`" for t in sw)
                )
            if cf:
                st.markdown(
                    f"  - Constant ({len(cf)}): "
                    + ", ".join(f"`{t}`" for t in cf)
                )
            if not sw and not cf:
                st.markdown("  - _no loaded tests_")

    # ── Pump comparison tabs ─────────────────────────────────────
    tab_labels: list[str] = []
    if n_sw:
        tab_labels += ["🔀 Sweep Overlay", "📏 Sweep Relative"]
    if n_cf:
        tab_labels += ["📦 Const Boxplots", "📊 Const Histograms"]
    tab_labels.append("📋 Summary")

    btabs = st.tabs(tab_labels)
    ti = 0

    # Sweep tabs
    if n_sw:
        with btabs[ti]:
            st.subheader("Frequency Sweep — Pump Comparison")
            show_indiv = st.checkbox(
                "Show individual test traces",
                value=False,
                key="cb_indiv",
            )
            fig = plot_bar_sweep_overlay(
                bar_sweep_binned,
                show_error_bars=S["err_bars"],
                show_individual=show_indiv,
                mode=S["plot_mode"],
                marker_size=S["marker_sz"],
            )
            st.plotly_chart(fig, use_container_width=True)
            _maybe_export(fig, "pump_sweep_overlay.html", S)
        ti += 1

        with btabs[ti]:
            st.subheader("Relative (0–100 %) — Pump Comparison")
            fig = plot_bar_sweep_relative(
                bar_sweep_binned,
                mode=S["plot_mode"],
                marker_size=S["marker_sz"],
            )
            st.plotly_chart(fig, use_container_width=True)
        ti += 1

    # Constant tabs
    if n_cf:
        with btabs[ti]:
            st.subheader("Constant-Frequency — Pump Comparison")
            view = st.radio(
                "View",
                ["Per-test (grouped by pump)", "Aggregated per pump"],
                horizontal=True,
                key="cb_const_view",
            )
            if view.startswith("Per"):
                fig = plot_bar_constant_boxplots(
                    bar_const_data, signal_col=signal_col,
                )
            else:
                fig = plot_bar_constant_aggregated(
                    bar_const_data, signal_col=signal_col,
                )
            st.plotly_chart(fig, use_container_width=True)
        ti += 1

        with btabs[ti]:
            st.subheader("Constant-Frequency Histograms")
            nbins = st.slider("Bins", 10, 200, 50, key="cb_hbins")
            fig = plot_bar_constant_histograms(
                bar_const_data, signal_col=signal_col, nbins=nbins,
            )
            st.plotly_chart(fig, use_container_width=True)
        ti += 1

    # Summary table
    with btabs[ti]:
        st.subheader("Summary Statistics per Pump")
        dfs: list[pd.DataFrame] = []
        if n_sw:
            pool = {b: t for b, t in bar_sweep_raw.items() if t}
            if pool:
                dfs.append(
                    build_bar_summary_table(
                        pool, signal_col=signal_col, test_type="Sweep",
                    )
                )
        if n_cf:
            pool = {b: t for b, t in bar_const_data.items() if t}
            if pool:
                dfs.append(
                    build_bar_summary_table(
                        pool, signal_col=signal_col, test_type="Constant",
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
