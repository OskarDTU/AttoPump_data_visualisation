"""Comprehensive Analysis page — multi-test comparison and EDA.

Inspired by:
  - 03-06-2025-1338_flow_and_pressure_analysis.py
  - 27-05-2025-0041_configuration_comparison.py

Allows selecting multiple test folders and performing cross-test
comparison, statistical analysis, and best-operating-region detection.
"""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
import streamlit as st

from Attopump_data_visualisation_1.io.onedrive_local import (
    list_run_dirs,
    normalize_root,
    pick_best_csv,
    read_csv_full,
)
from .config import PLOT_BIN_WIDTH_HZ, PLOT_HEIGHT
from .data_processor import (
    bin_by_frequency,
    detect_test_type,
    detect_time_format,
    guess_signal_column,
    guess_time_column,
    parse_sweep_spec_from_name,
    prepare_sweep_data,
    prepare_time_series_data,
)
from .analysis_plots import (
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
from .plot_generator import export_html


# ────────────────────────────────────────────────────────────────────────────
# Cached loader
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _load_csv_cached(csv_path_str: str) -> pd.DataFrame:
    """Load a CSV with caching (key = path string)."""
    return read_csv_full(csv_path_str)


# ────────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: C901  — unavoidable complexity for a full page
    """Entry point for the Comprehensive Analysis page."""
    try:
        st.title("🔬 Comprehensive Analysis")
        st.markdown(
            "Compare multiple tests side-by-side. "
            "Select **≥ 2 tests** for cross-test analyses."
        )

        # ================================================================
        # SIDEBAR
        # ================================================================
        with st.sidebar:
            st.header("📁 Data Source")
            if "last_data_path" not in st.session_state:
                st.session_state.last_data_path = ""

            data_folder_str = st.text_input(
                "Path to test data folder",
                value=st.session_state.last_data_path,
                placeholder="/Users/.../All_tests",
                help="Same path as the Explorer page.",
                key="analysis_data_path",
            )

            st.divider()
            st.header("⚙️ Analysis Settings")

            bin_hz = st.slider(
                "Frequency bin width (Hz)",
                min_value=0.5,
                max_value=100.0,
                value=float(PLOT_BIN_WIDTH_HZ),
                step=0.5,
                key="analysis_bin_hz",
                help="Bin width for per-test and combined frequency plots.",
            )
            avg_bin_hz = st.slider(
                "Average-curve bin width (Hz)",
                min_value=0.5,
                max_value=100.0,
                value=3.0,
                step=0.5,
                key="analysis_avg_bin",
                help="Bin width for the global average curve.",
            )
            max_raw_points = st.slider(
                "Max raw points per test",
                min_value=1000,
                max_value=500000,
                value=50000,
                step=5000,
                key="analysis_max_pts",
                help="Cap per test to keep browser responsive.",
            )

            st.divider()
            st.header("🎨 Plot Appearance")
            plot_mode = st.selectbox(
                "Plot type",
                ["lines+markers", "lines", "markers"],
                key="analysis_plot_mode",
            )
            marker_size = st.slider(
                "Marker size", 1, 20, 6, key="analysis_marker_size"
            )
            marker_opacity = st.slider(
                "Opacity", 0.1, 1.0, 0.8, step=0.05, key="analysis_opacity"
            )
            show_error_bars = st.checkbox(
                "Show ±1 std bands", value=True, key="analysis_error_bars"
            )
            export_html_toggle = st.checkbox(
                "Export plots as HTML", value=False, key="analysis_export"
            )

            st.divider()
            st.header("🎯 Stability Cloud")
            mean_pct = st.slider(
                "High-flow threshold (%)",
                50,
                99,
                75,
                key="analysis_mean_pct",
                help="Top N % by mean flow.",
            )
            std_pct = st.slider(
                "Stability threshold (%)",
                1,
                50,
                10,
                key="analysis_std_pct",
                help="Bottom N % by std within the high-flow set.",
            )

        # ================================================================
        # VALIDATE PATH
        # ================================================================
        if not data_folder_str.strip():
            st.info("ℹ️ Enter a data folder path in the sidebar to get started.")
            st.stop()

        try:
            root = normalize_root(data_folder_str)
            if data_folder_str != st.session_state.last_data_path:
                st.session_state.last_data_path = data_folder_str
        except Exception as e:
            st.error(f"❌ Invalid path: {e}")
            st.stop()

        try:
            run_dirs = list(list_run_dirs(root))
        except Exception as e:
            st.error(f"❌ Failed to list folders: {e}")
            st.stop()

        if not run_dirs:
            st.error("❌ No subfolders found.")
            st.stop()

        # ================================================================
        # TEST SELECTION
        # ================================================================
        run_names = [p.name for p in run_dirs]

        # Quick helpers: select-all / clear
        c_sel1, c_sel2, c_sel3 = st.columns([3, 0.6, 0.6])
        with c_sel2:
            if st.button("Select all", key="analysis_sel_all"):
                st.session_state["analysis_multiselect"] = run_names
                st.rerun()
        with c_sel3:
            if st.button("Clear", key="analysis_sel_clear"):
                st.session_state["analysis_multiselect"] = []
                st.rerun()

        with c_sel1:
            selected_names = st.multiselect(
                "📂 Select tests to compare",
                options=run_names,
                default=[],
                help="Pick 2+ tests for cross-comparison.",
                key="analysis_multiselect",
            )

        if not selected_names:
            st.info("👆 Select at least one test folder above.")
            st.stop()

        # ================================================================
        # LOAD ALL SELECTED TESTS
        # ================================================================
        with st.spinner(f"Loading {len(selected_names)} test(s)…"):
            # all_data   – every test's time-series DataFrame (for EDA)
            # sweep_data – tests with Frequency column (for freq plots)
            # binned_data – binned versions of sweep_data
            all_data: dict[str, pd.DataFrame] = {}
            sweep_data: dict[str, pd.DataFrame] = {}
            binned_data: dict[str, pd.DataFrame] = {}
            load_errors: list[str] = []
            signal_col_used: str | None = None

            for name in selected_names:
                run_dir = run_dirs[run_names.index(name)]
                try:
                    pick = pick_best_csv(run_dir)
                    df = _load_csv_cached(str(pick.csv_path))

                    if df.empty:
                        load_errors.append(f"{name}: empty CSV")
                        continue

                    # Detect columns
                    time_col = guess_time_column(df)
                    sig_col = guess_signal_column(df, "flow")
                    if not time_col or not sig_col:
                        load_errors.append(f"{name}: cannot detect columns")
                        continue

                    if signal_col_used is None:
                        signal_col_used = sig_col

                    # Prepare basic time-series data
                    time_fmt = detect_time_format(df, time_col)
                    ts_df = prepare_time_series_data(
                        df,
                        time_col,
                        sig_col,
                        parse_time=(time_fmt == "absolute_timestamp"),
                    )
                    all_data[name] = ts_df

                    # If the CSV has actual frequency data → sweep analysis
                    has_freq = "freq_set_hz" in df.columns
                    if has_freq:
                        spec = parse_sweep_spec_from_name(name)
                        sweep_df = prepare_sweep_data(
                            ts_df,
                            time_col,
                            sig_col,
                            spec=spec,
                            parse_time=(time_fmt == "absolute_timestamp"),
                            full_df=df,
                        )
                        sweep_data[name] = sweep_df

                        # Bin
                        try:
                            binned = bin_by_frequency(
                                sweep_df,
                                value_col=sig_col,
                                freq_col="Frequency",
                                bin_hz=float(bin_hz),
                            )
                            binned_data[name] = binned
                        except Exception as be:
                            load_errors.append(f"{name}: binning — {be}")
                    else:
                        # No frequency column — still try sweep via regex
                        spec = parse_sweep_spec_from_name(name)
                        if spec and spec.duration_s > 0:
                            try:
                                sweep_df = prepare_sweep_data(
                                    ts_df,
                                    time_col,
                                    sig_col,
                                    spec=spec,
                                    parse_time=(time_fmt == "absolute_timestamp"),
                                )
                                sweep_data[name] = sweep_df
                                binned = bin_by_frequency(
                                    sweep_df,
                                    value_col=sig_col,
                                    freq_col="Frequency",
                                    bin_hz=float(bin_hz),
                                )
                                binned_data[name] = binned
                            except Exception:
                                pass  # best-effort

                except Exception as e:
                    load_errors.append(f"{name}: {e}")

        # ── Load status ─────────────────────────────────────────
        n_all = len(all_data)
        n_sweep = len(sweep_data)
        n_binned = len(binned_data)

        if load_errors:
            with st.expander(f"⚠️ {len(load_errors)} load error(s)", expanded=False):
                for err in load_errors:
                    st.warning(err)

        if n_all == 0:
            st.error("❌ No tests loaded successfully.")
            st.stop()

        st.success(
            f"✅ Loaded **{n_all}** test(s) — "
            f"**{n_binned}** with frequency data for sweep analysis"
        )

        if signal_col_used is None:
            signal_col_used = "flow"

        # ================================================================
        # TABS
        # ================================================================
        tabs = st.tabs(
            [
                "📊 Individual",
                "🔀 Combined",
                "📏 Average",
                "⚡ Raw Points",
                "📦 EDA",
                "📐 Std vs Mean",
                "🎯 Best Region",
            ]
        )

        # ── TAB 1: Individual per-test plots ────────────────────
        with tabs[0]:
            st.subheader("Individual Test Results")
            if not binned_data:
                st.info(
                    "No tests with frequency data. "
                    "Only tests containing a `freq_set_hz` column "
                    "are shown here."
                )
            else:
                for name in binned_data:
                    with st.expander(
                        f"📈 {name}", expanded=(len(binned_data) <= 3)
                    ):
                        from .plot_generator import plot_sweep_binned

                        binned = binned_data[name]
                        fig_b = plot_sweep_binned(
                            binned,
                            title=f"{name} — Binned Mean ± Std (Δf = {bin_hz:g} Hz)",
                            mode=plot_mode,
                            marker_size=marker_size,
                        )
                        st.plotly_chart(fig_b, use_container_width=True)

                        # Per-sweep breakdown
                        if name in sweep_data:
                            fig_sw = plot_per_test_sweeps(
                                sweep_data[name],
                                signal_col=signal_col_used,
                                bin_hz=float(bin_hz),
                                title=f"{name} — Per Sweep",
                                mode=plot_mode,
                                marker_size=max(1, marker_size - 1),
                            )
                            st.plotly_chart(fig_sw, use_container_width=True)

                        # Quick metrics
                        if name in sweep_data and signal_col_used in sweep_data[name].columns:
                            s = sweep_data[name][signal_col_used].dropna()
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Mean", f"{s.mean():.2f} µL/min")
                            c2.metric("Std", f"{s.std():.2f} µL/min")
                            c3.metric("Min", f"{s.min():.2f} µL/min")
                            c4.metric("Max", f"{s.max():.2f} µL/min")

                        if export_html_toggle:
                            try:
                                path = export_html(fig_b, f"{name}_binned.html")
                                st.caption(f"✅ Exported: {path.name}")
                            except Exception:
                                pass

        # ── TAB 2: Combined overlay ─────────────────────────────
        with tabs[1]:
            st.subheader("Combined Frequency Sweep Comparison")
            if len(binned_data) < 1:
                st.info("Need at least 1 test with frequency data.")
            else:
                fig_comb = plot_combined_overlay(
                    binned_data,
                    show_error_bars=show_error_bars,
                    mode=plot_mode,
                    marker_size=marker_size,
                )
                st.plotly_chart(fig_comb, use_container_width=True)

                if export_html_toggle:
                    try:
                        path = export_html(fig_comb, "combined_overlay.html")
                        st.success(f"✅ Exported: {path.name}")
                    except Exception:
                        pass

                # Relative comparison
                st.divider()
                st.subheader("Relative (0–100 %) Comparison")
                st.caption(
                    "Each test's mean flow normalized to its own 0–100 % range."
                )
                fig_rel = plot_relative_comparison(
                    binned_data, mode=plot_mode, marker_size=marker_size
                )
                st.plotly_chart(fig_rel, use_container_width=True)

        # ── TAB 3: Global average ───────────────────────────────
        with tabs[2]:
            st.subheader("Global Average Across Tests")
            if len(binned_data) < 2:
                st.info("Need ≥ 2 tests with frequency data for averaging.")
            else:
                fig_avg, avg_df = plot_global_average(
                    binned_data,
                    bin_hz=float(avg_bin_hz),
                    mode=plot_mode,
                    marker_size=marker_size,
                )
                st.plotly_chart(fig_avg, use_container_width=True)

                if export_html_toggle:
                    try:
                        path = export_html(fig_avg, "global_average.html")
                        st.success(f"✅ Exported: {path.name}")
                    except Exception:
                        pass

                with st.expander("📋 Average Curve Data"):
                    st.dataframe(avg_df, use_container_width=True, hide_index=True)

        # ── TAB 4: All raw points ───────────────────────────────
        with tabs[3]:
            st.subheader("All Raw Data Points")
            if not sweep_data:
                st.info("No tests with frequency data.")
            else:
                # Cap per-test for browser performance
                capped_raw: dict[str, pd.DataFrame] = {}
                for name, sdf in sweep_data.items():
                    if len(sdf) > max_raw_points:
                        capped_raw[name] = sdf.sample(
                            n=max_raw_points, random_state=42
                        )
                    else:
                        capped_raw[name] = sdf

                total_pts = sum(len(d) for d in capped_raw.values())
                st.caption(
                    f"Showing {total_pts:,} points "
                    f"(capped at {max_raw_points:,} per test)"
                )

                fig_raw = plot_all_raw_points(
                    capped_raw,
                    freq_col="Frequency",
                    signal_col=signal_col_used,
                    marker_size=marker_size,
                    opacity=marker_opacity,
                )
                st.plotly_chart(fig_raw, use_container_width=True)

                if export_html_toggle:
                    try:
                        path = export_html(fig_raw, "all_raw_points.html")
                        st.success(f"✅ Exported: {path.name}")
                    except Exception:
                        pass

        # ── TAB 5: EDA ──────────────────────────────────────────
        with tabs[4]:
            st.subheader("Exploratory Data Analysis")

            # Summary statistics table
            st.markdown("### 📋 Summary Statistics")
            # Use all_data (includes non-sweep tests) for broad EDA
            stats_df = build_summary_table(all_data, signal_col=signal_col_used)
            if not stats_df.empty:
                st.dataframe(
                    stats_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "CV (%)": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
            else:
                st.info("No signal data available.")

            # Boxplots + Histograms side-by-side
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("### 📦 Boxplots")
                fig_box = plot_combined_boxplots(
                    all_data, signal_col=signal_col_used
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with col_r:
                st.markdown("### 📊 Histograms")
                nbins = st.slider(
                    "Histogram bins", 10, 200, 50, key="eda_hist_bins"
                )
                fig_hist = plot_combined_histograms(
                    all_data, signal_col=signal_col_used, nbins=nbins
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Correlation heatmap
            if len(binned_data) >= 2:
                st.markdown("### 🔗 Inter-Test Correlation")
                st.caption(
                    "Pearson correlation between binned mean-flow curves "
                    "(interpolated onto a common frequency grid)."
                )
                fig_corr = plot_correlation_heatmap(binned_data)
                st.plotly_chart(fig_corr, use_container_width=True)

        # ── TAB 6: Std vs Mean ──────────────────────────────────
        with tabs[5]:
            st.subheader("Variability Analysis: Std vs Mean")
            if not binned_data:
                st.info("No binned frequency data available.")
            else:
                fig_svm = plot_std_vs_mean(
                    binned_data, marker_size=marker_size
                )
                st.plotly_chart(fig_svm, use_container_width=True)

                with st.expander("ℹ️ Interpretation guide"):
                    st.markdown(
                        """
**How to read this plot:**

Each point represents one frequency bin from one test.

| Pattern | Meaning |
|---------|---------|
| **R² ≈ 1** | Variability scales with magnitude (proportional noise) |
| **R² ≈ 0** | Variability is roughly constant regardless of flow |
| Points **below** trend line | More stable than expected at that flow level |
| Points **above** trend line | Less predictable than expected |
| **Steep slope** | Higher flows come with disproportionately more noise |
| **Flat slope** | Noise is similar across all flow levels |
"""
                    )

                if export_html_toggle:
                    try:
                        path = export_html(fig_svm, "std_vs_mean.html")
                        st.success(f"✅ Exported: {path.name}")
                    except Exception:
                        pass

        # ── TAB 7: Best operating region ────────────────────────
        with tabs[6]:
            st.subheader("Best Operating Region Finder")
            if not binned_data:
                st.info("No binned frequency data available.")
            else:
                fig_stab, best_df = plot_stability_cloud(
                    binned_data,
                    mean_threshold_pct=float(mean_pct),
                    std_threshold_pct=float(std_pct),
                    marker_size=marker_size,
                )
                st.plotly_chart(fig_stab, use_container_width=True)

                st.markdown(
                    f"""
**Method:**
1. Pool all frequency bins from every selected test.
2. Select the **top {mean_pct} %** by mean flow → orange markers.
3. Within those, select the **bottom {std_pct} %** by std deviation
   → green stars.

🟢 **Green stars** = best candidates for the highest, most stable flow.
Hover over points to see the test name and frequency.
"""
                )

                if not best_df.empty:
                    st.markdown("### 🏆 Best Operating Points")
                    st.dataframe(
                        best_df[["test", "freq_center", "mean", "std"]]
                        .rename(
                            columns={
                                "test": "Test",
                                "freq_center": "Frequency (Hz)",
                                "mean": "Mean (µL/min)",
                                "std": "Std (µL/min)",
                            }
                        )
                        .sort_values("Mean (µL/min)", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )

                if export_html_toggle:
                    try:
                        path = export_html(fig_stab, "stability_cloud.html")
                        st.success(f"✅ Exported: {path.name}")
                    except Exception:
                        pass

    except Exception as e:
        st.error(f"❌ **CRITICAL ERROR:** {str(e)}")
        with st.expander("🔍 Debug Information"):
            st.code(traceback.format_exc())
