"""Streamlit app for AttoPump data visualization.

Run with: streamlit run streamlit_app.py

Modular architecture:
  - config.py: Constants and patterns
  - data_processor.py: Data loading, cleaning, sweep/constant frequency analysis
  - plot_generator.py: Plotly figure generation and HTML export
"""

import streamlit as st
import traceback

from Attopump_data_visualisation_1.io.onedrive_local import (
    list_run_dirs,
    normalize_root,
    pick_best_csv,
    read_csv_full,
    read_csv_preview,
)
from .config import (
    CONSTANT_FREQUENCY_CUTOFF,
    DEFAULT_CONSTANT_FREQUENCY_HZ,
    MAX_POINTS_DEFAULT,
    PLOT_BIN_WIDTH_HZ,
)
from .data_processor import (
    bin_by_frequency,
    detect_time_format,
    get_signal_columns,
    guess_signal_column,
    guess_time_column,
    is_constant_frequency_test,
    parse_sweep_spec_from_name,
    prepare_constant_frequency_data,
    prepare_sweep_data,
    prepare_time_series_data,
)
from .plot_generator import (
    export_html,
    plot_constant_frequency_boxplot,
    plot_flow_histogram,
    plot_sweep_all_points,
    plot_sweep_binned,
    plot_time_series,
)

# ============================================================================
# MAIN APP LOGIC WITH COMPREHENSIVE ERROR HANDLING
# ============================================================================

try:
    # ========================================================================
    # SESSION STATE: Path Memory
    # ========================================================================
    if "last_data_path" not in st.session_state:
        st.session_state.last_data_path = ""

    # ========================================================================
    # PAGE SETUP
    # ========================================================================
    st.set_page_config(page_title="AttoPump Data Visualization", layout="wide")
    st.title("AttoPump Data Visualization")

    # ========================================================================
    # SIDEBAR: INPUT
    # ========================================================================
    with st.sidebar:
        st.header("📁 Data Source")
        
        # Show last used path as hint
        hint_text = ""
        if st.session_state.last_data_path:
            hint_text = f"Last used: {st.session_state.last_data_path}"
        
        data_folder_str = st.text_input(
            "Path to test data folder",
            value=st.session_state.last_data_path,
            placeholder="/Users/.../All_tests",
            help="Paste path from Terminal (spaces are OK). Will be saved for next session.",
        )

        st.divider()
        st.header("📊 Options")
        auto_pick_csv = st.checkbox("Auto-pick best CSV", value=True, help="Prefer merged.csv, then trimmed_*.csv")
        parse_time_toggle = st.checkbox("Parse time as datetime", value=False, help="Enable for Flowboard timestamps. Disable for merged.csv elapsed seconds.")
        drop_na_toggle = st.checkbox("Drop NaN values", value=True)
        export_html_toggle = st.checkbox("Export plots as HTML", value=False)

    # ========================================================================
    # EARLY VALIDATION: Is folder provided?
    # ========================================================================
    if not data_folder_str.strip():
        st.info("ℹ️ Enter a data folder path above to get started.")
        st.stop()

    # ========================================================================
    # LOAD RUN DIRECTORIES
    # ========================================================================
    try:
        root = normalize_root(data_folder_str)
        # Save path to session state for next time
        if data_folder_str and data_folder_str != st.session_state.last_data_path:
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
        st.error("❌ No subfolders found. Check the path.")
        st.stop()

    # ========================================================================
    # SELECT RUN FOLDER AND CSV
    # ========================================================================
    run_names = [p.name for p in run_dirs]
    selected_run_name = st.selectbox("📂 Select run folder", run_names)
    run_dir = run_dirs[run_names.index(selected_run_name)]

    # Determine CSV file
    if auto_pick_csv:
        try:
            pick_result = pick_best_csv(run_dir)
            csv_path = pick_result.csv_path
        except Exception as e:
            st.error(f"❌ Auto-pick failed: {e}")
            st.stop()
    else:
        csvs = sorted([p for p in run_dir.glob("*.csv") if p.is_file()])
        if not csvs:
            st.error("❌ No CSV files found in selected folder.")
            st.stop()
        csv_choice = st.selectbox("Choose CSV", [p.name for p in csvs])
        csv_path = run_dir / csv_choice

    st.caption(f"📄 {csv_path.name}")

    # ========================================================================
    # LOAD AND PREVIEW DATA
    # ========================================================================
    try:
        preview_df = read_csv_preview(csv_path, nrows=300)
    except Exception as e:
        st.error(f"❌ Failed to read preview: {e}")
        st.stop()

    with st.expander("👁️ Preview (first 300 rows)", expanded=False):
        st.dataframe(preview_df, use_container_width=True)

    # Load full data for analysis
    try:
        df = read_csv_full(csv_path)
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        st.stop()

    if df.empty:
        st.error("❌ CSV is empty.")
        st.stop()

    # Debug: Show column info
    with st.expander("🔍 Debug: Column Info", expanded=False):
        st.write(f"**Total columns:** {len(df.columns)}")
        st.write(f"**Column names:** {list(df.columns)}")
        st.write(f"**Data shape:** {df.shape}")
        st.write(f"**Data types:**")
        st.write(df.dtypes)

    # ========================================================================
    # COLUMN SELECTION
    # ========================================================================
    # Auto-detect time and signal columns
    time_col_guess = guess_time_column(df)
    signal_col_guess = guess_signal_column(df, "flow")  # Get best single match for flow
    signal_cols = get_signal_columns(df, "flow")  # Get all candidates for dropdown

    if not signal_cols:
        st.error("❌ No numeric columns found in CSV.")
        st.write(f"**Available columns:** {list(df.columns)}")
        st.stop()

    if not time_col_guess:
        st.warning("⚠️ Could not auto-detect time column. Please select manually.")
        time_col_guess = df.columns[0]

    col1, col2 = st.columns(2)
    with col1:
        try:
            time_index = list(df.columns).index(time_col_guess) if time_col_guess in df.columns else 0
        except:
            time_index = 0
        
        time_col = st.selectbox(
            "⏱️ Time column",
            options=list(df.columns),
            index=time_index,
        )

    with col2:
        try:
            signal_index = signal_cols.index(signal_col_guess) if signal_col_guess and signal_col_guess in signal_cols else 0
        except:
            signal_index = 0
        
        signal_col = st.selectbox(
            "📈 Signal column (Flow)",
            options=signal_cols,
            index=signal_index,
        )

    # Detect time format
    time_format = detect_time_format(df, time_col)
    st.caption(f"⏱️ Time format: {time_format}")

    # ========================================================================
    # PROCESS DATA & DETERMINE TEST TYPE
    # ========================================================================
    is_constant_freq_test = is_constant_frequency_test(selected_run_name)

    if is_constant_freq_test:
        # ====================================================================
        # CONSTANT FREQUENCY TEST
        # ====================================================================
        st.subheader("📊 Constant Frequency Test Analysis")
        
        # Get frequency from folder name or use manual override
        sweep_spec = parse_sweep_spec_from_name(selected_run_name)
        default_freq = (
            sweep_spec.get("freq_center", DEFAULT_CONSTANT_FREQUENCY_HZ)
            if sweep_spec
            else DEFAULT_CONSTANT_FREQUENCY_HZ
        )
        
        # Manual frequency override
        col1, col2 = st.columns(2)
        with col1:
            manual_freq = st.number_input(
                "📈 Frequency (Hz)",
                value=default_freq,
                min_value=0.1,
                step=10.0,
                help="Override auto-detected frequency"
            )
        with col2:
            st.write("")  # Spacer

        # Prepare data
        try:
            const_freq_df = prepare_constant_frequency_data(
                df,
                time_col=time_col,
                signal_col=signal_col,
                frequency_hz=manual_freq,
                parse_time=False,  # Don't re-parse elapsed seconds
                drop_na=drop_na_toggle,
            )
        except Exception as e:
            st.error(f"❌ Failed to prepare constant frequency data: {e}")
            st.stop()

        # Show boxplot + histogram
        col1, col2 = st.columns(2)

        with col1:
            try:
                fig_box = plot_constant_frequency_boxplot(
                    const_freq_df,
                    x_col="Time_Window",
                    y_col=signal_col,
                    title=f"{selected_run_name} — Flow by Time Window",
                )
                st.plotly_chart(fig_box, use_container_width=True)

                if export_html_toggle:
                    try:
                        path = export_html(fig_box, f"{selected_run_name}_boxplot.html")
                        st.success(f"✅ Exported: {path.name}")
                    except Exception as e:
                        st.error(f"❌ Export failed: {e}")
            except Exception as e:
                st.error(f"❌ Boxplot failed: {e}")

        with col2:
            try:
                fig_hist = plot_flow_histogram(
                    const_freq_df,
                    value_col=signal_col,
                    title=f"{selected_run_name} — Flow Distribution",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                if export_html_toggle:
                    try:
                        path = export_html(fig_hist, f"{selected_run_name}_histogram.html")
                        st.success(f"✅ Exported: {path.name}")
                    except Exception as e:
                        st.error(f"❌ Export failed: {e}")
            except Exception as e:
                st.error(f"❌ Histogram failed: {e}")

        # Summary stats
        with st.expander("📋 Summary Statistics", expanded=False):
            st.write(f"**Mean:** {const_freq_df[signal_col].mean():.4f} µL/min")
            st.write(f"**Std Dev:** {const_freq_df[signal_col].std():.4f} µL/min")
            st.write(f"**Min:** {const_freq_df[signal_col].min():.4f} µL/min")
            st.write(f"**Max:** {const_freq_df[signal_col].max():.4f} µL/min")

    else:
        # ====================================================================
        # FREQUENCY SWEEP TEST
        # ====================================================================
        st.subheader("📈 Frequency Sweep Test Analysis")
        
        # Binning options
        col1, col2 = st.columns(2)
        with col1:
            bin_hz = st.number_input(
                "📊 Frequency bin width (Hz)",
                value=PLOT_BIN_WIDTH_HZ,
                min_value=0.5,
                step=1.0,
            )
        with col2:
            max_points = st.number_input(
                "Max points to plot",
                value=MAX_POINTS_DEFAULT,
                min_value=100,
                step=1000,
            )

        # Prepare data: time series, sweep analysis
        try:
            ts_df = prepare_time_series_data(
                df,
                time_col=time_col,
                signal_col=signal_col,
                parse_time=(time_format == "absolute_timestamp"),
                drop_na=drop_na_toggle,
            )
        except Exception as e:
            st.error(f"❌ Failed to prepare time series data: {e}")
            st.stop()

        try:
            sweep_df = prepare_sweep_data(
                ts_df,
                time_col=time_col,
                signal_col=signal_col,
            )
        except Exception as e:
            st.error(f"❌ Failed to prepare sweep data: {e}")
            st.stop()

        # Create tabs
        tab_ts, tab_freq = st.tabs(["⏱️ Time Series", "📊 Frequency Analysis"])

        # TIME SERIES TAB
        with tab_ts:
            try:
                fig_ts = plot_time_series(
                    ts_df,
                    x_col=time_col,
                    y_col=signal_col,
                    title=f"{selected_run_name} — Flow vs Time",
                )
                st.plotly_chart(fig_ts, use_container_width=True)

                if export_html_toggle:
                    try:
                        path = export_html(fig_ts, f"{selected_run_name}_time_series.html")
                        st.success(f"✅ Exported: {path.name}")
                    except Exception as e:
                        st.error(f"❌ Export failed: {e}")
            except Exception as e:
                st.error(f"❌ Time series plot failed: {e}")

            with st.expander("📋 Time Series Data", expanded=False):
                st.dataframe(ts_df, use_container_width=True)

        # FREQUENCY ANALYSIS TAB
        with tab_freq:
            # All points plot
            pts_df = sweep_df.head(max_points)

            try:
                fig_pts = plot_sweep_all_points(
                    pts_df,
                    x_col="Frequency",
                    y_col=signal_col,
                    color_col="Sweep",
                    title=f"{selected_run_name} — All Points",
                )
                st.plotly_chart(fig_pts, use_container_width=True)

                if export_html_toggle:
                    try:
                        path = export_html(fig_pts, f"{selected_run_name}_sweep_points.html")
                        st.success(f"✅ Exported: {path.name}")
                    except Exception as e:
                        st.error(f"❌ Export failed: {e}")
            except Exception as e:
                st.error(f"❌ All points plot failed: {e}")

            # Binned plot
            try:
                binned = bin_by_frequency(
                    sweep_df,
                    value_col=signal_col,
                    freq_col="Frequency",
                    bin_hz=float(bin_hz),
                )
            except Exception as e:
                st.error(f"❌ Binning failed: {e}")
                st.stop()

            try:
                fig_bin = plot_sweep_binned(
                    binned,
                    x_col="freq_center",
                    y_col="mean",
                    std_col="std",
                    title=f"{selected_run_name} — Binned Mean ± Std",
                )
                st.plotly_chart(fig_bin, use_container_width=True)

                if export_html_toggle:
                    try:
                        path = export_html(fig_bin, f"{selected_run_name}_sweep_binned.html")
                        st.success(f"✅ Exported: {path.name}")
                    except Exception as e:
                        st.error(f"❌ Export failed: {e}")
            except Exception as e:
                st.error(f"❌ Binned plot failed: {e}")

            with st.expander("📋 Binned Data Table", expanded=False):
                st.dataframe(binned, use_container_width=True)

except Exception as e:
    st.error(f"❌ **CRITICAL ERROR:** {str(e)}")
    with st.expander("🔍 Debug Information"):
        st.code(traceback.format_exc())
