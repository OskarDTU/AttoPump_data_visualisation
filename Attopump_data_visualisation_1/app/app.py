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
    DEFAULT_CONSTANT_FREQUENCY_HZ,
    MAX_POINTS_DEFAULT,
    PLOT_BIN_WIDTH_HZ,
)
from .data_processor import (
    bin_by_frequency,
    detect_test_type,
    detect_time_format,
    get_signal_columns,
    guess_signal_column,
    guess_time_column,
    is_constant_frequency_test,
    load_test_metadata,
    load_user_patterns,
    parse_sweep_spec_from_name,
    prepare_constant_frequency_data,
    prepare_sweep_data,
    prepare_time_series_data,
    save_metadata_entry,
    save_user_patterns,
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

        # ── PLOT APPEARANCE ─────────────────────────────────────────
        st.divider()
        st.header("🎨 Plot Appearance")
        plot_mode = st.selectbox(
            "Plot type",
            options=["lines", "markers", "lines+markers"],
            index=0,
            help="Lines = fast overview. Markers = see individual points. "
                 "Lines+markers = both (slower for large datasets).",
            key="plot_mode_select",
        )
        marker_size = st.slider(
            "Marker / point size (px)",
            min_value=1,
            max_value=20,
            value=4,
            step=1,
            key="marker_size_slider",
        )
        marker_opacity = st.slider(
            "Marker opacity",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.05,
            key="marker_opacity_slider",
        )

        # ── TEST TYPE OVERRIDE ──────────────────────────────────────
        st.divider()
        st.header("🔧 Test Type Override")
        manual_test_type = st.selectbox(
            "Override detected test type",
            options=["Auto-detect", "Constant Frequency", "Frequency Sweep"],
            index=0,
            help="Force a test type instead of relying on auto-detection. "
                 "Use this when folder names are misleading.",
        )
        with st.expander("ℹ️ How does test-type detection work?", expanded=False):
            st.markdown(
                """
**The app classifies every test folder as either *Constant Frequency*
or *Frequency Sweep* using a four-level priority system:**

| Priority | Method | When it fires |
|----------|--------|---------------|
| 1 | **`freq_set_hz` column** | The CSV (merged.csv) contains a `freq_set_hz` column. If there is only **1** unique frequency value → *Constant*. If there are **many** → *Sweep*. This is the most reliable method. |
| 2 | **Metadata file** | The folder name is found in `app/test_metadata.json`, which lists the correct type for every known test. You can edit this file manually or use the *Naming Conventions* section below. |
| 3 | **Regex patterns** | The folder name matches one of the built-in sweep patterns (e.g. `1Hz_1500H_Hz_500_seconds`) or a user-defined pattern. |
| 4 | **Unknown → Constant** | Nothing matched. The app defaults to *Constant* because that is the safest fallback (shows boxplot + histogram). |

**When to use the override:**
- A folder name looks like a sweep but was actually run at a constant
  frequency (microcontroller bug).
- A folder name is completely new and hasn't been added to the
  metadata file yet.
- You just want to quickly check how the data looks in the other
  visualisation mode.

Select **Constant Frequency** or **Frequency Sweep** above to bypass
auto-detection for the currently selected folder.
"""
            )

        st.divider()
        # ── NAMING CONVENTION EDITOR ────────────────────────────────
        with st.expander("🏷️ Naming Conventions", expanded=False):
            st.markdown(
                """
### What is this?

When you create a **new test** with a folder name the app has never
seen before, it needs to know whether it is a *constant frequency* or
a *frequency sweep* test.  This section lets you teach the app about
new naming patterns **without editing any code**.

---

### Option A — Register a single folder

Use the form below to permanently record a specific folder's test
type in the metadata file (`app/test_metadata.json`).  This is the
fastest way to fix a misclassified test.

1. Paste the **exact folder name** (e.g. `20260302-1244-PUMP-20260227-4`).
2. Choose **constant** or **sweep**.
3. For constant tests, enter the frequency in Hz.
4. Click **💾 Save to metadata**.

---

### Option B — Add a regex pattern for future folders

If you are about to start using a **new naming scheme** for many
folders, add a regex (regular expression) pattern here.  Any folder
whose name matches one of these patterns will be classified as a
sweep.  Folders that match *none* of the patterns fall back to the
metadata file or default to constant.

**Built-in patterns already handle:**
- `10Hz_500Hz_60s` — standard sweep
- `1Hz_1500H_Hz_500_seconds` — H_Hz typo variant
- `1Hz-1kHz` — kHz shorthand

**Example user pattern:**
```
(?P<start>\\d+)Hz_to_(?P<end>\\d+)Hz_(?P<dur>\\d+)min
```
This would match folder names like `100Hz_to_900Hz_5min`.

Enter one pattern per line and click **💾 Save patterns**.
"""
            )

            existing_patterns = load_user_patterns()
            patterns_text = st.text_area(
                "User sweep patterns (one regex per line)",
                value="\n".join(existing_patterns),
                height=100,
                help="Example: (?P<start>\\d+)Hz_to_(?P<end>\\d+)Hz",
            )
            if st.button("💾 Save patterns"):
                new_patterns = [
                    line.strip()
                    for line in patterns_text.splitlines()
                    if line.strip()
                ]
                save_user_patterns(new_patterns)
                st.success(f"Saved {len(new_patterns)} pattern(s).")

            st.divider()
            st.markdown("#### Register a single folder")
            meta_folder = st.text_input("Folder name", key="meta_folder_input")
            meta_type = st.selectbox(
                "Test type", ["constant", "sweep"], key="meta_type_input"
            )
            meta_freq = st.number_input(
                "Frequency (Hz, for constant)", value=500.0, key="meta_freq_input"
            )
            meta_note = st.text_input("Note", key="meta_note_input")
            if st.button("💾 Save to metadata"):
                if meta_folder.strip():
                    entry: dict = {"type": meta_type, "note": meta_note}
                    if meta_type == "constant":
                        entry["frequency_hz"] = meta_freq
                    save_metadata_entry(meta_folder.strip(), entry)
                    st.success(f"Saved metadata for **{meta_folder.strip()}**.")
                else:
                    st.warning("Enter a folder name first.")

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
    # Data-driven detection hierarchy
    detected_type, detection_method, meta_entry = detect_test_type(selected_run_name, df)

    # Manual override from sidebar
    if manual_test_type == "Constant Frequency":
        is_constant_freq_test = True
        detection_badge = "🟡 **Manual override → Constant**"
    elif manual_test_type == "Frequency Sweep":
        is_constant_freq_test = False
        detection_badge = "🟡 **Manual override → Sweep**"
    else:
        is_constant_freq_test = detected_type != "sweep"
        method_label = {
            "freq_set_hz_column": "📊 `freq_set_hz` column",
            "metadata":          "📋 metadata file",
            "regex":             "🔤 folder-name regex",
            "user_regex":        "🏷️ user regex pattern",
            "unknown":           "❓ unknown (defaulting to constant)",
        }.get(detection_method, detection_method)
        detection_badge = (
            f"{'🟢' if detected_type != 'unknown' else '🟠'} "
            f"Detected: **{detected_type}** via {method_label}"
        )

    st.markdown(detection_badge)

    # Show metadata info if available
    if meta_entry:
        with st.expander("📋 Test Metadata", expanded=False):
            for k, v in meta_entry.items():
                st.write(f"**{k}:** {v}")

    if is_constant_freq_test:
        # ====================================================================
        # CONSTANT FREQUENCY TEST
        # ====================================================================
        st.subheader("📊 Constant Frequency Test Analysis")

        # Determine default frequency: metadata → config fallback
        default_freq = DEFAULT_CONSTANT_FREQUENCY_HZ
        if meta_entry and meta_entry.get("frequency_hz"):
            default_freq = float(meta_entry["frequency_hz"])

        # Options row: frequency + histogram bins
        opt1, opt2 = st.columns(2)
        with opt1:
            manual_freq = st.number_input(
                "📈 Frequency (Hz)",
                value=float(default_freq),
                min_value=0.1,
                step=10.0,
                help="Override auto-detected frequency",
                key="const_freq_input",
            )
        with opt2:
            hist_bins = st.number_input(
                "📊 Histogram bins",
                value=30,
                min_value=5,
                max_value=200,
                step=5,
                key="hist_bins_input",
            )

        # Prepare data
        try:
            const_freq_df = prepare_constant_frequency_data(
                df,
                time_col=time_col,
                signal_col=signal_col,
                frequency_hz=float(manual_freq),
                parse_time=False,
                drop_na=drop_na_toggle,
            )
        except Exception as e:
            st.error(f"❌ Failed to prepare constant frequency data: {e}")
            st.stop()

        # ── 1) LARGE TIME SERIES (full width) ──
        try:
            fig_ts = plot_time_series(
                const_freq_df,
                x_col=time_col,
                y_col=signal_col,
                title=f"{selected_run_name} — Flow vs Time",
                mode=plot_mode,
                marker_size=marker_size,
                opacity=marker_opacity,
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

        # ── 2) BOXPLOT + HISTOGRAM (side-by-side) ──
        col_left, col_right = st.columns(2)

        with col_left:
            try:
                fig_box = plot_constant_frequency_boxplot(
                    const_freq_df,
                    y_col=signal_col,
                    title=f"Flow @ {manual_freq:g} Hz",
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

        with col_right:
            try:
                fig_hist = plot_flow_histogram(
                    const_freq_df,
                    y_col=signal_col,
                    nbins=int(hist_bins),
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

        # ── 3) SUMMARY STATS ──
        with st.expander("📋 Summary Statistics", expanded=True):
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Mean", f"{const_freq_df[signal_col].mean():.2f} µL/min")
            s2.metric("Std Dev", f"{const_freq_df[signal_col].std():.2f} µL/min")
            s3.metric("Min", f"{const_freq_df[signal_col].min():.2f} µL/min")
            s4.metric("Max", f"{const_freq_df[signal_col].max():.2f} µL/min")

    else:
        # ====================================================================
        # FREQUENCY SWEEP TEST
        # ====================================================================
        st.subheader("📈 Frequency Sweep Test Analysis")
        
        # Parse sweep specification from folder name
        sweep_spec = parse_sweep_spec_from_name(selected_run_name)
        if sweep_spec:
            st.caption(f"🔄 Sweep: {sweep_spec}")
        else:
            st.warning("⚠️ Could not parse sweep parameters from folder name. Frequency mapping will use the `freq_set_hz` column if available, or elapsed time as a proxy.")

        # Binning & display options
        col1, col2 = st.columns(2)
        with col1:
            bin_hz = st.slider(
                "📊 Frequency bin width (Hz)",
                min_value=0.5,
                max_value=100.0,
                value=float(PLOT_BIN_WIDTH_HZ),
                step=0.5,
                key="bin_hz_slider",
                help="Width of each frequency bin for the binned plot. "
                     "Smaller = more detail but noisier. Larger = smoother.",
            )
        with col2:
            max_points = st.slider(
                "Max points to plot",
                min_value=1000,
                max_value=500000,
                value=int(MAX_POINTS_DEFAULT),
                step=5000,
                key="max_points_slider",
                help="Limit the number of data points sent to the browser. "
                     "Lower values = faster rendering. The full dataset is "
                     "still used for binned statistics.",
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
                spec=sweep_spec,
                parse_time=(time_format == "absolute_timestamp"),
                full_df=df,
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
                    mode=plot_mode,
                    marker_size=marker_size,
                    opacity=marker_opacity,
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
            # All points plot (capped to max_points for browser performance)
            pts_df = sweep_df.head(int(max_points))

            try:
                fig_pts = plot_sweep_all_points(
                    pts_df,
                    x_col="Frequency",
                    y_col=signal_col,
                    color_col="Sweep",
                    title=f"{selected_run_name} — All Points ({len(pts_df):,} of {len(sweep_df):,})",
                    mode=plot_mode if plot_mode != "lines" else "markers",
                    marker_size=marker_size,
                    opacity=marker_opacity,
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
                binned = None

            if binned is not None:
                try:
                    fig_bin = plot_sweep_binned(
                        binned,
                        x_col="freq_center",
                        y_col="mean",
                        std_col="std",
                        title=f"{selected_run_name} — Binned Mean ± Std (Δf = {bin_hz:g} Hz)",
                        marker_size=marker_size,
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
