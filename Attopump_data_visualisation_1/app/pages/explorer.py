"""Single Test Explorer page — visualise one test at a time.

This is the default landing page of the application.  The user points
the app at a local folder of test-run sub-directories, selects a run,
and the page auto-detects whether the run is a *constant-frequency* or
*frequency-sweep* test.  It then renders the appropriate charts.

User workflow
-------------
1. **Sidebar → Data Source** — paste a OneDrive-synced root folder path.
2. **Sidebar → Options** — toggle auto-pick CSV, datetime parsing, NaN
   dropping, HTML export, and plot appearance (line/scatter, marker
   size, opacity).
3. **Sidebar → Test Type Override** — manually force constant-frequency
   or sweep detection when auto-detection is wrong.
4. **Sidebar → Naming Conventions** — register individual folders in
   ``test_metadata.json`` or add user-defined regex patterns.
5. **Main area** — select a run folder + CSV, then view:
   - *Constant-frequency*: time series, boxplot, histogram, summary
     metrics.
   - *Sweep*: time-series tab, frequency-analysis tab (all points +
     binned mean ± std).

Inputs
------
- Local folder path (via sidebar text input).
- CSV files discovered by ``io_local.pick_best_csv``.

Outputs
-------
- Interactive Plotly charts rendered in Streamlit.
- Optional HTML exports to ``app/data/exports/``.

Run with: ``streamlit run streamlit_app.py`` (this module is loaded
lazily by the root entry point).
"""

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import traceback

from ..data.io_local import (
    find_csvs,
    list_run_dirs,
    normalize_root,
)
from ..data.config import (
    DEFAULT_CONSTANT_FREQUENCY_HZ,
    MAX_POINTS_DEFAULT,
    PLOT_BIN_WIDTH_HZ,
    SweepSpec,
)
from ..data.data_processor import (
    format_bin_choice_label,
    detect_test_type,
    detect_time_format,
    explain_frequency_bin_recommendation,
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
    recommend_frequency_bin_widths,
    save_metadata_entry,
    save_user_patterns,
    summarize_frequency_holds,
)
from ..data.test_configs import (
    TestConfig,
    delete_test_config,
    get_test_config,
    load_test_configs,
    save_test_config,
)
from ..data.loader import (
    filter_run_names_with_local_csv_cached,
    get_test_cache_context,
    load_csv_cached,
    pick_best_csv_path_cached,
    resolve_data_path,
)
from ..data.persistent_cache import get_or_create_cached_test_figure
from ..data.test_catalog import (
    format_classification_summary,
    format_detection_method,
    rank_test_names,
    resolve_test_record,
)
from ..plots.plot_generator import (
    build_sweep_average_trace,
    downsample_sweep_points,
    export_html,
    plot_constant_frequency_boxplot,
    plot_flow_histogram,
    plot_sweep_all_points,
    plot_sweep_per_sweep_average,
    plot_sweep_binned,
    plot_time_series,
)

# ============================================================================
# MAIN APP LOGIC WITH COMPREHENSIVE ERROR HANDLING
# ============================================================================


def _add_constant_average_overlay(
    fig: go.Figure,
    *,
    const_freq_df,
    time_col: str,
    signal_col: str,
    time_format: str,
    avg_bin_s: float,
) -> go.Figure:
    """Overlay a time-binned mean ± std trace on a constant-test time series."""
    avg_df = const_freq_df[[time_col, signal_col]].dropna().copy()
    if avg_df.empty:
        return fig

    if time_format == "absolute_timestamp":
        t0 = avg_df[time_col].iloc[0]
        elapsed = (avg_df[time_col] - t0).dt.total_seconds()
    else:
        elapsed = avg_df[time_col].astype(float)

    avg_df["_bin"] = (elapsed // avg_bin_s).astype(int)
    binned_avg = (
        avg_df.groupby("_bin")
        .agg(
            mean=(signal_col, "mean"),
            std=(signal_col, "std"),
            t_mid=(time_col, "median"),
        )
        .reset_index()
    )
    binned_avg["std"] = binned_avg["std"].fillna(0)

    fig.add_trace(
        go.Scatter(
            x=binned_avg["t_mid"],
            y=binned_avg["mean"] + binned_avg["std"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup="avg",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=binned_avg["t_mid"],
            y=binned_avg["mean"] - binned_avg["std"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name=f"Avg ±1 std ({avg_bin_s:g}s bins)",
            fillcolor="rgba(255, 100, 0, 0.18)",
            hoverinfo="skip",
            legendgroup="avg",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=binned_avg["t_mid"],
            y=binned_avg["mean"],
            mode="lines",
            name=f"Average ({avg_bin_s:g}s bins)",
            line=dict(color="orangered", width=2.5),
            legendgroup="avg",
        )
    )
    return fig


def _get_cached_test_figure(
    *,
    cache_context: dict[str, object] | None,
    plot_kind: str,
    settings: dict[str, object],
    builder,
):
    """Load one persisted Explorer figure when available."""
    fig, _ = get_or_create_cached_test_figure(
        cache_context=cache_context,
        plot_kind=plot_kind,
        settings=settings,
        builder=builder,
    )
    return fig


def main():
    """Entry point for the Single Test Explorer page.

    Renders the full sidebar configuration, folder/CSV selection,
    test-type detection badge, and appropriate charts.  Called by the
    root ``streamlit_app.py`` via ``st.navigation``.
    """
    try:
        # ========================================================================
        # PAGE SETUP
        # ========================================================================
        st.title("AttoPump Data Visualization")

        # ========================================================================
        # SIDEBAR: DATA SOURCE (shared widget — persists path automatically)
        # ========================================================================
        data_folder_str, _, _ = resolve_data_path(key_suffix="ex", render_widget=False)

        with st.sidebar:
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
            show_error_bars = st.checkbox(
                "Show ±std error bars",
                value=True,
                help="Toggle the ±1 standard-deviation error bars on "
                     "frequency-binned plots.",
                key="show_error_bars_checkbox",
            )
            show_average = st.checkbox(
                "Show average on all-points plot",
                value=False,
                help="Overlay the per-frequency average line on the "
                     "all-points scatter plot.",
                key="show_average_checkbox",
            )
            show_avg_error_bars = st.checkbox(
                "Show ±std on average line",
                value=True,
                help="Show the ±1 std error bars around the average "
                     "line (only visible when average is enabled).",
                key="show_avg_error_bars_checkbox",
            )

            # ── Y-AXIS RANGE ───────────────────────────────────────────
            st.divider()
            st.header("📏 Y-Axis Range")
            y_axis_auto = st.checkbox(
                "Auto-range y-axis",
                value=True,
                help="When checked, the y-axis scales to fit the data. "
                     "Uncheck to set a fixed range.",
                key="y_axis_auto_checkbox",
            )
            if y_axis_auto:
                y_range = None
            else:
                y_min = st.number_input(
                    "Y min", value=0.0, step=0.1,
                    key="y_axis_min_input",
                )
                y_max = st.number_input(
                    "Y max", value=100.0, step=0.1,
                    key="y_axis_max_input",
                )
                y_range = (float(y_min), float(y_max))

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
    or *Frequency Sweep* using a five-level priority system:**

    | Priority | Method | When it fires |
    |----------|--------|---------------|
    | 1 | **`freq_set_hz` column** | The CSV (merged.csv) contains a `freq_set_hz` column. If there is only **1** unique frequency value → *Constant*. If there are **many** → *Sweep*. This is the most reliable method. |
    | 2 | **Experiment log** | The test is listed in the shared `Experiment logs_new format.xlsx` file next to the `All_tests` folder. |
    | 3 | **Metadata file** | The folder name is found in `app/test_metadata.json`, which lists saved corrections and extra metadata. |
    | 4 | **Regex patterns** | The folder name matches one of the built-in sweep patterns (e.g. `1Hz_1500H_Hz_500_seconds`) or a user-defined pattern. |
    | 5 | **Unknown → Constant** | Nothing matched. The app defaults to *Constant* because that is the safest fallback (shows boxplot + histogram). |

    **When to use the override:**
    - A folder name looks like a sweep but was actually run at a constant
      frequency (microcontroller bug).
    - A folder name is completely new and hasn't been added to the
      experiment log or metadata file yet.
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
    experiment log, metadata file, or default to constant.

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
        except Exception as e:
            st.error(f"❌ Invalid path: {e}")
            st.stop()

        try:
            run_dirs = sorted(list_run_dirs(root), key=lambda p: p.name, reverse=True)
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
        run_map = {run_dir.name: run_dir for run_dir in run_dirs}

        run_query = st.text_input(
            "🔎 Search for a test",
            placeholder="Pump 3, 500Hz, 20260224...",
            help="Type a folder fragment, date, pump label, or frequency hint.",
            key="explorer_run_query",
        )
        ranked_run_names = rank_test_names(run_names, run_query, limit=len(run_names))
        if not ranked_run_names:
            st.warning("No matching test folders.")
            st.stop()

        if run_query.strip():
            st.caption(
                f"Showing {len(ranked_run_names)} ranked match(es) out of {len(run_names)} tests."
            )

        selectable_run_names = ranked_run_names
        skipped_run_names: list[str] = []
        if auto_pick_csv:
            available_run_names, unavailable_run_names = (
                filter_run_names_with_local_csv_cached(
                    str(root),
                    tuple(ranked_run_names),
                )
            )
            selectable_run_names = list(available_run_names)
            skipped_run_names = list(unavailable_run_names)

            if not selectable_run_names:
                st.warning(
                    "No matching test folders have a locally available data CSV. "
                    "Download the folder or mark it 'Always Keep on This Device' in OneDrive and retry."
                )
                st.stop()

            if skipped_run_names:
                st.caption(
                    f"Skipped {len(skipped_run_names)} run(s) whose data CSVs are not downloaded locally."
                )

        selected_run_name = st.selectbox("📂 Matching run folders", selectable_run_names)
        run_dir = run_map[selected_run_name]
        cache_context = None

        # Determine CSV file
        if auto_pick_csv:
            try:
                csv_path = Path(pick_best_csv_path_cached(str(run_dir)))
                cache_context = get_test_cache_context(selected_run_name, run_dir)
            except Exception as e:
                st.error(f"❌ Auto-pick failed: {e}")
                st.stop()
        else:
            csvs = find_csvs(run_dir)
            if not csvs:
                st.error(
                    "❌ No locally available data CSV files found in selected folder. "
                    "Download the folder or mark it 'Always Keep on This Device' and retry."
                )
                st.stop()
            csv_choice = st.selectbox("Choose CSV", [p.name for p in csvs])
            csv_path = run_dir / csv_choice

        st.caption(f"📄 {csv_path.name}")

        # ========================================================================
        # LOAD AND PREVIEW DATA
        # ========================================================================
        try:
            df = load_csv_cached(str(csv_path))
        except Exception as e:
            st.error(f"❌ Failed to load CSV: {e}")
            st.stop()

        if df.empty:
            st.error("❌ CSV is empty.")
            st.stop()

        with st.expander("👁️ Preview (first 300 rows)", expanded=False):
            st.dataframe(df.head(300), use_container_width=True)

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
        # ── Load saved test config (if any) ─────────────────────────────
        saved_cfg = get_test_config(selected_run_name)

        # Data-driven detection hierarchy
        detected_type, detection_method, meta_entry = detect_test_type(
            selected_run_name,
            df,
            data_root=root,
        )

        resolved_record = resolve_test_record(
            selected_run_name,
            root,
            run_dir=run_dir,
            df=df,
        )

        st.caption(
            f"{format_classification_summary(resolved_record)}"
            + (
                f" • {resolved_record.pump_bar_id}"
                if resolved_record.pump_bar_id
                else ""
            )
        )

        # ── Prominent warning for unknown tests ────────────────────────
        if detected_type == "unknown" and saved_cfg is None:
            st.warning(
                "⚠️ **This test could not be auto-classified.**  "
                "The folder name doesn't match any known pattern.  "
                "Please define the test type below so the app knows "
                "how to analyse it correctly."
            )

        # ── Test Configuration editor ───────────────────────────────────
        with st.expander("⚙️ Test Configuration (define once, remembered forever)", expanded=(saved_cfg is None and detected_type == "unknown")):
            st.markdown(
                "Define the test type and parameters for **{}**. "
                "This is saved to disk and will be remembered across sessions "
                "and for all users.".format(selected_run_name)
            )

            # Pre-fill from saved config, else from auto-detection
            cfg_type_options = ["constant", "sweep"]
            if saved_cfg:
                cfg_type_default = cfg_type_options.index(saved_cfg.test_type)
            elif detected_type in cfg_type_options:
                cfg_type_default = cfg_type_options.index(detected_type)
            else:
                cfg_type_default = 0

            cfg_type = st.selectbox(
                "Test type",
                options=cfg_type_options,
                index=cfg_type_default,
                format_func=lambda x: "Constant Frequency" if x == "constant" else "Frequency Sweep",
                key="cfg_type_select",
            )

            if cfg_type == "constant":
                default_hz = DEFAULT_CONSTANT_FREQUENCY_HZ
                if saved_cfg and saved_cfg.frequency_hz is not None:
                    default_hz = saved_cfg.frequency_hz
                elif resolved_record.frequency_hz is not None:
                    default_hz = resolved_record.frequency_hz
                elif meta_entry and meta_entry.get("frequency_hz"):
                    default_hz = float(meta_entry["frequency_hz"])

                cfg_freq = st.number_input(
                    "Frequency (Hz)",
                    value=float(default_hz),
                    min_value=0.1,
                    step=10.0,
                    key="cfg_freq_input",
                )
                cfg_start = cfg_end = cfg_dur = None
            else:
                # Sweep defaults: saved_cfg → parsed spec → fallback
                parsed_spec = parse_sweep_spec_from_name(selected_run_name)
                c1, c2, c3 = st.columns(3)
                with c1:
                    cfg_start = st.number_input(
                        "Start frequency (Hz)",
                        value=float(
                            saved_cfg.start_hz if saved_cfg and saved_cfg.start_hz is not None
                            else (
                                resolved_record.sweep_start_hz
                                if resolved_record.sweep_start_hz is not None
                                else (parsed_spec.start_hz if parsed_spec else 1.0)
                            )
                        ),
                        min_value=0.0,
                        step=1.0,
                        key="cfg_start_input",
                    )
                with c2:
                    cfg_end = st.number_input(
                        "End frequency (Hz)",
                        value=float(
                            saved_cfg.end_hz if saved_cfg and saved_cfg.end_hz is not None
                            else (
                                resolved_record.sweep_end_hz
                                if resolved_record.sweep_end_hz is not None
                                else (parsed_spec.end_hz if parsed_spec else 1000.0)
                            )
                        ),
                        min_value=0.0,
                        step=10.0,
                        key="cfg_end_input",
                    )
                with c3:
                    cfg_dur = st.number_input(
                        "Sweep duration (s)",
                        value=float(
                            saved_cfg.duration_s if saved_cfg and saved_cfg.duration_s is not None
                            else (
                                resolved_record.duration_s
                                if resolved_record.duration_s is not None
                                else (parsed_spec.duration_s if parsed_spec else 0.0)
                            )
                        ),
                        min_value=0.0,
                        step=1.0,
                        help="Duration of one sweep cycle. Set to 0 if unknown (will estimate from data).",
                        key="cfg_dur_input",
                    )
                cfg_freq = None

            cfg_note = st.text_input(
                "Note (optional)",
                value=saved_cfg.note if saved_cfg else "",
                key="cfg_note_input",
            )

            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("💾 Save configuration", key="save_cfg_btn"):
                    new_cfg = TestConfig(
                        test_type=cfg_type,
                        frequency_hz=cfg_freq,
                        start_hz=cfg_start,
                        end_hz=cfg_end,
                        duration_s=cfg_dur,
                        note=cfg_note,
                    )
                    save_test_config(selected_run_name, new_cfg)
                    st.success(f"✅ Configuration saved for **{selected_run_name}**")
                    st.rerun()
            with btn_col2:
                if saved_cfg and st.button("🗑️ Delete configuration", key="del_cfg_btn"):
                    delete_test_config(selected_run_name)
                    st.success(f"Deleted configuration for **{selected_run_name}**")
                    st.rerun()

            if saved_cfg:
                st.caption(f"💾 Saved config: **{saved_cfg.test_type}**" + (
                    f" @ {saved_cfg.frequency_hz:g} Hz" if saved_cfg.test_type == "constant" and saved_cfg.frequency_hz else
                    f" {saved_cfg.start_hz:g}→{saved_cfg.end_hz:g} Hz, {saved_cfg.duration_s:g}s" if saved_cfg.test_type == "sweep" and saved_cfg.start_hz is not None else ""
                ))

        # ── Resolve final test type ────────────────────────────────────
        # Priority: manual sidebar override > saved config > auto-detection
        if manual_test_type == "Constant Frequency":
            is_constant_freq_test = True
            detection_badge = "🟡 **Manual override → Constant**"
        elif manual_test_type == "Frequency Sweep":
            is_constant_freq_test = False
            detection_badge = "🟡 **Manual override → Sweep**"
        elif saved_cfg:
            is_constant_freq_test = saved_cfg.test_type == "constant"
            detection_badge = f"💾 **Saved config → {saved_cfg.test_type.title()}**"
        else:
            is_constant_freq_test = detected_type != "sweep"
            method_label = {
                "saved_config":      "💾 saved configuration",
                "freq_set_hz_column": "📊 `freq_set_hz` column",
                "experiment_log":    "📝 experiment log",
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

        if (
            resolved_record.pump_bar_id
            or resolved_record.date
            or resolved_record.author
            or resolved_record.voltage
            or resolved_record.note
        ):
            with st.expander("📝 Resolved Test Details", expanded=False):
                st.write(f"**Classification:** {format_classification_summary(resolved_record)}")
                st.write(f"**Detected via:** {format_detection_method(resolved_record.detection_method)}")
                if resolved_record.pump_bar_id:
                    st.write(f"**Pump / BAR:** {resolved_record.pump_bar_id}")
                if resolved_record.date:
                    st.write(f"**Date:** {resolved_record.date}")
                if resolved_record.time:
                    st.write(f"**Time:** {resolved_record.time}")
                if resolved_record.author:
                    st.write(f"**Author:** {resolved_record.author}")
                if resolved_record.voltage:
                    st.write(f"**Voltage:** {resolved_record.voltage}")
                if resolved_record.success:
                    st.write(f"**Result:** {resolved_record.success}")
                if resolved_record.duration_s:
                    st.write(f"**Duration:** {resolved_record.duration_s:g} s")
                if resolved_record.note:
                    st.write(f"**Notes:** {resolved_record.note}")

        if is_constant_freq_test:
            # ====================================================================
            # CONSTANT FREQUENCY TEST
            # ====================================================================
            st.subheader("📊 Constant Frequency Test Analysis")

            # Determine default frequency from saved config / metadata
            default_freq = DEFAULT_CONSTANT_FREQUENCY_HZ
            if saved_cfg and saved_cfg.test_type == "constant" and saved_cfg.frequency_hz is not None:
                default_freq = saved_cfg.frequency_hz
            elif resolved_record.frequency_hz is not None:
                default_freq = resolved_record.frequency_hz
            elif meta_entry and meta_entry.get("frequency_hz"):
                default_freq = float(meta_entry["frequency_hz"])
            manual_freq = default_freq  # used internally by prepare_constant_frequency_data

            # Options row
            opt1, opt2, opt3 = st.columns(3)
            with opt1:
                hist_bins = st.number_input(
                    "📊 Histogram bins",
                    value=30,
                    min_value=5,
                    max_value=200,
                    step=5,
                    key="hist_bins_input",
                )
            with opt2:
                show_const_avg = st.checkbox(
                    "Show time-binned average",
                    value=False,
                    help="Overlay a running average computed over "
                         "time bins of the width you set.",
                    key="const_avg_toggle",
                )
            with opt3:
                avg_bin_s = st.number_input(
                    "Average bin width (s)",
                    value=5.0,
                    min_value=0.5,
                    max_value=120.0,
                    step=0.5,
                    help="Width of each time bin in seconds for the "
                         "rolling average overlay.",
                    key="const_avg_bin_s",
                    disabled=not show_const_avg,
                )

            # Prepare data
            try:
                const_freq_df = prepare_constant_frequency_data(
                    df,
                    time_col=time_col,
                    signal_col=signal_col,
                    frequency_hz=float(manual_freq),
                    parse_time=(time_format == "absolute_timestamp"),
                    drop_na=drop_na_toggle,
                )
            except Exception as e:
                st.error(f"❌ Failed to prepare constant frequency data: {e}")
                st.stop()

            # ── 1) LARGE TIME SERIES (full width) ──
            try:
                fig_ts = _get_cached_test_figure(
                    cache_context=cache_context,
                    plot_kind="explorer_constant_time_series",
                    settings={
                        "signal_col": signal_col,
                        "x_col": time_col,
                        "mode": plot_mode,
                        "marker_size": int(marker_size),
                        "opacity": float(marker_opacity),
                        "show_const_avg": bool(show_const_avg),
                        "avg_bin_s": float(avg_bin_s),
                        "time_format": time_format,
                        "y_range": list(y_range) if y_range is not None else None,
                    },
                    builder=lambda: _add_constant_average_overlay(
                        plot_time_series(
                            const_freq_df,
                            x_col=time_col,
                            y_col=signal_col,
                            title=f"{selected_run_name} — Flow vs Time",
                            mode=plot_mode,
                            marker_size=marker_size,
                            opacity=marker_opacity,
                            y_range=y_range,
                        ),
                        const_freq_df=const_freq_df,
                        time_col=time_col,
                        signal_col=signal_col,
                        time_format=time_format,
                        avg_bin_s=float(avg_bin_s),
                    )
                    if show_const_avg
                    else plot_time_series(
                        const_freq_df,
                        x_col=time_col,
                        y_col=signal_col,
                        title=f"{selected_run_name} — Flow vs Time",
                        mode=plot_mode,
                        marker_size=marker_size,
                        opacity=marker_opacity,
                        y_range=y_range,
                    ),
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
                    fig_box = _get_cached_test_figure(
                        cache_context=cache_context,
                        plot_kind="explorer_constant_boxplot",
                        settings={
                            "signal_col": signal_col,
                            "title_frequency_hz": float(default_freq),
                        },
                        builder=lambda: plot_constant_frequency_boxplot(
                            const_freq_df,
                            y_col=signal_col,
                            title=f"Flow @ {default_freq:g} Hz",
                        ),
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
                    fig_hist = _get_cached_test_figure(
                        cache_context=cache_context,
                        plot_kind="explorer_constant_histogram",
                        settings={
                            "signal_col": signal_col,
                            "nbins": int(hist_bins),
                        },
                        builder=lambda: plot_flow_histogram(
                            const_freq_df,
                            y_col=signal_col,
                            nbins=int(hist_bins),
                            title=f"{selected_run_name} — Flow Distribution",
                        ),
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
            
            # Build sweep spec: saved config → folder-name regex → None
            if saved_cfg and saved_cfg.test_type == "sweep" and saved_cfg.start_hz is not None:
                sweep_spec = SweepSpec(
                    start_hz=saved_cfg.start_hz,
                    end_hz=saved_cfg.end_hz or 1000.0,
                    duration_s=saved_cfg.duration_s or 0.0,
                )
                st.caption(f"💾 Sweep (saved config): {sweep_spec}")
            else:
                sweep_spec = parse_sweep_spec_from_name(selected_run_name)
                if sweep_spec:
                    st.caption(f"🔄 Sweep (parsed from name): {sweep_spec}")
                else:
                    st.warning("⚠️ Could not determine sweep parameters. Open the **Test Configuration** section above to define them, or the app will use the `freq_set_hz` column if available.")

            # Prepare data early so the bin recommendation can seed the slider.
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

            sweep_plot_df = (
                sweep_df.loc[~sweep_df["IsFrequencyHold"]].copy()
                if "IsFrequencyHold" in sweep_df.columns
                else sweep_df.copy()
            )
            if sweep_plot_df.empty:
                sweep_plot_df = sweep_df.copy()

            hold_summary = summarize_frequency_holds(
                sweep_df,
                signal_col=signal_col,
            )
            overlay_bin_reco = recommend_frequency_bin_widths(
                [sweep_df],
                value_col=signal_col,
            )
            bin_signature = (
                selected_run_name,
                repr(sweep_spec),
                len(sweep_df),
                int(sweep_df["IsFrequencyHold"].sum())
                if "IsFrequencyHold" in sweep_df.columns
                else 0,
            )
            if st.session_state.get("_explorer_bin_signature") != bin_signature:
                st.session_state["_explorer_bin_signature"] = bin_signature
                st.session_state["explorer_binned_bin_hz"] = float(PLOT_BIN_WIDTH_HZ)
                st.session_state["explorer_average_bin_hz"] = float(PLOT_BIN_WIDTH_HZ)
                if int(overlay_bin_reco.get("test_series_count", 0)) >= 2:
                    st.session_state["explorer_average_bin_hz"] = float(
                        overlay_bin_reco["test_bin_hz"]
                    )
                    st.session_state["_explorer_average_bin_recommendation"] = overlay_bin_reco
                else:
                    st.session_state.pop("_explorer_average_bin_recommendation", None)

            def _reset_explorer_average_bin() -> None:
                recommendation = st.session_state.get("_explorer_average_bin_recommendation")
                if recommendation and int(recommendation.get("test_series_count", 0)) >= 2:
                    st.session_state["explorer_average_bin_hz"] = float(
                        recommendation["test_bin_hz"]
                    )

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

            # Create tabs
            tab_ts, tab_freq = st.tabs(["⏱️ Time Series", "📊 Frequency Analysis"])

            # TIME SERIES TAB
            with tab_ts:
                try:
                    fig_ts = _get_cached_test_figure(
                        cache_context=cache_context,
                        plot_kind="explorer_sweep_time_series",
                        settings={
                            "signal_col": signal_col,
                            "x_col": time_col,
                            "mode": plot_mode,
                            "marker_size": int(marker_size),
                            "opacity": float(marker_opacity),
                            "y_range": list(y_range) if y_range is not None else None,
                        },
                        builder=lambda: plot_time_series(
                            ts_df,
                            x_col=time_col,
                            y_col=signal_col,
                            title=f"{selected_run_name} — Flow vs Time",
                            mode=plot_mode,
                            marker_size=marker_size,
                            opacity=marker_opacity,
                            y_range=y_range,
                        ),
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
                # ── Sweep visibility controls ────────────────────────────
                all_sweep_ids = (
                    sorted(sweep_plot_df["Sweep"].unique())
                    if "Sweep" in sweep_plot_df.columns
                    else []
                )
                all_sweep_labels = [f"Sweep {int(s) + 1}" for s in all_sweep_ids]

                if len(all_sweep_ids) > 1:
                    # Callbacks must run BEFORE the widget is instantiated
                    def _hide_all():
                        st.session_state["visible_sweeps_multiselect"] = []

                    def _show_all():
                        st.session_state["visible_sweeps_multiselect"] = all_sweep_labels

                    vis_col1, vis_col2, vis_col3 = st.columns([3, 1, 1])
                    with vis_col2:
                        st.button("🔲 Hide all", key="hide_all_sweeps", on_click=_hide_all)
                    with vis_col3:
                        st.button("✅ Show all", key="show_all_sweeps", on_click=_show_all)
                    with vis_col1:
                        selected_labels = st.multiselect(
                            "Visible sweeps",
                            options=all_sweep_labels,
                            default=all_sweep_labels,
                            key="visible_sweeps_multiselect",
                            help="Select which sweeps to display. "
                                 "You can also click legend items in the plot to toggle.",
                        )

                    # Convert selected labels back to 0-based indices
                    visible_sweep_set: set[int] | None = {
                        all_sweep_ids[all_sweep_labels.index(lbl)]
                        for lbl in selected_labels
                        if lbl in all_sweep_labels
                    }
                else:
                    visible_sweep_set = None  # single sweep → always visible

                explorer_avg_reco = st.session_state.get("_explorer_average_bin_recommendation")
                average_bin_hz = float(
                    st.session_state.get("explorer_average_bin_hz", float(PLOT_BIN_WIDTH_HZ))
                )
                if show_average:
                    avg_ctrl_col, avg_btn_col = st.columns([3, 1])
                    with avg_ctrl_col:
                        average_bin_hz = st.slider(
                            "Average overlay bin width (Hz)",
                            min_value=0.5,
                            max_value=100.0,
                            value=float(PLOT_BIN_WIDTH_HZ),
                            step=0.5,
                            key="explorer_average_bin_hz",
                            help=(
                                "Used only for the black average/error-bar overlay in the raw "
                                "all-sweeps plot below."
                            ),
                        )
                        if explorer_avg_reco and int(explorer_avg_reco.get("test_series_count", 0)) >= 2:
                            st.caption(
                                "Recommended for the overlaid sweep average: "
                                + explain_frequency_bin_recommendation(
                                    explorer_avg_reco,
                                    include_average_bin=False,
                                )
                            )
                    with avg_btn_col:
                        if explorer_avg_reco and int(explorer_avg_reco.get("test_series_count", 0)) >= 2:
                            st.button(
                                "Reset overlay bin",
                                key="explorer_reset_average_bin",
                                on_click=_reset_explorer_average_bin,
                            )

                average_per_sweep = st.checkbox(
                    "Average each sweep separately in the second plot",
                    value=(len(all_sweep_ids) > 1),
                    key="explorer_average_per_sweep",
                    help=(
                        "When enabled, the averaged plot bins each sweep separately and "
                        "shows one mean ± std trace per sweep instead of one combined mean "
                        "across all sweeps."
                    ),
                )

                # ── Compute average overlay (full dataset, not capped) ───
                avg_df = None
                if show_average:
                    try:
                        avg_df = build_sweep_average_trace(
                            sweep_plot_df,
                            signal_col=signal_col,
                            bin_hz=float(average_bin_hz),
                        )
                        avg_df = avg_df.rename(columns={"freq_center": "freq"})
                        avg_df = avg_df[[col for col in ["freq", "mean", "std"] if col in avg_df.columns]]
                    except Exception:
                        avg_df = None

                # ── All points plot (capped for browser performance) ─────
                pts_df = downsample_sweep_points(
                    sweep_plot_df,
                    max_points=int(max_points),
                    sweep_col="Sweep",
                )

                try:
                    points_caption = (
                        "This figure shows raw sweep points for all visible sweeps. "
                        "If the browser point cap is lower than the full dataset, the points are "
                        "downsampled evenly across sweeps so every sweep stays represented. "
                        "The average overlay uses the *Average overlay bin width* control above."
                    )
                    if not hold_summary.empty:
                        points_caption += (
                            f" Removed {int(sweep_df['IsFrequencyHold'].sum()):,} hold-point samples "
                            "from long constant-frequency pauses in the merged sweep."
                        )
                    st.caption(points_caption)
                    fig_pts = _get_cached_test_figure(
                        cache_context=cache_context,
                        plot_kind="explorer_sweep_all_points",
                        settings={
                            "signal_col": signal_col,
                            "max_points": int(max_points),
                            "mode": (
                                "markers"
                                if len(all_sweep_ids) > 1
                                else (plot_mode if plot_mode != "lines" else "markers")
                            ),
                            "marker_size": int(marker_size),
                            "opacity": float(marker_opacity),
                            "y_range": list(y_range) if y_range is not None else None,
                            "visible_sweeps": (
                                sorted(int(sw) for sw in visible_sweep_set)
                                if visible_sweep_set is not None
                                else None
                            ),
                            "show_average": bool(show_average and avg_df is not None and not avg_df.empty),
                            "show_average_error_bars": bool(show_avg_error_bars),
                            "average_bin_hz": float(average_bin_hz) if show_average else None,
                        },
                        builder=lambda: plot_sweep_all_points(
                            pts_df,
                            x_col="Frequency",
                            y_col=signal_col,
                            color_col="Sweep",
                            title=(
                                f"{selected_run_name} — All Sweep Points "
                                f"({len(pts_df):,} of {len(sweep_plot_df):,})"
                            ),
                            mode="markers" if len(all_sweep_ids) > 1 else (
                                plot_mode if plot_mode != "lines" else "markers"
                            ),
                            marker_size=marker_size,
                            opacity=marker_opacity,
                            y_range=y_range,
                            visible_sweeps=visible_sweep_set,
                            average_df=avg_df,
                            show_average_error_bars=show_avg_error_bars,
                        ),
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

                # ── Binned plot ──────────────────────────────────────────
                binned_bin_hz = st.slider(
                    "Averaged plot bin width (Hz)",
                    min_value=0.5,
                    max_value=100.0,
                    value=float(PLOT_BIN_WIDTH_HZ),
                    step=0.5,
                    key="explorer_binned_bin_hz",
                    help=(
                        "Used only for the averaged plot below. Smaller bins preserve more "
                        "detail; larger bins smooth more."
                    ),
                )
                combined_average_df = None
                try:
                    combined_average_df = build_sweep_average_trace(
                        sweep_plot_df,
                        signal_col=signal_col,
                        bin_hz=float(binned_bin_hz),
                    )
                except Exception as e:
                    st.error(f"❌ Averaging sweeps failed: {e}")

                if average_per_sweep:
                    try:
                        fig_bin = _get_cached_test_figure(
                            cache_context=cache_context,
                            plot_kind="explorer_sweep_per_sweep_average",
                            settings={
                                "signal_col": signal_col,
                                "bin_hz": float(binned_bin_hz),
                                "mode": plot_mode,
                                "marker_size": int(marker_size),
                                "show_error_bars": bool(show_error_bars),
                                "y_range": list(y_range) if y_range is not None else None,
                                "visible_sweeps": (
                                    sorted(int(sw) for sw in visible_sweep_set)
                                    if visible_sweep_set is not None
                                    else None
                                ),
                            },
                            builder=lambda: plot_sweep_per_sweep_average(
                                sweep_plot_df,
                                signal_col=signal_col,
                                bin_hz=float(binned_bin_hz),
                                title=(
                                    f"{selected_run_name} — Per-Sweep Mean ± Std Error "
                                    f"({format_bin_choice_label(float(binned_bin_hz), None)})"
                                ),
                                mode=plot_mode,
                                marker_size=marker_size,
                                show_error_bars=show_error_bars,
                                y_range=y_range,
                                visible_sweeps=visible_sweep_set,
                            ),
                        )
                        st.caption(
                            "This figure bins each sweep separately and plots the mean flow in "
                            "each frequency bin. Error bars show the within-sweep standard "
                            "deviation inside each bin."
                        )
                        st.plotly_chart(fig_bin, use_container_width=True)

                        if export_html_toggle:
                            try:
                                path = export_html(
                                    fig_bin,
                                    f"{selected_run_name}_sweep_per_sweep_average.html",
                                )
                                st.success(f"✅ Exported: {path.name}")
                            except Exception as e:
                                st.error(f"❌ Export failed: {e}")
                    except Exception as e:
                        st.error(f"❌ Per-sweep averaged plot failed: {e}")
                if combined_average_df is not None and not combined_average_df.empty:
                    n_bins = len(combined_average_df)
                    try:
                        fig_avg = _get_cached_test_figure(
                            cache_context=cache_context,
                            plot_kind="explorer_sweep_average",
                            settings={
                                "signal_col": signal_col,
                                "bin_hz": float(binned_bin_hz),
                                "mode": plot_mode,
                                "marker_size": int(marker_size),
                                "show_error_bars": bool(show_error_bars),
                                "y_range": list(y_range) if y_range is not None else None,
                            },
                            builder=lambda: plot_sweep_binned(
                                combined_average_df,
                                x_col="freq_center",
                                y_col="mean",
                                std_col="std",
                                title=(
                                    f"{selected_run_name} — Average Across All Sweeps "
                                    f"({n_bins} bins, {format_bin_choice_label(float(binned_bin_hz), None)})"
                                ),
                                mode=plot_mode,
                                marker_size=marker_size,
                                show_error_bars=show_error_bars,
                                y_range=y_range,
                            ),
                        )
                        st.caption(
                            "This single line is the per-frequency average of the individual "
                            "sweep curves, so each sweep contributes equally at a given "
                            "frequency. Its error bars show the sweep-to-sweep spread."
                        )
                        st.plotly_chart(fig_avg, use_container_width=True)

                        if export_html_toggle:
                            try:
                                path = export_html(fig_avg, f"{selected_run_name}_sweep_average.html")
                                st.success(f"✅ Exported: {path.name}")
                            except Exception as e:
                                st.error(f"❌ Export failed: {e}")
                    except Exception as e:
                        st.error(f"❌ Combined average plot failed: {e}")

                    with st.expander("📋 Averaged Data Table", expanded=False):
                        st.caption(
                            f"Each row is one frequency bin of width "
                            f"**{binned_bin_hz:g} Hz**. These values come from averaging the "
                            "per-sweep binned curves onto a common frequency grid."
                        )
                        st.dataframe(combined_average_df, use_container_width=True)

                if not hold_summary.empty:
                    with st.expander(
                        "⏸️ Constant-frequency hold segments removed from the frequency plots",
                        expanded=False,
                    ):
                        st.caption(
                            "These long constant-frequency pauses were detected in the merged "
                            "microcontroller data and excluded from the sweep-frequency plots "
                            "so they do not dominate setpoints such as 500 Hz."
                        )
                        st.dataframe(hold_summary, use_container_width=True, hide_index=True)

                # ── Per-sweep average summary table ──────────────────────
                import pandas as pd
                if "Sweep" in sweep_plot_df.columns and signal_col in sweep_plot_df.columns:
                    sweep_ids = sorted(sweep_plot_df["Sweep"].unique())
                    if len(sweep_ids) > 0:
                        rows = []
                        for sw in sweep_ids:
                            sub = sweep_plot_df.loc[
                                sweep_plot_df["Sweep"] == sw,
                                signal_col,
                            ].dropna()
                            if sub.empty:
                                continue
                            rows.append({
                                "Sweep": f"Sweep {int(sw) + 1}",
                                "Mean (µL/min)": round(float(sub.mean()), 3),
                                "Std (µL/min)": round(float(sub.std()), 3),
                                "Min (µL/min)": round(float(sub.min()), 3),
                                "Max (µL/min)": round(float(sub.max()), 3),
                                "Points": len(sub),
                            })
                        if rows:
                            summary_df = pd.DataFrame(rows)
                            st.subheader("📊 Per-Sweep Summary")
                            st.dataframe(
                                summary_df,
                                use_container_width=True,
                                hide_index=True,
                            )

    except Exception as e:
        st.error(f"❌ **CRITICAL ERROR:** {str(e)}")
        with st.expander("🔍 Debug Information"):
            st.code(traceback.format_exc())
