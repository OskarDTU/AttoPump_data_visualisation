"""Streamlit entrypoint for quick visualisation of AttoPump test data.

Run locally:
  uv run python -m streamlit run streamlit_app.py

Quicklook:
- Point the app at your test data folder (contains test run subfolders).
- Select a run folder.
- Auto-pick CSV (trimmed_*.csv preferred).
- Plot signal vs time.

Sweep quicklook:
- If the run folder name matches a sweep pattern like `10Hz_500Hz_60s`, the Sweep tab:
  - maps elapsed time -> frequency + sweep index
  - shows all points (colored by sweep)
  - shows binned mean ± std vs frequency
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from Attopump_data_visualisation_1.io.onedrive_local import (
    list_run_dirs,
    normalize_root,
    pick_best_csv,
    read_csv_full,
    read_csv_preview,
)


def _default_onedrive_root() -> Path:
    """Try to auto-detect the local OneDrive sync folder on macOS."""
    base = Path.home() / "Library" / "CloudStorage"
    if base.exists():
        candidates = sorted(base.glob("OneDrive-*"))
        # Prefer DTU OneDrive if present
        for c in candidates:
            if "DanmarksTekniskeUniversitet" in c.name:
                return c
        if candidates:
            return candidates[0]
    return base


def _guess_time_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "Time",
        "time",
        "timestamp",
        "Timestamp",
        "DateTime",
        "datetime",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _guess_signal_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        s = str(c)
        # legacy heuristics
        if "Flowboard" in s:
            cols.append(c)
        if s.startswith("IPS "):
            cols.append(c)

    # fallback: any numeric columns
    if not cols:
        cols = df.select_dtypes(include=["number"]).columns.tolist()
    return cols


@dataclass(frozen=True)
class SweepSpec:
    start_hz: float
    end_hz: float
    duration_s: float


_SWEEP_RE = re.compile(
    r"(?P<start>\d+(?:\.\d+)?)Hz_(?P<end>\d+(?:\.\d+)?)Hz_(?P<dur>\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)


def _parse_sweep_spec_from_name(name: str) -> SweepSpec | None:
    m = _SWEEP_RE.search(name)
    if not m:
        return None
    return SweepSpec(
        start_hz=float(m.group("start")),
        end_hz=float(m.group("end")),
        duration_s=float(m.group("dur")),
    )


def _add_elapsed_s(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col])
    out["Elapsed_s"] = (out[time_col] - out[time_col].iloc[0]).dt.total_seconds()
    return out


def _add_frequency_and_sweep_index(df: pd.DataFrame, spec: SweepSpec) -> pd.DataFrame:
    out = df.copy()
    sweep_s = float(spec.duration_s)
    out["Sweep"] = (out["Elapsed_s"] // sweep_s).astype(int)
    phase = (out["Elapsed_s"] % sweep_s) / sweep_s
    out["Frequency"] = spec.start_hz + (spec.end_hz - spec.start_hz) * phase
    return out


def _bin_by_frequency(
    df: pd.DataFrame,
    *,
    value_col: str,
    freq_col: str = "Frequency",
    bin_hz: float = 5.0,
) -> pd.DataFrame:
    out = df[[freq_col, value_col]].dropna().copy()

    f = out[freq_col].astype(float).to_numpy()
    v = out[value_col].astype(float).to_numpy()

    fmin = float(np.nanmin(f))
    fmax = float(np.nanmax(f))
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
        raise ValueError("Invalid frequency range for binning.")

    edges = np.arange(fmin, fmax + bin_hz, bin_hz)
    idx = np.digitize(f, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    centers = (edges[:-1] + edges[1:]) / 2.0

    binned = (
        pd.DataFrame({"bin": idx, "value": v})
        .groupby("bin")
        .agg(mean=("value", "mean"), std=("value", "std"), count=("value", "count"))
        .reset_index()
    )
    binned["freq_center"] = binned["bin"].map(lambda i: float(centers[int(i)]))
    return binned.sort_values("freq_center")


def _fig_sweep_all_points(
    df: pd.DataFrame,
    *,
    value_col: str,
    freq_col: str = "Frequency",
    sweep_col: str = "Sweep",
    title: str = "",
):
    color = sweep_col if sweep_col in df.columns else None
    fig = px.scatter(df, x=freq_col, y=value_col, color=color, title=title)
    fig.update_layout(height=550, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Frequency (Hz)")
    return fig


def _fig_sweep_binned(binned: pd.DataFrame, *, title: str = ""):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=binned["freq_center"],
            y=binned["mean"],
            mode="lines+markers",
            name="Mean",
        )
    )

    if "std" in binned.columns:
        y_upper = binned["mean"] + binned["std"]
        y_lower = binned["mean"] - binned["std"]

        fig.add_trace(
            go.Scatter(
                x=binned["freq_center"],
                y=y_upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=binned["freq_center"],
                y=y_lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name="±1 std",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        height=550,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Frequency (Hz)",
    )
    return fig


# ============================================================================
# PAGE SETUP
# ============================================================================
st.set_page_config(page_title="AttoPump data visualisation", layout="wide")
st.title("AttoPump data visualisation")

# ============================================================================
# SIDEBAR: INPUT
# ============================================================================
with st.sidebar:
    st.header("📁 Data Source")
    data_folder_str = st.text_input(
        "Path to test data folder",
        value="",
        placeholder="/Users/.../All_tests",
        help="Paste path from Terminal (spaces are OK).",
    )
    st.divider()
    st.header("⚙️ CSV Selection")
    auto_pick = st.checkbox("Auto-pick best CSV", value=True)

# ============================================================================
# MAIN APP: ONLY RUN IF PATH PROVIDED
# ============================================================================
if not data_folder_str.strip():
    st.info("👈 **Enter a path in the sidebar to get started.**")
    st.stop()

# Validate folder
try:
    data_folder = normalize_root(data_folder_str)
except Exception as e:
    st.error(f"❌ {e}")
    st.stop()

# List test runs
try:
    run_dirs = list_run_dirs(data_folder, ".")
except Exception as e:
    st.error(f"❌ {e}")
    st.stop()

if not run_dirs:
    st.warning("⚠️ No test folders found in this path.")
    st.stop()

# Select test run
run_names = [p.name for p in run_dirs]
selected_run_name = st.selectbox("Select test run", run_names)
run_dir = run_dirs[run_names.index(selected_run_name)]

# Select or auto-pick CSV
if auto_pick:
    try:
        csv_path = pick_best_csv(run_dir).csv_path
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()
else:
    csvs = [p for p in sorted(run_dir.glob("*.csv")) if p.is_file()]
    if not csvs:
        st.warning("⚠️ No CSV files found.")
        st.stop()
    csv_choice = st.selectbox("Select CSV", [p.name for p in csvs])
    csv_path = run_dir / csv_choice

st.caption(f"📊 {csv_path.name}")

# ============================================================================
# LOAD DATA
# ============================================================================
try:
    df = read_csv_full(csv_path)
except Exception as e:
    st.error(f"❌ Failed to load CSV: {e}")
    st.stop()

# ============================================================================
# COLUMN SELECTION
# ============================================================================
time_guess = _guess_time_column(df)
signal_cols = _guess_signal_columns(df)

if not signal_cols:
    st.error("❌ No numeric signal columns found.")
    st.stop()

c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
with c1:
    time_col = st.selectbox(
        "Time column",
        options=list(df.columns),
        index=(list(df.columns).index(time_guess) if time_guess in df.columns else 0),
    )
with c2:
    signal_col = st.selectbox("Signal column", options=signal_cols)
with c3:
    parse_time = st.checkbox("Parse time", value=True)
    drop_na = st.checkbox("Drop NaNs", value=True)

# ============================================================================
# PREPARE PLOT DATA
# ============================================================================
plot_df = df[[time_col, signal_col]].copy()
if parse_time:
    plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors="coerce")
if drop_na:
    plot_df = plot_df.dropna(subset=[time_col, signal_col])

# ============================================================================
# TABS: TIME SERIES & SWEEP
# ============================================================================
spec = _parse_sweep_spec_from_name(selected_run_name)
tab_time, tab_sweep = st.tabs(["📈 Time Series", "🔄 Sweep"])

with tab_time:
    st.subheader("Signal vs Time")
    fig = px.line(plot_df, x=time_col, y=signal_col, title=f"{selected_run_name}")
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📊 Statistics"):
        st.write(plot_df[signal_col].describe())

with tab_sweep:
    st.subheader("Frequency Sweep")
    
    if spec is None:
        st.info("No sweep pattern detected (e.g., `10Hz_500Hz_60s`).")
    else:
        st.caption(f"{spec.start_hz:g}→{spec.end_hz:g} Hz, {spec.duration_s:g}s")
        
        sweep_df = df[[time_col, signal_col]].copy()
        if parse_time:
            sweep_df[time_col] = pd.to_datetime(sweep_df[time_col], errors="coerce")
        if drop_na:
            sweep_df = sweep_df.dropna(subset=[time_col, signal_col])
        
        if not np.issubdtype(sweep_df[time_col].dtype, np.datetime64):
            st.error("Need datetime for sweep analysis.")
            st.stop()
        
        sweep_df = _add_elapsed_s(sweep_df, time_col=time_col)
        sweep_df = _add_frequency_and_sweep_index(sweep_df, spec)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            bin_hz = st.number_input("Bin (Hz)", min_value=0.1, value=5.0, step=0.5)
        with col2:
            show_points = st.checkbox("Show all points", value=True)
        with col3:
            max_pts = st.number_input("Max points", min_value=1000, value=200000, step=10000)
        
        if show_points:
            pts = sweep_df if len(sweep_df) <= max_pts else sweep_df.sample(n=max_pts, random_state=0)
            if len(pts) < len(sweep_df):
                st.caption(f"Plotted {len(pts):,} of {len(sweep_df):,}")
            fig_pts = _fig_sweep_all_points(pts, value_col=signal_col, title="All points")
            st.plotly_chart(fig_pts, use_container_width=True)
        
        try:
            binned = _bin_by_frequency(sweep_df, value_col=signal_col, bin_hz=float(bin_hz))
            fig_bin = _fig_sweep_binned(binned, title="Binned mean ± std")
            st.plotly_chart(fig_bin, use_container_width=True)
            with st.expander("📋 Binned Data"):
                st.dataframe(binned, use_container_width=True)
        except Exception as e:
            st.error(f"❌ {e}")

