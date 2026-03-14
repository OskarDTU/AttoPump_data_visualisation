"""Plot generation for multi-test comprehensive analysis.

This module powers the **Comprehensive Analysis** page.  Unlike
``plot_generator`` (single-test figures), every function here accepts
*dictionaries* of DataFrames keyed by test name, enabling cross-test
comparison and statistical analysis.

Inspired by:
  - 03-06-2025-1338_flow_and_pressure_analysis.py
  - 27-05-2025-0041_configuration_comparison.py

Figure catalogue
----------------
1.  ``plot_combined_overlay``     — overlay binned mean±std from N tests.
2.  ``plot_global_average``       — inter-test average curve on a common
                                   frequency grid.
3.  ``plot_all_raw_points``       — scatter of every data point, coloured
                                   by test.
4.  ``plot_relative_comparison``  — each test normalised to 0–100 %.
5.  ``plot_combined_boxplots``    — side-by-side box-and-whisker plots.
6.  ``plot_combined_histograms``  — overlaid semi-transparent histograms.
7.  ``plot_std_vs_mean``          — variability scatter with linear fit.
8.  ``plot_stability_cloud``      — best operating region finder.
9.  ``plot_correlation_heatmap``  — Pearson correlation between binned
                                   mean-flow curves.
10. ``build_summary_table``       — summary statistics DataFrame.
11. ``plot_per_test_sweeps``      — binned per-sweep breakdown for one test.

Inputs
------
- ``dict[str, pd.DataFrame]`` of binned data (``test_binned``).
- ``dict[str, pd.DataFrame]`` of raw/time-series data (``test_raw``).
- Display options (mode, marker size, thresholds, …).

Outputs
-------
- ``plotly.graph_objects.Figure`` instances.
- ``pd.DataFrame`` for summary tables and best-region results.

All functions produce Plotly figures for interactive use in Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..data.config import PLOT_HEIGHT
from ..data.data_processor import bin_by_frequency
from .shared import PALETTE, flow_label, std_error_bar_style


# ═══════════════════════════════════════════════════════════════════════════
# 1. Combined overlay
# ═══════════════════════════════════════════════════════════════════════════


def plot_combined_overlay(
    test_binned: dict[str, pd.DataFrame],
    show_error_bars: bool = True,
    title: str = "Combined Frequency Sweep Comparison",
    height: int = PLOT_HEIGHT,
    mode: str = "lines+markers",
    marker_size: int = 6,
) -> go.Figure:
    """Overlay binned mean±std from multiple tests on one figure."""
    fig = go.Figure()
    names = list(test_binned.keys())

    for i, name in enumerate(names):
        binned = test_binned[name]
        color = PALETTE[i % len(PALETTE)]

        fig.add_trace(
            go.Scatter(
                x=binned["freq_center"],
                y=binned["mean"],
                mode=mode,
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=marker_size),
                error_y=(
                    std_error_bar_style(
                        binned["std"].fillna(0),
                        color=color,
                    )
                    if show_error_bars and "std" in binned.columns
                    else None
                ),
                legendgroup=name,
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Flow (µL/min)",
        hovermode="x unified",
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2. Global average across tests
# ═══════════════════════════════════════════════════════════════════════════


def plot_global_average(
    test_binned: dict[str, pd.DataFrame],
    bin_hz: float = 3.0,
    title: str = "Global Average Across Tests",
    height: int = PLOT_HEIGHT,
    mode: str = "lines+markers",
    marker_size: int = 6,
    show_error_bars: bool = True,
) -> go.Figure:
    """Compute and plot the average curve across all tests.

    Re-bins every test onto a common frequency grid, then computes
    the inter-test mean ± std at each grid point.

    Parameters
    ----------
    show_error_bars : bool
        When *True* (default), draw ±1 std error bars on the mean.
    """
    # Collect all frequency ranges
    all_freqs: list[float] = []
    for binned in test_binned.values():
        all_freqs.extend(binned["freq_center"].tolist())

    if not all_freqs:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig, pd.DataFrame()

    fmin, fmax = min(all_freqs), max(all_freqs)
    edges = np.arange(fmin, fmax + bin_hz, bin_hz)
    centers = (edges[:-1] + edges[1:]) / 2.0

    avg_vals: list[float] = []
    std_vals: list[float] = []

    for center in centers:
        vals: list[float] = []
        half = bin_hz / 2.0
        for binned in test_binned.values():
            mask = (binned["freq_center"] >= center - half) & (
                binned["freq_center"] < center + half
            )
            matched = binned.loc[mask, "mean"]
            if not matched.empty:
                vals.append(float(matched.mean()))
        if vals:
            avg_vals.append(float(np.mean(vals)))
            std_vals.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        else:
            avg_vals.append(np.nan)
            std_vals.append(np.nan)

    avg_df = pd.DataFrame(
        {"freq_center": centers, "mean": avg_vals, "std": std_vals}
    ).dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=avg_df["freq_center"],
            y=avg_df["mean"],
            mode=mode,
            name="Global Mean",
            line=dict(color="blue", width=3),
            marker=dict(size=marker_size),
            error_y=(
                std_error_bar_style(
                    avg_df["std"].fillna(0),
                    color="blue",
                )
                if show_error_bars
                else None
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Flow (µL/min)",
        hovermode="x unified",
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig, avg_df


# ═══════════════════════════════════════════════════════════════════════════
# 3. All raw data points
# ═══════════════════════════════════════════════════════════════════════════


def plot_all_raw_points(
    test_raw: dict[str, pd.DataFrame],
    freq_col: str = "Frequency",
    signal_col: str = "flow",
    title: str = "All Raw Data Points",
    height: int = PLOT_HEIGHT,
    mode: str = "markers",
    marker_size: int = 3,
    opacity: float = 0.5,
) -> go.Figure:
    """Scatter of every data point from all tests, colored by test name."""
    fig = go.Figure()
    names = list(test_raw.keys())

    for i, name in enumerate(names):
        df = test_raw[name]
        if freq_col not in df.columns or signal_col not in df.columns:
            continue
        if "IsFrequencyHold" in df.columns:
            df = df.loc[~df["IsFrequencyHold"]].copy()
        if df.empty:
            continue
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Scattergl(
                x=df[freq_col],
                y=df[signal_col],
                mode=mode,
                marker=dict(size=marker_size, color=color, opacity=opacity),
                line=dict(color=color, width=1.2),
                name=name,
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Frequency (Hz)",
        yaxis_title=flow_label(signal_col),
        hovermode="closest",
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. Relative (0–100 %) comparison
# ═══════════════════════════════════════════════════════════════════════════


def plot_relative_comparison(
    test_binned: dict[str, pd.DataFrame],
    title: str = "Relative Flow (0–100 %) Comparison",
    height: int = PLOT_HEIGHT,
    mode: str = "lines+markers",
    marker_size: int = 6,
) -> go.Figure:
    """Normalize each test's binned mean to 0-100 % and overlay."""
    fig = go.Figure()

    for i, (name, binned) in enumerate(test_binned.items()):
        vals = binned["mean"].values
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        if vmax == vmin:
            norm = np.zeros_like(vals)
        else:
            norm = (vals - vmin) / (vmax - vmin) * 100.0
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=binned["freq_center"],
                y=norm,
                mode=mode,
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=marker_size),
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Relative Flow (%)",
        hovermode="x unified",
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5. Combined boxplots
# ═══════════════════════════════════════════════════════════════════════════


def plot_combined_boxplots(
    test_raw: dict[str, pd.DataFrame],
    signal_col: str = "flow",
    title: str = "Flow Distribution Comparison",
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """Side-by-side boxplots for each test."""
    fig = go.Figure()

    for i, (name, df) in enumerate(test_raw.items()):
        if signal_col not in df.columns:
            continue
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Box(
                y=df[signal_col].dropna(),
                name=name,
                boxmean="sd",
                marker_color=color,
                line_color=color,
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        yaxis_title=flow_label(signal_col),
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. Combined histograms
# ═══════════════════════════════════════════════════════════════════════════


def plot_combined_histograms(
    test_raw: dict[str, pd.DataFrame],
    signal_col: str = "flow",
    nbins: int = 50,
    title: str = "Flow Distribution Histograms",
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """Overlaid semi-transparent histograms for each test."""
    fig = go.Figure()

    for i, (name, df) in enumerate(test_raw.items()):
        if signal_col not in df.columns:
            continue
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Histogram(
                x=df[signal_col].dropna(),
                name=name,
                nbinsx=nbins,
                marker_color=color,
                opacity=0.55,
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        barmode="overlay",
        xaxis_title=flow_label(signal_col),
        yaxis_title="Count",
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 7. Std vs Mean scatter
# ═══════════════════════════════════════════════════════════════════════════


def plot_std_vs_mean(
    test_binned: dict[str, pd.DataFrame],
    title: str = "Std vs Mean — Variability Analysis",
    height: int = PLOT_HEIGHT,
    marker_size: int = 8,
) -> go.Figure:
    """Scatter of std vs mean per frequency bin, with linear regression."""
    means: list[float] = []
    stds: list[float] = []
    labels: list[str] = []

    for name, binned in test_binned.items():
        valid = binned.dropna(subset=["mean", "std"])
        means.extend(valid["mean"].tolist())
        stds.extend(valid["std"].tolist())
        labels.extend([name] * len(valid))

    if not means:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig

    scatter_df = pd.DataFrame({"Mean": means, "Std": stds, "Test": labels})

    fig = go.Figure()
    for i, (name, group) in enumerate(scatter_df.groupby("Test", sort=True)):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Scattergl(
                x=group["Mean"],
                y=group["Std"],
                mode="markers",
                name=name,
                marker=dict(size=marker_size, color=color),
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Mean Flow (µL/min)",
        yaxis_title="Std Dev (µL/min)",
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    # Linear regression trend line
    m_arr = np.array(means, dtype=float)
    s_arr = np.array(stds, dtype=float)
    mask = np.isfinite(m_arr) & np.isfinite(s_arr)

    if mask.sum() > 2:
        try:
            from scipy import stats as scipy_stats

            slope, intercept, r_value, _, _ = scipy_stats.linregress(
                m_arr[mask], s_arr[mask]
            )
            x_line = np.linspace(float(m_arr[mask].min()), float(m_arr[mask].max()), 100)
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=intercept + slope * x_line,
                    mode="lines",
                    name=f"Fit (R²={r_value**2:.3f})",
                    line=dict(color="red", width=2, dash="dash"),
                )
            )

            # Compute Pearson & Spearman
            pearson_r = r_value
            try:
                spearman_r, _ = scipy_stats.spearmanr(m_arr[mask], s_arr[mask])
            except Exception:
                spearman_r = float("nan")

            fig.add_annotation(
                text=(
                    f"Pearson r = {pearson_r:.3f}<br>"
                    f"Spearman ρ = {spearman_r:.3f}<br>"
                    f"slope = {slope:.4f}"
                ),
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                showarrow=False,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="grey",
                borderwidth=1,
            )
        except ImportError:
            pass  # scipy not available

    fig.update_layout(
        height=height,
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 8. Stability cloud (best operating region)
# ═══════════════════════════════════════════════════════════════════════════


def plot_stability_cloud(
    test_binned: dict[str, pd.DataFrame],
    mean_threshold_pct: float = 75.0,
    std_threshold_pct: float = 10.0,
    title: str = "High-Flow / High-Stability Region",
    height: int = PLOT_HEIGHT,
    marker_size: int = 8,
) -> tuple[go.Figure, pd.DataFrame]:
    """Highlight the best operating bins directly on flow-versus-frequency axes."""
    rows: list[dict] = []
    for name, binned in test_binned.items():
        for _, row in binned.iterrows():
            if pd.notna(row.get("mean")) and pd.notna(row.get("std")):
                rows.append(
                    {
                        "freq_center": row["freq_center"],
                        "mean": row["mean"],
                        "std": row["std"],
                        "test": name,
                    }
                )

    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig, pd.DataFrame()

    pool = pd.DataFrame(rows)

    # High-flow threshold
    mean_cutoff = float(np.percentile(pool["mean"], 100 - mean_threshold_pct))
    high_flow = pool[pool["mean"] >= mean_cutoff]

    if high_flow.empty:
        high_stability = pd.DataFrame()
    else:
        std_cutoff = float(np.percentile(high_flow["std"], std_threshold_pct))
        high_stability = high_flow[high_flow["std"] <= std_cutoff]

    fig = go.Figure()
    for i, (name, sub) in enumerate(pool.groupby("test", sort=True)):
        sub = sub.sort_values("freq_center").reset_index(drop=True)
        hover = (
            sub["test"]
            + " @ "
            + sub["freq_center"].round(1).astype(str)
            + " Hz"
            + "<br>mean = "
            + sub["mean"].round(1).astype(str)
            + " µL/min"
            + "<br>std = "
            + sub["std"].round(1).astype(str)
            + " µL/min"
        )
        fig.add_trace(
            go.Scatter(
                x=sub["freq_center"],
                y=sub["mean"],
                mode="lines+markers",
                name="All bins" if i == 0 else name,
                showlegend=(i == 0),
                marker=dict(
                    size=max(1, marker_size - 3),
                    color="lightgrey",
                    opacity=0.45,
                ),
                line=dict(color="rgba(180, 180, 180, 0.45)", width=1.2),
                error_y=std_error_bar_style(
                    sub["std"].fillna(0),
                    color="lightgrey",
                    alpha=0.2,
                    thickness=0.8,
                    width=2.0,
                ),
                legendgroup="all_bins",
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    if not high_flow.empty:
        hover_hf = (
            high_flow["test"]
            + " @ "
            + high_flow["freq_center"].round(1).astype(str)
            + " Hz"
            + "<br>mean = "
            + high_flow["mean"].round(1).astype(str)
            + " µL/min"
            + "<br>std = "
            + high_flow["std"].round(1).astype(str)
            + " µL/min"
        )
        fig.add_trace(
            go.Scatter(
                x=high_flow["freq_center"],
                y=high_flow["mean"],
                mode="markers",
                name=f"High flow (top {mean_threshold_pct:.0f} %)",
                marker=dict(size=marker_size, color="orange", opacity=0.75),
                error_y=std_error_bar_style(
                    high_flow["std"].fillna(0),
                    color="orange",
                    alpha=0.35,
                    thickness=1.0,
                    width=3.0,
                ),
                text=hover_hf,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    if not high_stability.empty:
        high_stability = high_stability.sort_values(
            ["std", "mean"],
            ascending=[True, False],
        ).reset_index(drop=True)
        hover_hs = (
            high_stability["test"]
            + " @ "
            + high_stability["freq_center"].round(1).astype(str)
            + " Hz"
            + "<br>mean = "
            + high_stability["mean"].round(1).astype(str)
            + " µL/min"
            + "<br>std = "
            + high_stability["std"].round(1).astype(str)
            + " µL/min"
        )
        fig.add_trace(
            go.Scatter(
                x=high_stability["freq_center"],
                y=high_stability["mean"],
                mode="markers",
                name=f"Best-region bins (bottom {std_threshold_pct:.0f} % std)",
                marker=dict(
                    size=marker_size + 4,
                    color="green",
                    symbol="star",
                    opacity=0.95,
                    line=dict(color="darkgreen", width=1.0),
                ),
                error_y=std_error_bar_style(
                    high_stability["std"].fillna(0),
                    color="green",
                    alpha=0.45,
                    thickness=1.1,
                    width=3.0,
                ),
                text=hover_hs,
                hovertemplate="%{text}<extra></extra>",
            )
        )

        best_point = high_stability.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[best_point["freq_center"]],
                y=[best_point["mean"]],
                mode="markers",
                name=(
                    f"Best point — {best_point['mean']:.1f} µL/min "
                    f"±{best_point['std']:.1f} @ {best_point['freq_center']:.0f} Hz"
                ),
                marker=dict(
                    size=marker_size + 8,
                    color="#ffd54f",
                    symbol="star",
                    line=dict(color="red", width=2.0),
                ),
                error_y=std_error_bar_style(
                    [best_point["std"]],
                    color="red",
                    alpha=0.6,
                    thickness=1.2,
                    width=4.0,
                ),
                hovertemplate=(
                    f"{best_point['test']} @ {best_point['freq_center']:.1f} Hz"
                    f"<br>mean = {best_point['mean']:.1f} µL/min"
                    f"<br>std = {best_point['std']:.1f} µL/min"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Flow (µL/min)",
        hovermode="closest",
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig, high_stability


# ═══════════════════════════════════════════════════════════════════════════
# 9. Correlation heatmap
# ═══════════════════════════════════════════════════════════════════════════


def plot_correlation_heatmap(
    test_binned: dict[str, pd.DataFrame],
    title: str = "Inter-Test Correlation (Binned Means)",
    height: int = 500,
) -> go.Figure:
    """Correlation heatmap across tests using interpolated binned means."""
    try:
        from scipy import interpolate as interp
    except ImportError:
        fig = go.Figure()
        fig.add_annotation(text="scipy required for correlation", showarrow=False)
        return fig

    # Build common frequency grid
    all_freqs: set[float] = set()
    for binned in test_binned.values():
        all_freqs.update(binned["freq_center"].tolist())

    if len(all_freqs) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for correlation", showarrow=False)
        return fig

    common_freqs = np.array(sorted(all_freqs))
    interp_means: dict[str, np.ndarray] = {}

    for name, binned in test_binned.items():
        f = binned["freq_center"].values
        m = binned["mean"].values
        if len(f) >= 2:
            func = interp.interp1d(f, m, bounds_error=False, fill_value=np.nan)
            interp_means[name] = func(common_freqs)

    if len(interp_means) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need ≥ 2 tests for correlation", showarrow=False)
        return fig

    corr_df = pd.DataFrame(interp_means).corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns.tolist(),
            y=corr_df.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_df.values, 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 10. Summary statistics table
# ═══════════════════════════════════════════════════════════════════════════


def build_summary_table(
    test_raw: dict[str, pd.DataFrame],
    signal_col: str = "flow",
) -> pd.DataFrame:
    """Build a comparison table of summary statistics across tests."""
    rows: list[dict] = []
    for name, df in test_raw.items():
        if signal_col not in df.columns:
            continue
        s = df[signal_col].dropna()
        if s.empty:
            continue
        cv = round(float(s.std() / s.mean() * 100), 2) if (s.mean() != 0 and len(s) > 1 and np.isfinite(s.std())) else 0.0
        rows.append(
            {
                "Test": name,
                "N": len(s),
                "Mean (µL/min)": round(float(s.mean()), 2),
                "Std (µL/min)": round(float(s.std()), 2),
                "CV (%)": cv,
                "Min": round(float(s.min()), 2),
                "Q1": round(float(s.quantile(0.25)), 2),
                "Median": round(float(s.median()), 2),
                "Q3": round(float(s.quantile(0.75)), 2),
                "Max": round(float(s.max()), 2),
            }
        )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Per-test: binned flow vs frequency per sweep
# ═══════════════════════════════════════════════════════════════════════════


def plot_per_test_sweeps(
    sweep_df: pd.DataFrame,
    signal_col: str,
    bin_hz: float = 5.0,
    title: str = "",
    height: int = PLOT_HEIGHT,
    mode: str = "lines+markers",
    marker_size: int = 5,
) -> go.Figure:
    """For a single test, plot binned mean±std per sweep cycle."""
    fig = go.Figure()

    if "Sweep" not in sweep_df.columns or "Frequency" not in sweep_df.columns:
        fig.add_annotation(text="No sweep/frequency data", showarrow=False)
        return fig

    sweeps = sorted(sweep_df["Sweep"].unique())
    for i, sw in enumerate(sweeps):
        sub = sweep_df[sweep_df["Sweep"] == sw]
        if "IsFrequencyHold" in sub.columns:
            sub = sub.loc[~sub["IsFrequencyHold"]].copy()
        if sub.empty:
            continue

        # Quick binning within this sweep
        f = sub["Frequency"].values.astype(float)
        v = sub[signal_col].values.astype(float)
        mask = np.isfinite(f) & np.isfinite(v)
        f, v = f[mask], v[mask]
        if len(f) < 2:
            continue

        fmin, fmax = float(f.min()), float(f.max())
        if fmax <= fmin:
            continue
        edges = np.arange(fmin, fmax + bin_hz, bin_hz)
        if len(edges) < 2:
            continue
        idx = np.clip(np.digitize(f, edges) - 1, 0, len(edges) - 2)
        centers = (edges[:-1] + edges[1:]) / 2.0

        binned = (
            pd.DataFrame({"bin": idx, "value": v})
            .groupby("bin")
            .agg(mean=("value", "mean"), std=("value", "std"))
            .reset_index()
        )
        binned["freq_center"] = binned["bin"].map(
            lambda j, c=centers: float(c[int(j)]) if int(j) < len(c) else float(fmax)
        )

        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=binned["freq_center"],
                y=binned["mean"],
                mode=mode,
                name=f"Sweep {sw}",
                line=dict(color=color, width=1.5),
                marker=dict(size=marker_size),
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Frequency (Hz)",
        yaxis_title=flow_label(signal_col),
        hovermode="x unified",
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_target_sweep_drilldown(
    test_sweeps: dict[str, pd.DataFrame],
    *,
    signal_col: str,
    bin_hz: float = 5.0,
    title: str = "Focused Sweep Drill-Down",
    height: int = PLOT_HEIGHT,
    mode: str = "lines+markers",
    marker_size: int = 4,
) -> go.Figure:
    """Overlay every sweep from several tests on one focused figure."""
    fig = go.Figure()
    dash_styles = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
    plotted = 0

    for test_idx, (test_name, sweep_df) in enumerate(test_sweeps.items()):
        if (
            sweep_df is None
            or sweep_df.empty
            or signal_col not in sweep_df.columns
            or "Frequency" not in sweep_df.columns
        ):
            continue

        plot_df = sweep_df.copy()
        if "IsFrequencyHold" in plot_df.columns:
            plot_df = plot_df.loc[~plot_df["IsFrequencyHold"]].copy()
        if plot_df.empty:
            continue

        color = PALETTE[test_idx % len(PALETTE)]
        if "Sweep" in plot_df.columns and plot_df["Sweep"].notna().any():
            sweep_groups = plot_df.groupby("Sweep", sort=True)
        else:
            sweep_groups = [(0, plot_df)]

        for sweep_idx, (sweep_id, sweep_sub) in enumerate(sweep_groups):
            try:
                binned = bin_by_frequency(
                    sweep_sub,
                    value_col=signal_col,
                    bin_hz=float(bin_hz),
                )
            except ValueError:
                continue
            if binned.empty:
                continue

            trace_name = f"{test_name} / S{int(sweep_id) + 1}"
            hover_template = (
                f"{test_name}<br>Sweep {int(sweep_id) + 1}"
                "<br>Frequency=%{x:.1f} Hz"
                "<br>Flow=%{y:.2f}"
                "<extra></extra>"
            )
            fig.add_trace(
                go.Scatter(
                    x=binned["freq_center"],
                    y=binned["mean"],
                    mode=mode,
                    name=trace_name,
                    line=dict(
                        color=color,
                        width=1.6,
                        dash=dash_styles[sweep_idx % len(dash_styles)],
                    ),
                    marker=dict(size=marker_size, color=color),
                    legendgroup=test_name,
                    hovertemplate=hover_template,
                )
            )
            plotted += 1

    if plotted == 0:
        fig.add_annotation(text="No sweep data available", showarrow=False)

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Frequency (Hz)",
        yaxis_title=flow_label(signal_col),
        hovermode="x unified",
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
