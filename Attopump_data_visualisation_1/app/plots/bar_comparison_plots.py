"""Plot generation for bar-vs-bar comparison.

Produces side-by-side and overlay figures that compare **bars**
(groups of tests belonging to the same pump) rather than individual tests.

Two comparison modes:
  • **Constant-frequency tests** – boxplots, histograms, summary stats
  • **Frequency-sweep tests** – overlay of binned mean±std curves
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from ..data.config import PLOT_HEIGHT
from .shared import PUMP_PALETTE, color_to_rgba, flow_label


def _bar_color(index: int) -> str:
    """Return a distinct colour for the *index*-th bar."""
    return PUMP_PALETTE[index % len(PUMP_PALETTE)]


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Sweep comparison – overlay binned curves per bar
# ═══════════════════════════════════════════════════════════════════════════

def plot_bar_sweep_overlay(
    bar_binned: dict[str, dict[str, pd.DataFrame]],
    show_error_bars: bool = True,
    show_individual: bool = False,
    mode: str = "lines+markers",
    marker_size: int = 6,
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """Overlay the **per-bar average** binned sweep curves.

    Parameters
    ----------
    bar_binned : {bar_name: {test_name: binned_df, …}, …}
        Nested dict — outer key = bar name, inner key = test name.
    show_individual : bool
        If True, also draw faint per-test lines behind each bar's average.
    """
    fig = go.Figure()

    for i, (bar_name, tests_binned) in enumerate(bar_binned.items()):
        color = _bar_color(i)

        # Compute bar-level average across all its tests
        avg_df = _average_binned(tests_binned)
        if avg_df.empty:
            continue

        # Optional: faint individual test lines
        if show_individual:
            for test_name, tdf in tests_binned.items():
                fig.add_trace(
                    go.Scatter(
                        x=tdf["freq_center"],
                        y=tdf["mean"],
                        mode="lines",
                        line=dict(color=color_to_rgba(color, 0.25), width=1),
                        name=f"{bar_name} / {test_name}",
                        legendgroup=bar_name,
                        showlegend=False,
                        hoverinfo="text",
                        text=[f"{bar_name} / {test_name}"] * len(tdf),
                    )
                )

        # Bar average
        fig.add_trace(
            go.Scatter(
                x=avg_df["freq_center"],
                y=avg_df["mean"],
                mode=mode,
                name=bar_name,
                line=dict(color=color, width=3),
                marker=dict(size=marker_size),
                legendgroup=bar_name,
            )
        )

        # ± std band
        if show_error_bars and "std" in avg_df.columns:
            upper = avg_df["mean"] + avg_df["std"].fillna(0)
            lower = avg_df["mean"] - avg_df["std"].fillna(0)
            fill = color_to_rgba(color, 0.15)
            fig.add_trace(
                go.Scatter(
                    x=avg_df["freq_center"], y=upper,
                    mode="lines", line=dict(width=0),
                    showlegend=False, legendgroup=bar_name, hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=avg_df["freq_center"], y=lower,
                    mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=fill,
                    showlegend=False, legendgroup=bar_name, hoverinfo="skip",
                )
            )

    fig.update_layout(
        title="Frequency Sweep – Bar Comparison",
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
# 2.  Sweep comparison – relative (0–100%) overlay
# ═══════════════════════════════════════════════════════════════════════════

def plot_bar_sweep_relative(
    bar_binned: dict[str, dict[str, pd.DataFrame]],
    mode: str = "lines+markers",
    marker_size: int = 6,
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """Overlay per-bar average curves normalised to 0–100 %."""
    fig = go.Figure()

    for i, (bar_name, tests_binned) in enumerate(bar_binned.items()):
        avg_df = _average_binned(tests_binned)
        if avg_df.empty:
            continue
        vals = avg_df["mean"].values
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        norm = (vals - vmin) / (vmax - vmin) * 100.0 if vmax > vmin else np.zeros_like(vals)
        color = _bar_color(i)
        fig.add_trace(
            go.Scatter(
                x=avg_df["freq_center"], y=norm,
                mode=mode, name=bar_name,
                line=dict(color=color, width=2),
                marker=dict(size=marker_size),
            )
        )

    fig.update_layout(
        title="Relative Flow (0–100 %) – Bar Comparison",
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
# 3.  Constant-frequency – boxplots per bar
# ═══════════════════════════════════════════════════════════════════════════

def plot_bar_constant_boxplots(
    bar_const_data: dict[str, dict[str, pd.DataFrame]],
    signal_col: str = "flow",
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """Side-by-side boxplots grouped by bar, one box per test within each bar."""
    fig = go.Figure()

    for i, (bar_name, tests) in enumerate(bar_const_data.items()):
        color = _bar_color(i)
        for test_name, df in tests.items():
            if signal_col not in df.columns:
                continue
            fig.add_trace(
                go.Box(
                    y=df[signal_col].dropna(),
                    name=f"{bar_name} / {test_name}",
                    boxmean="sd",
                    marker_color=color,
                    line_color=color,
                    legendgroup=bar_name,
                    legendgrouptitle_text=bar_name,
                )
            )

    fig.update_layout(
        title="Constant-Frequency Flow – Bar Comparison",
        height=height,
        yaxis_title=flow_label(signal_col),
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        boxmode="group",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Constant-frequency – aggregated bar boxplots
# ═══════════════════════════════════════════════════════════════════════════

def plot_bar_constant_aggregated(
    bar_const_data: dict[str, dict[str, pd.DataFrame]],
    signal_col: str = "flow",
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """One boxplot per bar, pooling all constant-freq tests together."""
    fig = go.Figure()

    for i, (bar_name, tests) in enumerate(bar_const_data.items()):
        pooled: list[float] = []
        for df in tests.values():
            if signal_col in df.columns:
                pooled.extend(df[signal_col].dropna().tolist())
        if not pooled:
            continue
        color = _bar_color(i)
        fig.add_trace(
            go.Box(
                y=pooled,
                name=bar_name,
                boxmean="sd",
                marker_color=color,
                line_color=color,
            )
        )

    fig.update_layout(
        title="Aggregated Constant-Frequency Flow per Bar",
        height=height,
        yaxis_title=flow_label(signal_col),
        dragmode="zoom",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Constant-frequency – overlay histograms per bar
# ═══════════════════════════════════════════════════════════════════════════

def plot_bar_constant_histograms(
    bar_const_data: dict[str, dict[str, pd.DataFrame]],
    signal_col: str = "flow",
    nbins: int = 50,
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """Overlaid histograms (one colour per bar, pooling tests)."""
    fig = go.Figure()

    for i, (bar_name, tests) in enumerate(bar_const_data.items()):
        pooled: list[float] = []
        for df in tests.values():
            if signal_col in df.columns:
                pooled.extend(df[signal_col].dropna().tolist())
        if not pooled:
            continue
        color = _bar_color(i)
        fig.add_trace(
            go.Histogram(
                x=pooled,
                name=bar_name,
                nbinsx=nbins,
                marker_color=color,
                opacity=0.55,
            )
        )

    fig.update_layout(
        title="Constant-Frequency Flow Histograms – Bar Comparison",
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
# 6.  Summary statistics table for bars
# ═══════════════════════════════════════════════════════════════════════════

def build_bar_summary_table(
    bar_data: dict[str, dict[str, pd.DataFrame]],
    signal_col: str = "flow",
    test_type: str = "",
) -> pd.DataFrame:
    """Build a summary table with one row per bar (pooled across tests)."""
    rows: list[dict] = []
    for bar_name, tests in bar_data.items():
        pooled: list[float] = []
        for df in tests.values():
            if signal_col in df.columns:
                pooled.extend(df[signal_col].dropna().tolist())
        if not pooled:
            continue
        s = pd.Series(pooled, dtype=float)
        cv = round(float(s.std() / s.mean() * 100), 2) if s.mean() != 0 and len(s) > 1 else 0.0
        rows.append({
            "Bar": bar_name,
            "Type": test_type,
            "# Tests": len(tests),
            "N points": len(s),
            "Mean (µL/min)": round(float(s.mean()), 2),
            "Std (µL/min)": round(float(s.std()), 2),
            "CV (%)": cv,
            "Min": round(float(s.min()), 2),
            "Q1": round(float(s.quantile(0.25)), 2),
            "Median": round(float(s.median()), 2),
            "Q3": round(float(s.quantile(0.75)), 2),
            "Max": round(float(s.max()), 2),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Helper – average multiple binned DataFrames onto a common grid
# ═══════════════════════════════════════════════════════════════════════════

def _average_binned(
    tests_binned: dict[str, pd.DataFrame],
    bin_hz: float = 5.0,
) -> pd.DataFrame:
    """Compute mean ± inter-test std on a common frequency grid.

    All binned DataFrames are re-sampled onto a shared set of equal-width
    frequency bins.  At each grid point the inter-test mean and std are
    computed, producing a single representative curve for the bar.

    Parameters
    ----------
    tests_binned : dict[str, pd.DataFrame]
        Mapping of test name → binned DataFrame (must contain
        ``freq_center`` and ``mean`` columns).
    bin_hz : float
        Width of each frequency bin on the common grid.

    Returns
    -------
    pd.DataFrame
        Columns: ``freq_center``, ``mean``, ``std``.
        Empty DataFrame if no input data.
    """
    all_freqs: list[float] = []
    for b in tests_binned.values():
        all_freqs.extend(b["freq_center"].tolist())
    if not all_freqs:
        return pd.DataFrame()

    fmin, fmax = min(all_freqs), max(all_freqs)
    edges = np.arange(fmin, fmax + bin_hz, bin_hz)
    centers = (edges[:-1] + edges[1:]) / 2.0

    avg_vals: list[float] = []
    std_vals: list[float] = []
    for center in centers:
        vals: list[float] = []
        half = bin_hz / 2.0
        for b in tests_binned.values():
            mask = (b["freq_center"] >= center - half) & (b["freq_center"] < center + half)
            matched = b.loc[mask, "mean"]
            if not matched.empty:
                vals.append(float(matched.mean()))
        if vals:
            avg_vals.append(float(np.mean(vals)))
            std_vals.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        else:
            avg_vals.append(np.nan)
            std_vals.append(np.nan)

    return pd.DataFrame({
        "freq_center": centers,
        "mean": avg_vals,
        "std": std_vals,
    }).dropna()
