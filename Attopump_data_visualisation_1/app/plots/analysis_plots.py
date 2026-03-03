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
import plotly.express as px
import plotly.graph_objects as go

from ..data.config import PLOT_HEIGHT

# ---------------------------------------------------------------------------
# Color palette (enough for many tests)
# ---------------------------------------------------------------------------
PALETTE = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.Set2
    + px.colors.qualitative.Dark24
)


def _color_to_rgba(color: str, alpha: float = 0.2) -> str:
    """Convert any Plotly color string to ``rgba(R, G, B, alpha)``.

    Handles ``#RRGGBB``, ``rgb(R, G, B)``, and bare color names.
    """
    import re as _re

    # rgb(R, G, B) or rgba(R, G, B, A)
    m = _re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", color)
    if m:
        return f"rgba({m.group(1)}, {m.group(2)}, {m.group(3)}, {alpha})"

    # #RRGGBB
    hex_color = color.lstrip("#")
    if len(hex_color) == 6:
        try:
            r = int(hex_color[:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f"rgba({r}, {g}, {b}, {alpha})"
        except ValueError:
            pass

    # Fallback: semi-transparent grey
    return f"rgba(128, 128, 128, {alpha})"


def _flow_label(col: str) -> str:
    """Return a human-readable axis label for a flow column."""
    return f"{col} (µL/min)" if "flow" in col.lower() else col


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
                legendgroup=name,
            )
        )

        if show_error_bars and "std" in binned.columns:
            upper = binned["mean"] + binned["std"].fillna(0)
            lower = binned["mean"] - binned["std"].fillna(0)
            fill_color = _color_to_rgba(color, 0.15)

            fig.add_trace(
                go.Scatter(
                    x=binned["freq_center"],
                    y=upper,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    legendgroup=name,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=binned["freq_center"],
                    y=lower,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=fill_color,
                    showlegend=False,
                    legendgroup=name,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Flow (µL/min)",
        hovermode="x unified",
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
        When *True* (default), draw a ±1 std shaded band around the mean.
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
        )
    )

    upper = avg_df["mean"] + avg_df["std"]
    lower = avg_df["mean"] - avg_df["std"]

    if show_error_bars:
        fig.add_trace(
            go.Scatter(
                x=avg_df["freq_center"],
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=avg_df["freq_center"],
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name="±1 std (inter-test)",
                fillcolor="rgba(0, 100, 255, 0.2)",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Flow (µL/min)",
        hovermode="x unified",
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
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Scattergl(
                x=df[freq_col],
                y=df[signal_col],
                mode="markers",
                marker=dict(size=marker_size, color=color, opacity=opacity),
                name=name,
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Frequency (Hz)",
        yaxis_title=_flow_label(signal_col),
        hovermode="closest",
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
        yaxis_title=_flow_label(signal_col),
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
        xaxis_title=_flow_label(signal_col),
        yaxis_title="Count",
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

    fig = px.scatter(
        scatter_df,
        x="Mean",
        y="Std",
        color="Test",
        title=title,
        labels={"Mean": "Mean Flow (µL/min)", "Std": "Std Dev (µL/min)"},
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=marker_size))

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
    """Find frequency bins with highest flow and lowest variability.

    Method (from reference script):
      1. Pool all binned data from all tests.
      2. Top ``mean_threshold_pct`` % by mean  → *high-flow* set.
      3. Within that, bottom ``std_threshold_pct`` % by std  → *high-stability* set.

    Returns
    -------
    (fig, best_df)
        ``best_df`` contains the green-star rows for downstream display.
    """
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

    # All points (grey)
    hover = pool["test"] + " @ " + pool["freq_center"].round(1).astype(str) + " Hz"
    fig.add_trace(
        go.Scatter(
            x=pool["mean"],
            y=pool["std"],
            mode="markers",
            name="All bins",
            marker=dict(size=max(1, marker_size - 2), color="lightgrey", opacity=0.5),
            text=hover,
            hoverinfo="text+x+y",
        )
    )

    # High flow (orange)
    if not high_flow.empty:
        hover_hf = (
            high_flow["test"]
            + " @ "
            + high_flow["freq_center"].round(1).astype(str)
            + " Hz"
        )
        fig.add_trace(
            go.Scatter(
                x=high_flow["mean"],
                y=high_flow["std"],
                mode="markers",
                name=f"High flow (top {mean_threshold_pct:.0f} %)",
                marker=dict(size=marker_size, color="orange", opacity=0.7),
                text=hover_hf,
                hoverinfo="text+x+y",
            )
        )

    # High stability (green stars)
    if not high_stability.empty:
        hover_hs = (
            high_stability["test"]
            + " @ "
            + high_stability["freq_center"].round(1).astype(str)
            + " Hz"
        )
        fig.add_trace(
            go.Scatter(
                x=high_stability["mean"],
                y=high_stability["std"],
                mode="markers",
                name=f"High stability (bottom {std_threshold_pct:.0f} % std)",
                marker=dict(
                    size=marker_size + 4, color="green", symbol="star", opacity=0.9
                ),
                text=hover_hs,
                hoverinfo="text+x+y",
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Mean Flow (µL/min)",
        yaxis_title="Std Dev (µL/min)",
        hovermode="closest",
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
        yaxis_title=_flow_label(signal_col),
        hovermode="x unified",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
