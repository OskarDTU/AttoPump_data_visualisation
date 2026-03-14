"""Core Plotly figure generation for single-test visualisation.

This module produces interactive Plotly figures consumed by the
**Single Test Explorer** page.  Each public function takes a cleaned
``pd.DataFrame`` (prepared by ``data_processor``) plus display options
and returns a ``go.Figure`` ready to embed with ``st.plotly_chart``.

Figure catalogue
----------------
- ``plot_time_series``              — flow vs time (line / scatter / both).
- ``plot_constant_frequency_boxplot`` — overall flow-rate distribution box.
- ``plot_flow_histogram``           — histogram of flow rates.
- ``downsample_sweep_points``       — balanced downsampling that keeps all sweeps visible.
- ``plot_sweep_all_points``         — raw scatter coloured by sweep cycle.
- ``plot_sweep_per_sweep_average``  — one binned mean+-std trace per sweep.
- ``plot_sweep_binned``             — binned mean ± std band.
- ``export_html``                   — save any figure to a standalone HTML file.

Inputs
------
- ``pd.DataFrame`` with at least the time column and one signal column.
- Plot-appearance options (mode, marker size, opacity, height).

Outputs
-------
- ``plotly.graph_objects.Figure`` instances.
- ``pathlib.Path`` for exported HTML files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from ..data.config import PLOT_HEIGHT
from .shared import flow_label, std_error_bar_style, time_label


def plot_time_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    height: int = PLOT_HEIGHT,
    mode: str = "lines",
    marker_size: int = 4,
    opacity: float = 1.0,
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Create time series plot (line, scatter, or both).

    Parameters
    ----------
    mode : str
        ``"lines"`` | ``"markers"`` | ``"lines+markers"``
    marker_size : int
        Point diameter in pixels.
    opacity : float
        Marker / line opacity 0-1.
    y_range : tuple or None
        Fixed ``(ymin, ymax)`` for the y-axis.  ``None`` = auto-range.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df[x_col],
            y=df[y_col],
            mode=mode,
            marker=dict(size=marker_size, opacity=opacity),
            line=dict(width=2),
            name=y_col,
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        dragmode="zoom",
        font=dict(size=12),
        xaxis_title=time_label(x_col),
        yaxis_title=flow_label(y_col),
    )
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    fig.update_xaxes(title_font=dict(size=12))
    fig.update_yaxes(title_font=dict(size=12))
    return fig


def plot_constant_frequency_boxplot(
    df: pd.DataFrame,
    y_col: str = "flow",
    title: str = "",
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """Single boxplot showing overall flow-rate distribution."""
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=df[y_col].dropna(),
            name=y_col,
            boxmean="sd",
            boxpoints="outliers",
            marker_color="#636EFA",
            line_color="#636EFA",
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title=flow_label(y_col),
        dragmode="zoom",
        font=dict(size=12),
        showlegend=False,
    )
    fig.update_yaxes(title_font=dict(size=12))
    return fig


def plot_flow_histogram(
    df: pd.DataFrame,
    y_col: str = "flow",
    nbins: int = 30,
    title: str = "",
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """Histogram of flow rates for constant frequency test."""
    fig = go.Figure(
        data=[
            go.Histogram(
                x=df[y_col].dropna(),
                nbinsx=nbins,
                marker_color="#636EFA",
                name=y_col,
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x",
        dragmode="zoom",
        font=dict(size=12),
        xaxis_title=flow_label(y_col),
        yaxis_title="Frequency (count)",
    )
    fig.update_xaxes(title_font=dict(size=12))
    fig.update_yaxes(title_font=dict(size=12))
    return fig


# Qualitative palette for sweep colouring (10 distinct colours).
_SWEEP_PALETTE = qualitative.Plotly


def _evenly_spaced_subset(df: pd.DataFrame, n_points: int) -> pd.DataFrame:
    """Keep an evenly spaced subset without dropping an entire segment."""
    if n_points <= 0 or df.empty:
        return df.iloc[0:0].copy()
    if len(df) <= n_points:
        return df.copy()
    idx = np.linspace(0, len(df) - 1, num=n_points, dtype=int)
    return df.iloc[np.unique(idx)].copy()


def downsample_sweep_points(
    df: pd.DataFrame,
    *,
    max_points: int,
    sweep_col: str = "Sweep",
) -> pd.DataFrame:
    """Downsample raw sweep data while keeping every sweep represented."""
    if max_points <= 0 or df.empty:
        return df.iloc[0:0].copy()
    if len(df) <= max_points:
        return df.copy()
    if sweep_col not in df.columns:
        return _evenly_spaced_subset(df, max_points)

    groups = [(sw, sub.copy()) for sw, sub in df.groupby(sweep_col, sort=True)]
    if not groups:
        return _evenly_spaced_subset(df, max_points)

    sizes = np.array([len(sub) for _, sub in groups], dtype=int)
    group_count = len(groups)
    if group_count >= max_points:
        chosen = np.argsort(-sizes, kind="stable")[:max_points]
        selected = [
            _evenly_spaced_subset(groups[int(i)][1], 1)
            for i in np.sort(chosen, kind="stable")
        ]
        return pd.concat(selected, axis=0).sort_index()

    allocations = np.floor(sizes / sizes.sum() * max_points).astype(int)
    allocations = np.maximum(allocations, 1)
    allocations = np.minimum(allocations, sizes)

    target_total = min(max_points, int(sizes.sum()))
    while allocations.sum() > target_total:
        reducible = np.where(allocations > 1)[0]
        if len(reducible) == 0:
            break
        idx = reducible[np.argmax(allocations[reducible])]
        allocations[idx] -= 1

    while allocations.sum() < target_total:
        room = sizes - allocations
        expandable = np.where(room > 0)[0]
        if len(expandable) == 0:
            break
        idx = expandable[np.argmax(room[expandable])]
        allocations[idx] += 1

    selected = [
        _evenly_spaced_subset(sub, int(alloc))
        for (_, sub), alloc in zip(groups, allocations)
        if int(alloc) > 0
    ]
    if not selected:
        return df.iloc[0:0].copy()
    return pd.concat(selected, axis=0).sort_index()


def _bin_single_sweep(
    df: pd.DataFrame,
    *,
    freq_col: str,
    value_col: str,
    bin_hz: float,
) -> pd.DataFrame:
    """Bin one sweep trace into mean/std points on a frequency grid."""
    if df.empty or freq_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()

    freq = pd.to_numeric(df[freq_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    mask = freq.notna() & values.notna()
    if mask.sum() < 2:
        return pd.DataFrame()

    f = freq.loc[mask].to_numpy(dtype=float)
    v = values.loc[mask].to_numpy(dtype=float)
    fmin, fmax = float(f.min()), float(f.max())
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
        return pd.DataFrame()

    edges = np.arange(fmin, fmax + bin_hz, bin_hz)
    if len(edges) < 2:
        edges = np.array([fmin, fmin + float(bin_hz)], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    idx = np.clip(np.digitize(f, edges) - 1, 0, len(edges) - 2)

    binned = (
        pd.DataFrame({"bin": idx, "value": v})
        .groupby("bin", as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            count=("value", "size"),
        )
    )
    if binned.empty:
        return binned

    binned["freq_center"] = binned["bin"].map(
        lambda j, c=centers: float(c[int(j)]) if int(j) < len(c) else float(fmax)
    )
    binned["std"] = binned["std"].fillna(0.0)
    return binned


def build_sweep_average_trace(
    df: pd.DataFrame,
    *,
    signal_col: str,
    freq_col: str = "Frequency",
    sweep_col: str = "Sweep",
    bin_hz: float = 5.0,
) -> pd.DataFrame:
    """Average multiple sweeps onto one shared frequency grid."""
    plot_df = df
    if "IsFrequencyHold" in plot_df.columns:
        plot_df = plot_df.loc[~plot_df["IsFrequencyHold"]].copy()

    if plot_df.empty or freq_col not in plot_df.columns or signal_col not in plot_df.columns:
        return pd.DataFrame()

    sweep_bins: dict[str, pd.DataFrame] = {}
    if sweep_col in plot_df.columns and plot_df[sweep_col].notna().any():
        for sw in sorted(plot_df[sweep_col].dropna().unique()):
            binned = _bin_single_sweep(
                plot_df.loc[plot_df[sweep_col] == sw].copy(),
                freq_col=freq_col,
                value_col=signal_col,
                bin_hz=float(bin_hz),
            )
            if not binned.empty:
                sweep_bins[str(sw)] = binned[["freq_center", "mean", "std"]].copy()
    else:
        binned = _bin_single_sweep(
            plot_df,
            freq_col=freq_col,
            value_col=signal_col,
            bin_hz=float(bin_hz),
        )
        if binned.empty:
            return pd.DataFrame()
        return binned[["freq_center", "mean", "std"]].reset_index(drop=True)

    if not sweep_bins:
        return pd.DataFrame()
    if len(sweep_bins) == 1:
        return next(iter(sweep_bins.values())).reset_index(drop=True)

    all_freqs: list[float] = []
    for binned in sweep_bins.values():
        all_freqs.extend(binned["freq_center"].tolist())
    if not all_freqs:
        return pd.DataFrame()

    half_bin = float(bin_hz) / 2.0
    fmin = min(all_freqs) - half_bin
    fmax = max(all_freqs) + half_bin
    edges = np.arange(fmin, fmax + float(bin_hz), float(bin_hz))
    if len(edges) < 2:
        edges = np.array([fmin, fmin + float(bin_hz)], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0

    avg_vals: list[float] = []
    std_vals: list[float] = []
    for center in centers:
        values: list[float] = []
        for binned in sweep_bins.values():
            mask = (
                (binned["freq_center"] >= center - half_bin)
                & (binned["freq_center"] < center + half_bin)
            )
            matched = binned.loc[mask, "mean"]
            if not matched.empty:
                values.append(float(matched.mean()))
        if values:
            avg_vals.append(float(np.mean(values)))
            std_vals.append(float(np.std(values)) if len(values) > 1 else 0.0)
        else:
            avg_vals.append(np.nan)
            std_vals.append(np.nan)

    return pd.DataFrame(
        {
            "freq_center": centers,
            "mean": avg_vals,
            "std": std_vals,
        }
    ).dropna().reset_index(drop=True)


def plot_sweep_all_points(
    df: pd.DataFrame,
    x_col: str = "Frequency",
    y_col: str = "value",
    color_col: str | None = "Sweep",
    title: str = "",
    height: int = PLOT_HEIGHT,
    mode: str = "markers",
    marker_size: int = 4,
    opacity: float = 0.7,
    y_range: tuple[float, float] | None = None,
    visible_sweeps: set[int] | None = None,
    average_df: pd.DataFrame | None = None,
    show_average_error_bars: bool = True,
) -> go.Figure:
    """Scatter / line plot of all sweep data points.

    Each sweep cycle gets its own colour and a legend entry labelled
    ``Sweep 1``, ``Sweep 2``, etc.

    Parameters
    ----------
    mode : str
        ``"markers"`` | ``"lines"`` | ``"lines+markers"``
    marker_size : int
        Diameter of each point in pixels.
    opacity : float
        Marker opacity 0-1.
    y_range : tuple or None
        Fixed ``(ymin, ymax)`` for the y-axis.  ``None`` = auto-range.
    visible_sweeps : set[int] or None
        Set of sweep indices (0-based) to display.  ``None`` = show all.
    average_df : pd.DataFrame or None
        If provided, overlay a per-frequency average line.  Expected
        columns: ``freq``, ``mean``, ``std``.
    show_average_error_bars : bool
        When *True* and *average_df* is provided, draw ±1 std error
        bars around the average line.
    """
    fig = go.Figure()
    plot_df = df
    if "IsFrequencyHold" in plot_df.columns:
        plot_df = plot_df.loc[~plot_df["IsFrequencyHold"]].copy()

    if color_col and color_col in plot_df.columns:
        sweep_ids = sorted(plot_df[color_col].unique())
        for i, sw in enumerate(sweep_ids):
            # If visible_sweeps is set, hide sweeps not in the set
            is_visible = (
                True if visible_sweeps is None else (int(sw) in visible_sweeps)
            )
            sub = plot_df[plot_df[color_col] == sw]
            colour = _SWEEP_PALETTE[i % len(_SWEEP_PALETTE)]
            fig.add_trace(
                go.Scattergl(
                    x=sub[x_col],
                    y=sub[y_col],
                    mode=mode,
                    name=f"Sweep {int(sw) + 1}",
                    marker=dict(size=marker_size, opacity=opacity, color=colour),
                    line=dict(width=1.5, color=colour),
                    visible=True if is_visible else "legendonly",
                    legendgroup=f"sweep_{int(sw)}",
                )
            )
    else:
        fig.add_trace(
            go.Scattergl(
                x=plot_df[x_col],
                y=plot_df[y_col],
                mode=mode,
                name=y_col,
                marker=dict(size=marker_size, opacity=opacity),
                line=dict(width=1.5),
            )
        )

    # ── Average overlay ─────────────────────────────────────────────
    if average_df is not None and not average_df.empty:
        fig.add_trace(
            go.Scatter(
                x=average_df["freq"],
                y=average_df["mean"],
                mode="lines",
                name="Average",
                line=dict(color="black", width=2.5, dash="solid"),
                error_y=(
                    std_error_bar_style(
                        average_df["std"].fillna(0),
                        color="black",
                    )
                    if show_average_error_bars and "std" in average_df.columns
                    else None
                ),
                legendgroup="avg",
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="closest",
        dragmode="zoom",
        font=dict(size=12),
        showlegend=True,
        xaxis_title="Frequency (Hz)",
        yaxis_title=flow_label(y_col),
    )
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    fig.update_xaxes(title_font=dict(size=12))
    fig.update_yaxes(title_font=dict(size=12))
    return fig


def plot_sweep_per_sweep_average(
    df: pd.DataFrame,
    *,
    signal_col: str,
    freq_col: str = "Frequency",
    sweep_col: str = "Sweep",
    bin_hz: float = 5.0,
    title: str = "",
    height: int = PLOT_HEIGHT,
    mode: str = "lines+markers",
    marker_size: int = 5,
    show_error_bars: bool = True,
    y_range: tuple[float, float] | None = None,
    visible_sweeps: set[int] | None = None,
) -> go.Figure:
    """Plot one binned mean+-std trace per sweep for a single test."""
    fig = go.Figure()
    plot_df = df
    if "IsFrequencyHold" in plot_df.columns:
        plot_df = plot_df.loc[~plot_df["IsFrequencyHold"]].copy()

    if sweep_col not in plot_df.columns or freq_col not in plot_df.columns:
        fig.add_annotation(text="No sweep/frequency data", showarrow=False)
        return fig

    plotted = 0
    sweep_ids = sorted(plot_df[sweep_col].dropna().unique())
    for i, sw in enumerate(sweep_ids):
        if visible_sweeps is not None and int(sw) not in visible_sweeps:
            continue
        sub = plot_df.loc[plot_df[sweep_col] == sw].copy()
        binned = _bin_single_sweep(
            sub,
            freq_col=freq_col,
            value_col=signal_col,
            bin_hz=float(bin_hz),
        )
        if binned.empty:
            continue

        color = _SWEEP_PALETTE[i % len(_SWEEP_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=binned["freq_center"],
                y=binned["mean"],
                mode=mode,
                name=f"Sweep {int(sw) + 1}",
                line=dict(color=color, width=1.8),
                marker=dict(size=marker_size, color=color),
                error_y=(
                    std_error_bar_style(
                        binned["std"],
                        color=color,
                    )
                    if show_error_bars
                    else None
                ),
                legendgroup=f"sweep_{int(sw)}",
            )
        )
        plotted += 1

    if plotted == 0:
        fig.add_annotation(text="No sweep data available", showarrow=False)

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        dragmode="zoom",
        font=dict(size=12),
        showlegend=True,
        xaxis_title="Frequency (Hz)",
        yaxis_title=flow_label(signal_col),
    )
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    fig.update_xaxes(title_font=dict(size=12))
    fig.update_yaxes(title_font=dict(size=12))
    return fig


def plot_sweep_binned(
    binned_df: pd.DataFrame,
    x_col: str = "freq_center",
    y_col: str = "mean",
    std_col: str = "std",
    title: str = "",
    height: int = PLOT_HEIGHT,
    mode: str = "lines+markers",
    marker_size: int = 6,
    show_error_bars: bool = True,
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Binned sweep plot with optional mean +/- std error bars.

    Parameters
    ----------
    show_error_bars : bool
        When *True* (default), draw +/-1 std error bars on the mean.
    y_range : tuple or None
        Fixed ``(ymin, ymax)`` for the y-axis.  ``None`` = auto-range.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=binned_df[x_col],
            y=binned_df[y_col],
            mode=mode,
            name="Mean",
            line=dict(color="blue", width=2),
            marker=dict(size=marker_size),
            error_y=(
                std_error_bar_style(
                    binned_df[std_col].fillna(0),
                    color="blue",
                )
                if show_error_bars and std_col in binned_df.columns
                else None
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Frequency (Hz)",
        yaxis_title=flow_label(y_col),
        hovermode="x unified",
        dragmode="zoom",
        font=dict(size=12),
    )
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    fig.update_xaxes(title_font=dict(size=12))
    fig.update_yaxes(title_font=dict(size=12))
    return fig


def export_html(
    fig: go.Figure,
    filename: str,
    export_dir: Path | str = None,
) -> Path:
    """Export Plotly figure to standalone HTML file.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure to export
    filename : str
        Output filename (without path)
    export_dir : Path or str, optional
        Export directory. If None, uses config.DATA_EXPORT_DIR
    
    Returns
    -------
    Path
        Path to exported HTML file
    """
    if export_dir is None:
        from ..data.config import DATA_EXPORT_DIR
        export_dir = DATA_EXPORT_DIR
    else:
        export_dir = Path(export_dir)
    
    export_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = export_dir / filename
    export_fig = _prepare_figure_for_html_export(fig)
    export_fig.write_html(
        str(filepath),
        include_plotlyjs=True,
        full_html=True,
    )
    
    return filepath


def _prepare_figure_for_html_export(fig: go.Figure) -> go.Figure:
    """Return an HTML-export-safe copy of *fig*.

    Standalone HTML files have been unreliable with ``Scattergl`` traces in
    some environments, resulting in axes with no visible data. Convert those
    traces to standard SVG ``scatter`` traces before saving.
    """
    export_fig = go.Figure(fig)
    converted_traces = []
    for trace in export_fig.data:
        trace_dict = trace.to_plotly_json()
        if trace_dict.get("type") == "scattergl":
            trace_dict["type"] = "scatter"
        converted_traces.append(trace_dict)

    export_fig.data = ()
    for trace_dict in converted_traces:
        export_fig.add_trace(trace_dict)
    export_fig.update_layout(fig.layout)
    return export_fig
