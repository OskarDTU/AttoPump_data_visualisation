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
- ``plot_sweep_all_points``         — raw scatter coloured by sweep cycle.
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

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..data.config import PLOT_HEIGHT
from .shared import flow_label, time_label


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
    fig = px.histogram(
        df,
        x=y_col,
        nbins=nbins,
        title=title,
        labels={y_col: flow_label(y_col)},
    )
    fig.update_layout(
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
_SWEEP_PALETTE = px.colors.qualitative.Plotly


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
        When *True* and *average_df* is provided, draw ±1 std band
        around the average line.
    """
    fig = go.Figure()

    if color_col and color_col in df.columns:
        sweep_ids = sorted(df[color_col].unique())
        for i, sw in enumerate(sweep_ids):
            # If visible_sweeps is set, hide sweeps not in the set
            is_visible = (
                True if visible_sweeps is None else (int(sw) in visible_sweeps)
            )
            sub = df[df[color_col] == sw]
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
                x=df[x_col],
                y=df[y_col],
                mode=mode,
                name=y_col,
                marker=dict(size=marker_size, opacity=opacity),
                line=dict(width=1.5),
            )
        )

    # ── Average overlay ─────────────────────────────────────────────
    if average_df is not None and not average_df.empty:
        # ±1 std shaded band (behind the mean line)
        if show_average_error_bars and "std" in average_df.columns:
            y_upper = average_df["mean"] + average_df["std"].fillna(0)
            y_lower = average_df["mean"] - average_df["std"].fillna(0)
            fig.add_trace(
                go.Scatter(
                    x=average_df["freq"],
                    y=y_upper,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup="avg",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=average_df["freq"],
                    y=y_lower,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    name="Avg ±1 std",
                    fillcolor="rgba(0, 0, 0, 0.15)",
                    hoverinfo="skip",
                    legendgroup="avg",
                )
            )

        # Mean line (on top of the band)
        fig.add_trace(
            go.Scattergl(
                x=average_df["freq"],
                y=average_df["mean"],
                mode="lines",
                name="Average",
                line=dict(color="black", width=2.5, dash="solid"),
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
    """Binned sweep plot with optional mean +/- std band.

    Parameters
    ----------
    show_error_bars : bool
        When *True* (default), draw a +/-1 std shaded band around the mean.
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
        )
    )

    if show_error_bars and std_col in binned_df.columns:
        y_upper = binned_df[y_col] + binned_df[std_col].fillna(0)
        y_lower = binned_df[y_col] - binned_df[std_col].fillna(0)

        fig.add_trace(
            go.Scatter(
                x=binned_df[x_col],
                y=y_upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=binned_df[x_col],
                y=y_lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name="±1 std",
                fillcolor="rgba(0, 100, 255, 0.2)",
                hoverinfo="skip",
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
