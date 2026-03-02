"""Plot generation with Plotly (interactive HTML output)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import PLOT_HEIGHT


# ============================================================================
# HELPERS
# ============================================================================

def _flow_label(col: str) -> str:
    return f"{col} (µL/min)" if "flow" in col.lower() else col


def _time_label(col: str) -> str:
    return "Time (seconds)" if col.lower() in ["t_s", "elapsed_s", "t"] else "Time"


def plot_time_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    height: int = PLOT_HEIGHT,
    mode: str = "lines",
    marker_size: int = 4,
    opacity: float = 1.0,
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
        font=dict(size=12),
        xaxis_title=_time_label(x_col),
        yaxis_title=_flow_label(y_col),
    )
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
        yaxis_title=_flow_label(y_col),
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
        labels={y_col: _flow_label(y_col)},
    )
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x",
        font=dict(size=12),
        xaxis_title=_flow_label(y_col),
        yaxis_title="Frequency (count)",
    )
    fig.update_xaxes(title_font=dict(size=12))
    fig.update_yaxes(title_font=dict(size=12))
    return fig


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
) -> go.Figure:
    """Scatter / line plot of all sweep data points.

    Parameters
    ----------
    mode : str
        ``"markers"`` | ``"lines"`` | ``"lines+markers"``
    marker_size : int
        Diameter of each point in pixels.
    opacity : float
        Marker opacity 0-1.
    """
    if color_col and color_col in df.columns:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            color_discrete_sequence=px.colors.qualitative.Set3,
            title=title,
            labels={
                x_col: "Frequency (Hz)",
                y_col: _flow_label(y_col),
                color_col: "Sweep #",
            },
            render_mode="webgl",
        )
        fig.update_traces(
            marker=dict(size=marker_size, opacity=opacity),
            mode=mode,
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=title,
            labels={x_col: "Frequency (Hz)", y_col: _flow_label(y_col)},
            render_mode="webgl",
        )
        fig.update_traces(
            marker=dict(size=marker_size, opacity=opacity),
            mode=mode,
        )

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="closest",
        font=dict(size=12),
        showlegend=True,
    )
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
) -> go.Figure:
    """Binned sweep plot with mean ± std band."""
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

    if std_col in binned_df.columns:
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
        yaxis_title=_flow_label(y_col),
        hovermode="x unified",
        font=dict(size=12),
    )
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
        from .config import DATA_EXPORT_DIR
        export_dir = DATA_EXPORT_DIR
    else:
        export_dir = Path(export_dir)
    
    export_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = export_dir / filename
    fig.write_html(str(filepath))
    
    return filepath
