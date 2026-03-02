"""Plot generation with Plotly (interactive HTML output)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import PLOT_HEIGHT


def plot_time_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """Create time series line plot with proper axis labels."""
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title,
        labels={
            x_col: "Time (seconds)" if x_col.lower() in ['t_s', 'elapsed_s'] else "Time",
            y_col: f"{y_col} (µL/min)" if "flow" in y_col.lower() else y_col
        },
    )
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        font=dict(size=12),
    )
    fig.update_xaxes(title_font=dict(size=12))
    fig.update_yaxes(title_font=dict(size=12))
    return fig


def plot_constant_frequency_boxplot(
    df: pd.DataFrame,
    x_col: str = "Time_Window",
    y_col: str = "flow",
    title: str = "",
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """Create boxplot for constant frequency test showing distribution over time windows.
    
    Displays quartiles, median, and outliers to show spread and confidence intervals.
    """
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        title=title,
        labels={
            x_col: "Time Window (seconds)",
            y_col: f"{y_col} (µL/min)" if "flow" in y_col.lower() else y_col
        },
        points="outliers",  # Show outliers as individual points
    )
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        showlegend=True,
        font=dict(size=12),
    )
    fig.update_xaxes(title_font=dict(size=12))
    fig.update_yaxes(title_font=dict(size=12))
    fig.update_traces(boxmean="sd")  # Show mean and std dev
    return fig


def plot_flow_histogram(
    df: pd.DataFrame,
    y_col: str = "flow",
    title: str = "",
    height: int = PLOT_HEIGHT,
) -> go.Figure:
    """Create histogram of flow rates for constant frequency test."""
    fig = px.histogram(
        df,
        x=y_col,
        nbins=30,
        title=title,
        labels={
            y_col: f"{y_col} (µL/min)" if "flow" in y_col.lower() else y_col
        },
    )
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x",
        font=dict(size=12),
        xaxis_title=f"{y_col} (µL/min)" if "flow" in y_col.lower() else y_col,
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
) -> go.Figure:
    """Create scatter plot of all sweep points with sweep coloring and legend."""
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
                y_col: f"{y_col} (µL/min)" if "flow" in y_col.lower() else y_col,
                color_col: "Sweep #"
            },
        )
        # Update legend to show sweep numbers clearly
        fig.update_traces(marker=dict(size=6, opacity=0.7))
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=title,
            labels={
                x_col: "Frequency (Hz)",
                y_col: f"{y_col} (µL/min)" if "flow" in y_col.lower() else y_col
            },
        )
        fig.update_traces(marker=dict(size=6))
    
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
) -> go.Figure:
    """Create binned sweep plot with mean ± std band."""
    fig = go.Figure()
    
    # Mean line
    fig.add_trace(
        go.Scatter(
            x=binned_df[x_col],
            y=binned_df[y_col],
            mode="lines+markers",
            name="Mean",
            line=dict(color="blue", width=2),
            marker=dict(size=6),
        )
    )
    
    # Std error band
    if std_col in binned_df.columns:
        y_upper = binned_df[y_col] + binned_df[std_col]
        y_lower = binned_df[y_col] - binned_df[std_col]
        
        # Upper bound
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
        
        # Lower bound with fill
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
        yaxis_title=f"{y_col} (µL/min)" if "flow" in y_col.lower() else y_col,
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
