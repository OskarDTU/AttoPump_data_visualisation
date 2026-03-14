"""Shared plot helpers — eliminates duplication across plot modules.

Centralises utility functions that were previously copy-pasted across
``plot_generator.py``, ``analysis_plots.py``, and
``bar_comparison_plots.py``.
"""

from __future__ import annotations

import re

from plotly.colors import qualitative

# ── Shared color palettes ───────────────────────────────────────────────
PALETTE = (
    qualitative.Plotly
    + qualitative.Set2
    + qualitative.Dark24
)

PUMP_PALETTE = (
    qualitative.Bold
    + qualitative.Plotly
    + qualitative.Set1
)


def color_to_rgba(color: str, alpha: float = 0.2) -> str:
    """Convert any Plotly color string to ``rgba(R, G, B, alpha)``.

    Handles ``#RRGGBB``, ``rgb(R, G, B)``, and bare color names.
    """
    m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", color)
    if m:
        return f"rgba({m.group(1)}, {m.group(2)}, {m.group(3)}, {alpha})"

    hex_color = color.lstrip("#")
    if len(hex_color) == 6:
        try:
            r = int(hex_color[:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f"rgba({r}, {g}, {b}, {alpha})"
        except ValueError:
            pass

    return f"rgba(128, 128, 128, {alpha})"


def std_error_bar_style(
    std_values,
    *,
    color: str,
    alpha: float = 0.45,
    thickness: float = 1.1,
    width: float = 3.0,
) -> dict:
    """Return a Plotly ``error_y`` config for ±1 standard-deviation whiskers."""
    return {
        "type": "data",
        "array": std_values,
        "visible": True,
        "color": color_to_rgba(color, alpha),
        "thickness": thickness,
        "width": width,
    }


def flow_label(col: str) -> str:
    """Return a human-readable axis label for a flow column."""
    return f"{col} (µL/min)" if "flow" in col.lower() else col


def time_label(col: str) -> str:
    """Return a human-readable axis label for a time column."""
    return "Time (seconds)" if col.lower() in ("t_s", "elapsed_s", "t") else "Time"
