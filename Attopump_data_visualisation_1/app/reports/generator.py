"""Report generation engine — produces self-contained HTML report packages.

This module takes a **report definition** (which entries to include, what
comparisons to draw) and generates a single self-contained HTML file with
embedded interactive Plotly charts, summary tables, and metadata.

Architecture
------------
1. **ReportSection** — one logical block in the report (a chart, a table,
   or a text note).
2. **ReportDefinition** — the full specification: title, author, which
   BAR/pump entries to include, which comparisons to generate, notes.
3. ``build_report_html()`` — the main entry point: takes a definition,
   loads data, generates plots, and returns an HTML string.
4. ``save_report()`` — writes the HTML to disk (Downloads folder).

The HTML uses Plotly.js (CDN) so each figure is fully interactive in any
browser — the recipient does not need Python or Streamlit.
"""

from __future__ import annotations

import html as html_mod
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


# ══════════════════════════════════════════════════════════════════════════
# REPORT DEFINITION
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ReportSection:
    """One section in a report.

    Attributes
    ----------
    kind : str
        ``"plot"``, ``"table"``, ``"text"``, ``"heading"``, ``"divider"``.
    title : str
        Section heading.
    content : Any
        - For ``"plot"``: a ``plotly.graph_objects.Figure``.
        - For ``"table"``: a ``pd.DataFrame``.
        - For ``"text"``: a plain string.
        - For ``"heading"``/``"divider"``: ignored.
    description : str
        Explanatory note shown below the section.
    """

    kind: str
    title: str = ""
    content: Any = None
    description: str = ""
    collapsible: bool = False
    collapsed: bool = False


@dataclass
class AxisBounds:
    """Optional manual axis limits for one figure family."""

    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None


@dataclass
class ReportDefinition:
    """Full report specification.

    Attributes
    ----------
    title : str
        Report title.
    author : str
        Report author.
    entry_ids : list[str]
        BAR / pump entry IDs to include.
    comparisons : list[str]
        Which comparison types to generate.  Subset of:
        ``"sweep_overlay"``, ``"sweep_relative"``, ``"boxplots"``,
        ``"histograms"``, ``"summary_table"``, ``"individual_sweeps"``,
        ``"global_average"``, ``"constant_time_series"``,
        ``"raw_points"``, ``"std_vs_mean"``, ``"best_region"``,
        ``"correlation"``.
    notes : str
        Free-text notes included in the report header.
    bin_hz : float
        Per-test frequency bin width for sweep analysis.
    avg_bin_hz : float
        Cross-test / cross-target averaging bin width.
    show_error_bars : bool
        Whether to include ±1 std error bars on plots.
    show_individual_tests : bool
        Whether to show per-test lines behind bar averages.
    show_raw_all_sweeps : bool
        Whether to include the raw all-sweeps layer for individual sweep tests.
    plot_mode : str
        Legacy fallback curve rendering style for compatible report plots.
    plot_modes : dict[str, str]
        Optional per-plot-family curve rendering styles.
    marker_size : int
        Default marker size for compatible report plots.
    opacity : float
        Default trace opacity for raw-point and time-series plots.
    max_raw_points : int
        Per-test cap for raw-point plots in the exported report.
    mean_threshold_pct : int
        High-flow retention percentage for best-region analysis.
    std_threshold_pct : int
        Low-variability retention percentage within the high-flow subset.
    sweep_axis : AxisBounds
        Manual axis overrides for frequency-vs-flow plots.
    relative_axis : AxisBounds
        Manual axis overrides for relative sweep plots.
    time_axis : AxisBounds
        Manual axis overrides for time-series plots.
    variability_axis : AxisBounds
        Manual axis overrides for mean-vs-std plots.
    selection_mode : str
        ``"pumps"`` or ``"sub_groups"``.
    overlay_entry_ids : list[str]
        Selected report-target IDs included in the all-tests overlay view.
    auto_use_recommended_avg_bin : bool
        Whether average-style plots should automatically use the recommended
        averaging bin when available.
    overlay_include_sweep_drilldown : bool
        Whether the report should include per-target all-sweeps drill-down
        plots for the focused overlay targets.
    """

    title: str = "AttoPump Test Report"
    author: str = ""
    entry_ids: list[str] = field(default_factory=list)
    comparisons: list[str] = field(default_factory=lambda: [
        "sweep_overlay", "summary_table", "boxplots",
    ])
    notes: str = ""
    bin_hz: float = 5.0
    avg_bin_hz: float = 3.0
    show_error_bars: bool = True
    show_individual_tests: bool = False
    show_raw_all_sweeps: bool = True
    plot_mode: str = "lines+markers"
    plot_modes: dict[str, str] = field(default_factory=dict)
    marker_size: int = 6
    opacity: float = 0.8
    max_raw_points: int = 50_000
    mean_threshold_pct: int = 75
    std_threshold_pct: int = 10
    sweep_axis: AxisBounds = field(default_factory=AxisBounds)
    relative_axis: AxisBounds = field(default_factory=AxisBounds)
    time_axis: AxisBounds = field(default_factory=AxisBounds)
    variability_axis: AxisBounds = field(default_factory=AxisBounds)
    selection_mode: str = "pumps"
    overlay_entry_ids: list[str] = field(default_factory=list)
    auto_use_recommended_avg_bin: bool = True
    overlay_include_sweep_drilldown: bool = False


# Available comparison types with labels
COMPARISON_OPTIONS: dict[str, str] = {
    "sweep_overlay": "📈 Sweep overlay (entries and tests)",
    "sweep_relative": "📉 Relative sweep comparison (0–100 %)",
    "individual_sweeps": "📊 Individual sweep diagnostics",
    "global_average": "📏 Global average curve",
    "boxplots": "📦 Flow distribution boxplots",
    "histograms": "📊 Flow histograms",
    "summary_table": "📋 Summary statistics table",
    "constant_time_series": "⏱️ Constant flow vs time",
    "raw_points": "⚡ All raw sweep points",
    "std_vs_mean": "📐 Std vs Mean scatter",
    "best_region": "🎯 Best-region analysis (high mean, low std)",
    "correlation": "🔗 Inter-test correlation heatmap",
}


# ══════════════════════════════════════════════════════════════════════════
# REPORT STORE — save / load report definitions
# ══════════════════════════════════════════════════════════════════════════

_REPORTS_PATH = Path(__file__).parent.parent / "data" / "saved_reports.json"


def load_saved_reports() -> dict[str, dict]:
    """Load saved report definitions from disk."""
    if _REPORTS_PATH.exists():
        with open(_REPORTS_PATH, "r") as f:
            raw = json.load(f)
        return raw.get("reports", {})
    return {}


def save_report_definition(name: str, defn: ReportDefinition) -> None:
    """Persist a report definition by name."""
    if _REPORTS_PATH.exists():
        with open(_REPORTS_PATH, "r") as f:
            raw = json.load(f)
    else:
        raw = {"_description": "Saved report definitions.", "reports": {}}

    raw.setdefault("reports", {})[name] = {
        "title": defn.title,
        "author": defn.author,
        "entry_ids": defn.entry_ids,
        "comparisons": defn.comparisons,
        "notes": defn.notes,
        "bin_hz": defn.bin_hz,
        "avg_bin_hz": defn.avg_bin_hz,
        "show_error_bars": defn.show_error_bars,
        "show_individual_tests": defn.show_individual_tests,
        "show_raw_all_sweeps": defn.show_raw_all_sweeps,
        "plot_mode": defn.plot_mode,
        "plot_modes": defn.plot_modes,
        "marker_size": defn.marker_size,
        "opacity": defn.opacity,
        "max_raw_points": defn.max_raw_points,
        "mean_threshold_pct": defn.mean_threshold_pct,
        "std_threshold_pct": defn.std_threshold_pct,
        "sweep_axis": asdict(defn.sweep_axis),
        "relative_axis": asdict(defn.relative_axis),
        "time_axis": asdict(defn.time_axis),
        "variability_axis": asdict(defn.variability_axis),
        "selection_mode": defn.selection_mode,
        "overlay_entry_ids": defn.overlay_entry_ids,
        "auto_use_recommended_avg_bin": defn.auto_use_recommended_avg_bin,
        "overlay_include_sweep_drilldown": defn.overlay_include_sweep_drilldown,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(_REPORTS_PATH, "w") as f:
        json.dump(raw, f, indent=2)


def delete_saved_report(name: str) -> None:
    """Remove a saved report definition."""
    if not _REPORTS_PATH.exists():
        return
    with open(_REPORTS_PATH, "r") as f:
        raw = json.load(f)
    raw.get("reports", {}).pop(name, None)
    with open(_REPORTS_PATH, "w") as f:
        json.dump(raw, f, indent=2)


def load_report_definition(name: str) -> ReportDefinition | None:
    """Load a single saved report definition by name."""
    reports = load_saved_reports()
    d = reports.get(name)
    if d is None:
        return None

    def _load_axis_bounds(key: str) -> AxisBounds:
        raw = d.get(key, {})
        if not isinstance(raw, dict):
            return AxisBounds()
        return AxisBounds(
            x_min=raw.get("x_min"),
            x_max=raw.get("x_max"),
            y_min=raw.get("y_min"),
            y_max=raw.get("y_max"),
        )

    return ReportDefinition(
        title=d.get("title", name),
        author=d.get("author", ""),
        entry_ids=d.get("entry_ids", []),
        comparisons=d.get("comparisons", []),
        notes=d.get("notes", ""),
        bin_hz=d.get("bin_hz", 5.0),
        avg_bin_hz=d.get("avg_bin_hz", 3.0),
        show_error_bars=d.get("show_error_bars", True),
        show_individual_tests=d.get("show_individual_tests", False),
        show_raw_all_sweeps=d.get("show_raw_all_sweeps", True),
        plot_mode=d.get("plot_mode", "lines+markers"),
        plot_modes=(
            d.get("plot_modes", {})
            if isinstance(d.get("plot_modes", {}), dict)
            else {}
        ),
        marker_size=d.get("marker_size", 6),
        opacity=d.get("opacity", 0.8),
        max_raw_points=d.get("max_raw_points", 50_000),
        mean_threshold_pct=d.get("mean_threshold_pct", 75),
        std_threshold_pct=d.get("std_threshold_pct", 10),
        sweep_axis=_load_axis_bounds("sweep_axis"),
        relative_axis=_load_axis_bounds("relative_axis"),
        time_axis=_load_axis_bounds("time_axis"),
        variability_axis=_load_axis_bounds("variability_axis"),
        selection_mode=d.get("selection_mode", "pumps"),
        overlay_entry_ids=d.get("overlay_entry_ids", []),
        auto_use_recommended_avg_bin=d.get("auto_use_recommended_avg_bin", True),
        overlay_include_sweep_drilldown=d.get("overlay_include_sweep_drilldown", False),
    )


# ══════════════════════════════════════════════════════════════════════════
# HTML GENERATION
# ══════════════════════════════════════════════════════════════════════════

def _fig_to_html(fig: go.Figure, include_js: bool = False) -> str:
    """Convert a Plotly figure to an HTML <div> snippet.

    Parameters
    ----------
    include_js : bool
        If True, include the Plotly.js library in this div (only needed
        for the first figure; subsequent ones reuse the CDN script).
    """
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_js else False,
        full_html=False,
        config={"responsive": True, "displayModeBar": True},
    )


def _df_to_html_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a styled HTML table."""
    return df.to_html(
        index=False,
        classes="report-table",
        border=0,
        float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else str(x),
    )


def _escape(text: str) -> str:
    """HTML-escape user-supplied text."""
    return html_mod.escape(text)


def _split_section_description(description: str) -> tuple[str, str]:
    """Split a long section note into a visible subtitle and hidden detail."""
    paragraphs = [part.strip() for part in str(description or "").split("\n\n") if part.strip()]
    if not paragraphs:
        return "", ""

    priority_prefixes = (
        "What you are seeing:",
        "Tabulated",
        "Summary:",
        "What this table shows:",
    )
    subtitle_index = next(
        (
            idx
            for idx, paragraph in enumerate(paragraphs)
            if paragraph.startswith(priority_prefixes)
        ),
        0,
    )
    subtitle = paragraphs.pop(subtitle_index)
    details = "\n\n".join(paragraphs)
    return subtitle, details


_CSS = """
<style>
  :root {
    --bg: #ffffff;
    --fg: #1a1a2e;
    --accent: #0f3460;
    --muted: #6c757d;
    --border: #dee2e6;
    --card-bg: #f8f9fa;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: var(--fg);
    background: var(--bg);
    margin: 0;
    padding: 0;
    line-height: 1.6;
  }
  .report-container {
    max-width: 1500px;
    margin: 0 auto;
    padding: 40px 24px;
  }
  .report-header {
    border-bottom: 3px solid var(--accent);
    padding-bottom: 20px;
    margin-bottom: 30px;
  }
  .report-header h1 {
    font-size: 2em;
    color: var(--accent);
    margin: 0 0 8px 0;
  }
  .report-meta {
    color: var(--muted);
    font-size: 0.9em;
  }
  .report-notes {
    background: var(--card-bg);
    border-left: 4px solid var(--accent);
    padding: 12px 16px;
    margin: 20px 0;
    border-radius: 0 4px 4px 0;
    white-space: pre-wrap;
  }
  .section { margin: 30px 0; }
  .section h2 {
    font-size: 1.4em;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 16px;
  }
  .section h3 {
    font-size: 1.15em;
    color: var(--fg);
    margin: 20px 0 10px 0;
  }
  .section-subtitle {
    color: #364152;
    font-size: 0.98em;
    margin: -4px 0 14px 0;
    max-width: 980px;
  }
  .section-description {
    color: var(--muted);
    font-size: 0.9em;
    margin-bottom: 12px;
    white-space: pre-line;
  }
  details.section-details,
  details.section-table {
    margin: 12px 0;
    padding: 0;
  }
  details.section-details > summary,
  details.section-table > summary {
    cursor: pointer;
    color: var(--accent);
    font-weight: 600;
    list-style: none;
    margin-bottom: 10px;
  }
  details.section-details > summary::-webkit-details-marker,
  details.section-table > summary::-webkit-details-marker {
    display: none;
  }
  details.section-details[open] > summary,
  details.section-table[open] > summary {
    margin-bottom: 12px;
  }
  .report-table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 0.9em;
  }
  .report-table th {
    background: var(--accent);
    color: white;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
  }
  .report-table td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }
  .report-table tr:nth-child(even) { background: var(--card-bg); }
  .report-table tr:hover { background: #e8f0fe; }
  .entry-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
  }
  .entry-card h3 { margin-top: 0; }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 600;
  }
  .badge-success { background: #d4edda; color: #155724; }
  .badge-fail { background: #f8d7da; color: #721c24; }
  .badge-sweep { background: #cce5ff; color: #004085; }
  .badge-constant { background: #fff3cd; color: #856404; }
  .plot-container { margin: 16px 0; width: 100%; }
  .report-footer {
    border-top: 1px solid var(--border);
    margin-top: 40px;
    padding-top: 16px;
    color: var(--muted);
    font-size: 0.8em;
    text-align: center;
  }
  hr.section-divider {
    border: none;
    border-top: 2px solid var(--border);
    margin: 30px 0;
  }
  @media print {
    .report-container { padding: 20px; }
    .plot-container { break-inside: avoid; }
  }
</style>
"""


def build_report_html(
    defn: ReportDefinition,
    sections: list[ReportSection],
    entry_metadata: dict[str, dict] | None = None,
) -> str:
    """Assemble a full self-contained HTML report.

    Parameters
    ----------
    defn : ReportDefinition
        The report spec (title, author, notes, etc.).
    sections : list[ReportSection]
        Pre-built sections with plots, tables, and text.
    entry_metadata : dict
        Optional dict of ``{entry_id: {"display_name": ..., "tests": ...}}``
        for rendering the entry summary cards.

    Returns
    -------
    str
        Complete HTML document.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    first_plot = True  # include plotly.js only once

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html lang='en'>")
    parts.append("<head>")
    parts.append(f"<title>{_escape(defn.title)}</title>")
    parts.append('<meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(_CSS)
    parts.append("</head>")
    parts.append("<body>")
    parts.append('<div class="report-container">')

    # ── Header ──────────────────────────────────────────────────
    parts.append('<div class="report-header">')
    parts.append(f"<h1>{_escape(defn.title)}</h1>")
    meta_items = []
    if defn.author:
        meta_items.append(f"<strong>Author:</strong> {_escape(defn.author)}")
    meta_items.append(f"<strong>Generated:</strong> {now}")
    meta_items.append(
        f"<strong>Entries:</strong> {len(defn.entry_ids)}"
    )
    parts.append(f'<div class="report-meta">{" &nbsp;|&nbsp; ".join(meta_items)}</div>')
    parts.append("</div>")

    # ── Notes ───────────────────────────────────────────────────
    if defn.notes.strip():
        parts.append(f'<div class="report-notes">{_escape(defn.notes)}</div>')

    # ── Entry metadata cards ────────────────────────────────────
    if entry_metadata:
        parts.append('<div class="section">')
        parts.append("<h2>📋 Entries Included</h2>")
        for eid, meta in entry_metadata.items():
            parts.append('<div class="entry-card">')
            dname = meta.get("display_name", eid)
            parts.append(f"<h3>{_escape(dname)}</h3>")
            tests = meta.get("tests", [])
            if tests:
                parts.append("<ul>")
                for t in tests:
                    badge = ""
                    if t.get("test_type") == "sweep":
                        badge = ' <span class="badge badge-sweep">sweep</span>'
                    elif t.get("test_type") == "constant":
                        badge = ' <span class="badge badge-constant">constant</span>'
                    if t.get("success") is True:
                        badge += ' <span class="badge badge-success">✓ pass</span>'
                    elif t.get("success") is False:
                        badge += ' <span class="badge badge-fail">✗ fail</span>'
                    desc = f" — {_escape(t['description'])}" if t.get("description") else ""
                    parts.append(
                        f"<li><code>{_escape(t['folder'])}</code>{desc}{badge}</li>"
                    )
                parts.append("</ul>")
            notes = meta.get("notes", "")
            if notes:
                parts.append(f"<p><em>{_escape(notes)}</em></p>")
            parts.append("</div>")
        parts.append("</div>")

    # ── Sections ────────────────────────────────────────────────
    for sec in sections:
        if sec.kind == "divider":
            parts.append('<hr class="section-divider">')
            continue

        parts.append('<div class="section">')

        if sec.kind == "heading":
            parts.append(f"<h2>{_escape(sec.title)}</h2>")
        elif sec.kind == "text":
            if sec.title:
                parts.append(f"<h3>{_escape(sec.title)}</h3>")
            parts.append(f"<p>{_escape(str(sec.content))}</p>")
        elif sec.kind == "plot":
            if sec.title:
                parts.append(f"<h3>{_escape(sec.title)}</h3>")
            if sec.description:
                subtitle, details = _split_section_description(sec.description)
                if subtitle:
                    parts.append(
                        f'<div class="section-subtitle">{_escape(subtitle)}</div>'
                    )
                if details:
                    parts.append(
                        "<details class='section-details'>"
                        "<summary>More detail</summary>"
                        f"<div class='section-description'>{_escape(details)}</div>"
                        "</details>"
                    )
            parts.append('<div class="plot-container">')
            parts.append(_fig_to_html(sec.content, include_js=first_plot))
            first_plot = False
            parts.append("</div>")
        elif sec.kind == "table":
            if sec.title:
                parts.append(f"<h3>{_escape(sec.title)}</h3>")
            if sec.description:
                subtitle, details = _split_section_description(sec.description)
                if subtitle:
                    parts.append(
                        f'<div class="section-subtitle">{_escape(subtitle)}</div>'
                    )
                if details:
                    parts.append(
                        "<details class='section-details'>"
                        "<summary>More detail</summary>"
                        f"<div class='section-description'>{_escape(details)}</div>"
                        "</details>"
                    )
            table_html = _df_to_html_table(sec.content)
            if sec.collapsible:
                summary_label = "Show table"
                open_attr = "" if sec.collapsed else " open"
                parts.append(
                    f"<details class='section-table'{open_attr}>"
                    f"<summary>{summary_label}</summary>"
                    f"{table_html}"
                    "</details>"
                )
            else:
                parts.append(table_html)

        parts.append("</div>")

    # ── Footer ──────────────────────────────────────────────────
    parts.append('<div class="report-footer">')
    parts.append(
        f"Generated by AttoPump Data Visualisation &middot; {now}"
    )
    parts.append("</div>")

    parts.append("</div>")  # report-container
    parts.append("</body></html>")
    return "\n".join(parts)


def save_report(html: str, filename: str, directory: Path | None = None) -> Path:
    """Write report HTML to disk.

    Parameters
    ----------
    html : str
        Complete HTML string.
    filename : str
        Output file name (e.g. ``"report_2026-03-15.html"``).
    directory : Path or None
        Target directory.  Defaults to ``~/Downloads``.

    Returns
    -------
    Path
        Absolute path of the saved file.
    """
    if directory is None:
        directory = Path.home() / "Downloads"
    directory.mkdir(parents=True, exist_ok=True)

    # Sanitise filename
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in filename)
    if not safe.endswith(".html"):
        safe += ".html"

    path = directory / safe
    path.write_text(html, encoding="utf-8")
    return path
