"""Shared thesis-based plot explanations for analysis and reporting views."""

from __future__ import annotations

import streamlit as st


PLOT_GUIDANCE: dict[str, dict[str, str]] = {
    "time_series": {
        "title": "Time Series Overlay",
        "purpose": (
            "Use this first when checking raw temporal behaviour. In the thesis, time-based views were used to diagnose drift, "
            "warm-up effects, settling, and abnormal sweeps before moving to frequency-resolved summaries."
        ),
        "method": (
            "The raw measurements are kept in time order and plotted directly against time. No frequency binning is applied, so the view "
            "preserves transient behaviour and any temporal mismatch between repeated sweeps."
        ),
        "interpretation": (
            "Look for baseline drift, early low-performing sweeps, sudden step changes, or gradual decay. If the trace changes systematically "
            "with time, that should be understood before trusting later frequency-domain averages."
        ),
    },
    "constant_time_series": {
        "title": "Constant Flow vs Time",
        "purpose": (
            "Use this to check whether a constant-frequency hold is stable in time. This follows the thesis logic of validating temporal effects "
            "before drawing performance conclusions from summary statistics."
        ),
        "method": (
            "The cleaned constant-frequency signal is plotted directly against time and can then be paired with correlation or slope estimates to "
            "quantify drift."
        ),
        "interpretation": (
            "A flat trace indicates stable output. Upward or downward trends indicate settling, decay, or another time effect that should be stated "
            "explicitly in the report."
        ),
    },
    "boxplots": {
        "title": "Distribution Boxplots",
        "purpose": (
            "The thesis used boxplots as the first exploratory layer to visualise distributions, identify outliers, and compare spread between tests "
            "before doing frequency-resolved analysis."
        ),
        "method": (
            "Raw measurements are summarized by median, quartiles, whiskers, and outliers. Depending on the view, the app shows per-test boxplots "
            "or pooled boxplots across all selected measurements."
        ),
        "interpretation": (
            "Use the median to compare typical performance, the IQR to judge spread, and the upper whisker/outliers to see whether rare high-output "
            "events are driving the apparent performance."
        ),
    },
    "histograms": {
        "title": "Distribution Histograms",
        "purpose": (
            "These complement boxplots by showing how the full distribution is shaped. The thesis used exploratory distribution plots to decide whether "
            "frequency-resolved follow-up analysis was necessary."
        ),
        "method": (
            "Measurements are grouped by value into bins and plotted as counts or densities, either per test or pooled by selected target."
        ),
        "interpretation": (
            "Look for skewness, multiple modes, or very broad tails. A strongly right-skewed distribution often means that high performance occurs "
            "only in limited frequency regions rather than throughout the sweep."
        ),
    },
    "individual_sweeps": {
        "title": "Individual Sweep Diagnostics",
        "purpose": (
            "The thesis treated per-sweep plots as a core diagnostic layer for repeatability. They answer whether the pump reproduces the same response "
            "shape from sweep to sweep and whether any early or late sweeps should be excluded."
        ),
        "method": (
            "Each sweep is reconstructed in frequency, then shown separately either as binned sweep curves or as a raw all-sweeps layer. This makes "
            "cycle-to-cycle variation visible before any grand averaging."
        ),
        "interpretation": (
            "Consistent sweep shapes indicate repeatability. Large vertical separation, shifted peaks, or weak early sweeps suggest hysteresis, air, "
            "priming effects, or depletion effects that can bias a test-level average."
        ),
    },
    "sweep_overlay": {
        "title": "Binned Sweep Overlay",
        "purpose": (
            "The thesis used combined frequency overlays to compare tests, patches, or grouped devices after sweep-level quality checks were complete."
        ),
        "method": (
            "Data are frequency-binned to compensate for timing uncertainty between sweep start and data acquisition, then mean curves are overlaid. "
            "Optional error bars summarize the standard deviation within each test or target."
        ),
        "interpretation": (
            "Use this to compare where peaks occur, how broad the useful frequency bands are, and how much uncertainty surrounds them. Large error bars in "
            "high-output regions mean those peaks are less repeatable than the mean line alone suggests."
        ),
    },
    "sweep_relative": {
        "title": "Relative Sweep Comparison",
        "purpose": (
            "This mirrors the thesis use of normalized sweep profiles: compare shape consistency independently of absolute magnitude."
        ),
        "method": (
            "Each curve is normalized to its own range so the plot emphasises structural features such as shared peaks, shoulders, and drop-offs."
        ),
        "interpretation": (
            "If normalized curves line up well, the devices share the same qualitative response shape even when their absolute performance differs."
        ),
    },
    "raw_points": {
        "title": "All Raw Sweep Points",
        "purpose": (
            "Use this when you want the least processed view of the sweep data. It supports the thesis workflow of checking raw sweep structure before "
            "trusting aggregated comparisons."
        ),
        "method": (
            "Raw points are plotted directly in the frequency domain without averaging. This preserves revisits to the same nominal frequency and any "
            "comb-like patterns introduced by repeated sweeps."
        ),
        "interpretation": (
            "Dense vertical bands indicate repeated visits to nearby frequencies with different outputs. That is useful evidence for sweep-to-sweep "
            "variation or hysteresis, even when the binned mean looks smooth."
        ),
    },
    "global_average": {
        "title": "Grand Average Curve",
        "purpose": (
            "The thesis used grand-average plots to condense overall behaviour, identify common high-performance bands, and find the frequency with the "
            "highest mean output."
        ),
        "method": (
            "After binning, all selected tests are aligned to a common frequency grid and averaged bin by bin. Standard-deviation error bars summarize "
            "between-test variability around that grand mean."
        ),
        "interpretation": (
            "This is the best plot for identifying broad operating regions rather than isolated single-test peaks. A high mean with wide error bars "
            "should be treated as promising but uncertain."
        ),
    },
    "summary_table": {
        "title": "Summary Statistics Table",
        "purpose": (
            "Use this to support the visual plots with compact quantitative descriptors, as in the thesis tables that accompanied distribution and peak analyses."
        ),
        "method": (
            "The app computes statistics such as mean, standard deviation, extrema, and coefficient of variation on the selected test or grouped data."
        ),
        "interpretation": (
            "The table is helpful for ranking tests, checking whether visual impressions are borne out numerically, and documenting the spread behind each plot."
        ),
    },
    "std_vs_mean": {
        "title": "Standard Deviation vs Mean",
        "purpose": (
            "This follows the thesis analysis of whether variability grows with performance. It is used to quantify the trade-off between high output and stability."
        ),
        "method": (
            "For each frequency bin, the mean output and its standard deviation are paired in a scatter plot, often alongside a fitted trend line."
        ),
        "interpretation": (
            "A strong positive trend means higher-output regions are also more variable. That can explain why peak-performing frequencies are not always the best "
            "operating choice."
        ),
    },
    "best_region": {
        "title": "Best Operating Region",
        "purpose": (
            "This is based on the thesis two-step operating-frequency method: first keep high-output bins, then choose the most stable among them."
        ),
        "method": (
            "The app first filters to the higher-mean bins, then searches within that subset for the lowest-variability bins. The result is shown as a "
            "mean-versus-standard-deviation cloud."
        ),
        "interpretation": (
            "A good operating region is high enough on the mean axis to be useful, but still low on the variability axis. This balances output and repeatability."
        ),
    },
    "correlation": {
        "title": "Correlation View",
        "purpose": (
            "The thesis used Pearson and Spearman correlation to quantify whether two performance metrics changed together. Here the same logic is applied to "
            "similarity between response curves."
        ),
        "method": (
            "Frequency-binned curves are aligned on a common frequency axis and then compared pairwise using correlation coefficients."
        ),
        "interpretation": (
            "High correlation means the compared curves rise and fall together across frequency, even if their absolute magnitudes differ."
        ),
    },
}


_GUIDE_SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("purpose", "What this plot is for", "What this plot is for:"),
    ("method", "Method", "Method:"),
    ("interpretation", "How to read it", "How to read it:"),
)


def get_plot_guidance(plot_id: str) -> dict[str, str] | None:
    """Return the guidance entry for one plot type."""
    return PLOT_GUIDANCE.get(plot_id)


def build_plot_guide_markdown(plot_id: str) -> str:
    """Format one guidance entry for Streamlit markdown rendering."""
    guide = get_plot_guidance(plot_id)
    if guide is None:
        return ""
    return "\n\n".join(
        f"**{heading}**\n\n{guide[field]}"
        for field, heading, _ in _GUIDE_SECTIONS
    )


def build_plot_guide_text(plot_id: str) -> str:
    """Format one guidance entry for plain-text report descriptions."""
    guide = get_plot_guidance(plot_id)
    if guide is None:
        return ""
    return "\n\n".join(
        f"{label} {guide[field]}"
        for field, _, label in _GUIDE_SECTIONS
    )


def render_plot_explanation(
    plot_id: str,
    *,
    label: str | None = None,
    expanded: bool = False,
    extra_markdown: str | None = None,
) -> None:
    """Render an inline explanation for one concrete plot instance."""
    guide = get_plot_guidance(plot_id)
    if guide is None:
        return

    expander_label = label or f"ℹ️ {guide['title']} — what it means"
    with st.expander(expander_label, expanded=expanded):
        st.markdown(build_plot_guide_markdown(plot_id))
        if extra_markdown and extra_markdown.strip():
            st.markdown(extra_markdown.strip())


def render_plot_guide(
    plot_ids: list[str],
    *,
    key_prefix: str,
    label_lookup: dict[str, str] | None = None,
) -> None:
    """Render a thesis-based explanation dropdown for the supplied plot IDs."""
    available_ids = [plot_id for plot_id in plot_ids if plot_id in PLOT_GUIDANCE]
    if not available_ids:
        return

    with st.expander("📘 Thesis-Based Plot Guide", expanded=False):
        chosen_plot = st.selectbox(
            "Plot type",
            available_ids,
            format_func=lambda plot_id: (
                label_lookup.get(plot_id, PLOT_GUIDANCE[plot_id]["title"])
                if label_lookup
                else PLOT_GUIDANCE[plot_id]["title"]
            ),
            key=f"{key_prefix}_plot_guide",
        )
        st.caption(
            "Based on the thesis methodology sections covering binning, plot families, and result interpretation."
        )
        st.markdown(build_plot_guide_markdown(chosen_plot))
