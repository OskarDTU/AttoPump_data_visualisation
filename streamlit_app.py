"""Streamlit multipage entry point for AttoPump data visualization.

Run with:
    ./run_app.sh          (from repo root)
    uv run streamlit run streamlit_app.py   (manual)

Pages:
    📊 Single Test Explorer    – one test at a time
    🔬 Comprehensive Analysis  – multi-test comparison & EDA
    📦 Bar Comparison          – compare groups of pumps (bars & shipments)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ------------------------------------------------------------------
# Ensure both the project dir and its src/ are importable.
# ------------------------------------------------------------------
_PROJECT = Path(__file__).resolve().parent / "Attopump_data_visualisation_1"
for _p in [str(_PROJECT / "src"), str(_PROJECT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st


# ------------------------------------------------------------------
# Page wrappers — import at call time so relative imports work.
# ------------------------------------------------------------------

def _run_explorer():
    from app.pages.explorer import main
    main()


def _run_analysis():
    from app.pages.analysis import main
    main()


def _run_bar_comparison():
    from app.pages.bar_comparison import main
    main()


# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------

st.set_page_config(page_title="AttoPump Data Visualization", layout="wide")

pg = st.navigation(
    [
        st.Page(_run_explorer, title="Single Test Explorer", icon="📊", default=True),
        st.Page(_run_analysis, title="Comprehensive Analysis", icon="🔬"),
        st.Page(_run_bar_comparison, title="Bar Comparison", icon="📦"),
    ]
)
pg.run()
