"""Streamlit multipage entry point for AttoPump data visualization.

Run locally:
    uv run python -m streamlit run streamlit_app.py

Pages:
    📊 Single Test Explorer  – original quick-look page (one test at a time)
    🔬 Comprehensive Analysis – multi-test comparison, EDA, best-region finder
"""

from __future__ import annotations

import sys
from pathlib import Path

# ------------------------------------------------------------------
# Ensure the package and app directories are importable.
# ------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st


# ------------------------------------------------------------------
# Page wrappers — import at call time so relative imports work.
# ------------------------------------------------------------------

def _run_explorer():
    from app.app import main
    main()


def _run_analysis():
    from app.analysis import main
    main()


# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------

st.set_page_config(page_title="AttoPump Data Visualization", layout="wide")

pg = st.navigation(
    [
        st.Page(_run_explorer, title="Single Test Explorer", icon="📊", default=True),
        st.Page(_run_analysis, title="Comprehensive Analysis", icon="🔬"),
    ]
)
pg.run()
