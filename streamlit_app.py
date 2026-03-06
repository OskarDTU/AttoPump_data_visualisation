"""Streamlit multipage entry point for AttoPump data visualization.

Run with:
    ./run_app.sh          (from repo root)
    uv run streamlit run streamlit_app.py   (manual)

Pages:
    🗂️ Test Overview           – resolved classifications and missing info
    📊 Single Test Explorer    – one test at a time
    🔬 Comprehensive Analysis  – multi-test / multi-pump comparison & EDA
    🛠️ Manage Groups           – CRUD for pumps, shipments, test groups
    📑 Report Builder          – compose & export report packages
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


def _run_test_overview():
    from app.pages.test_overview import main
    main()


def _run_analysis():
    from app.pages.analysis import main
    main()


def _run_report_builder():
    from app.pages.report_builder import main
    main()


def _run_manage_groups():
    from app.pages.manage_groups import main
    main()


# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------

st.set_page_config(page_title="AttoPump Data Visualization", layout="wide")

pg = st.navigation(
    [
        st.Page(_run_explorer, title="Single Test Explorer", icon="📊", default=True),
        st.Page(_run_test_overview, title="Test Overview", icon="🗂️"),
        st.Page(_run_analysis, title="Comprehensive Analysis", icon="🔬"),
        st.Page(_run_manage_groups, title="Manage Groups", icon="🛠️"),
        st.Page(_run_report_builder, title="Report Builder", icon="📑"),
    ]
)
pg.run()
