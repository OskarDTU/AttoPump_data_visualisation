#!/bin/bash
# ────────────────────────────────────────────────────────────────────
# run_app.sh — Launch the AttoPump Data Visualization Streamlit app.
#
# This script activates the project's Python virtual environment and
# starts the Streamlit server.  Run it from the repository root:
#
#     ./run_app.sh
#
# The app will open in the default browser at http://localhost:8501.
# ────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"
./Attopump_data_visualisation_1/.venv/bin/streamlit run streamlit_app.py
