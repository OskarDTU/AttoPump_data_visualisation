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

if [ -x "./.venv/bin/streamlit" ]; then
  ./.venv/bin/streamlit run streamlit_app.py
elif [ -x "./Attopump_data_visualisation_1/.venv/bin/streamlit" ]; then
  ./Attopump_data_visualisation_1/.venv/bin/streamlit run streamlit_app.py
else
  echo "No Streamlit executable found in ./.venv or ./Attopump_data_visualisation_1/.venv" >&2
  exit 1
fi
