"""Streamlit entry point for AttoPump data visualization.

This wrapper properly configures the Python path and runs the app.
Run with: uv run streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

# Add the project directory to sys.path so Attopump_data_visualisation_1 can be imported
project_dir = Path(__file__).parent / "Attopump_data_visualisation_1"
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# Now the app can import from Attopump_data_visualisation_1
from app.app import *  # noqa: F401, F403
