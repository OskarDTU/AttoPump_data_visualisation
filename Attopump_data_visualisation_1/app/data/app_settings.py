"""Persistent application settings (data path, preferences).

Stores user preferences in ``app_settings.json`` next to this file.
The settings survive between Streamlit sessions and across machines
(the file can be committed or added to ``.gitignore`` as desired).

Public API
----------
- ``load_settings()`` → full ``AppSettings`` dataclass.
- ``save_settings(...)`` → persist changes to disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

_SETTINGS_PATH = Path(__file__).parent / "app_settings.json"


@dataclass
class AppSettings:
    """Application-wide user preferences.

    Attributes
    ----------
    data_folder_path : str
        Last-used root folder path for test data.
    """

    data_folder_path: str = ""


def load_settings() -> AppSettings:
    """Load settings from disk, returning defaults if the file is missing."""
    if _SETTINGS_PATH.exists():
        try:
            with open(_SETTINGS_PATH, "r") as f:
                raw = json.load(f)
            return AppSettings(
                data_folder_path=raw.get("data_folder_path", ""),
            )
        except Exception:
            return AppSettings()
    return AppSettings()


def save_settings(settings: AppSettings) -> None:
    """Persist the settings to disk."""
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(asdict(settings), f, indent=2)
