"""Persistent application settings (data path, preferences).

Stores user preferences in ``~/.attopump_data_visualisation/app_settings.json``.
Legacy settings from ``app/data/app_settings.json`` are migrated
automatically the first time they are loaded.

Public API
----------
- ``load_settings()`` → full ``AppSettings`` dataclass.
- ``save_settings(...)`` → persist changes to disk.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path

_SETTINGS_DIR = Path.home() / ".attopump_data_visualisation"
_SETTINGS_PATH = _SETTINGS_DIR / "app_settings.json"
_LEGACY_SETTINGS_PATH = Path(__file__).parent / "app_settings.json"


@dataclass
class AppSettings:
    """Application-wide user preferences.

    Attributes
    ----------
    data_folder_path : str
        Last-used root folder path for test data.
    saved_data_paths : dict[str, str]
        Named data-source paths for the current user.
    selected_data_path_name : str
        Name of the last selected saved data source, if any.
    """

    data_folder_path: str = ""
    saved_data_paths: dict[str, str] = field(default_factory=dict)
    selected_data_path_name: str = ""


def clean_data_folder_path(raw_path: str) -> str:
    """Normalize pasted folder paths without being overly aggressive.

    Handles:
    - surrounding single or double quotes
    - shell-escaped spaces from macOS terminal drag-and-drop
    """
    cleaned = str(raw_path).strip()
    if not cleaned:
        return ""

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()

    if "\\ " in cleaned:
        try:
            parts = shlex.split(cleaned)
        except ValueError:
            parts = []
        if len(parts) == 1:
            cleaned = parts[0]

    return cleaned


def clean_saved_data_paths(raw_paths: dict[str, str] | None) -> dict[str, str]:
    """Normalize saved-path records and drop empty entries."""
    cleaned: dict[str, str] = {}
    for raw_name, raw_path in (raw_paths or {}).items():
        name = str(raw_name).strip()
        path = clean_data_folder_path(raw_path)
        if name and path:
            cleaned[name] = path
    return cleaned


def load_settings() -> AppSettings:
    """Load settings from disk, returning defaults if no file exists."""
    for path in (_SETTINGS_PATH, _LEGACY_SETTINGS_PATH):
        if not path.exists():
            continue
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            raw_data_path = raw.get("data_folder_path", "")
            raw_saved_paths = raw.get("saved_data_paths", {})
            raw_selected_name = str(raw.get("selected_data_path_name", "")).strip()
            saved_data_paths = clean_saved_data_paths(raw_saved_paths)
            selected_name = raw_selected_name if raw_selected_name in saved_data_paths else ""
            settings = AppSettings(
                data_folder_path=clean_data_folder_path(raw_data_path),
                saved_data_paths=saved_data_paths,
                selected_data_path_name=selected_name,
            )
            if selected_name:
                settings.data_folder_path = saved_data_paths[selected_name]
            if path == _LEGACY_SETTINGS_PATH and not _SETTINGS_PATH.exists():
                save_settings(settings)
            elif path == _SETTINGS_PATH and (
                settings.data_folder_path != raw_data_path
                or settings.saved_data_paths != raw_saved_paths
                or settings.selected_data_path_name != raw_selected_name
            ):
                save_settings(settings)
            return settings
        except Exception:
            continue
    return AppSettings()


def save_settings(settings: AppSettings) -> None:
    """Persist the settings to disk."""
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_saved_paths = clean_saved_data_paths(settings.saved_data_paths)
    selected_name = settings.selected_data_path_name.strip()
    if selected_name not in cleaned_saved_paths:
        selected_name = ""
    current_path = clean_data_folder_path(settings.data_folder_path)
    if selected_name:
        current_path = cleaned_saved_paths[selected_name]

    payload = asdict(
        AppSettings(
            data_folder_path=current_path,
            saved_data_paths=cleaned_saved_paths,
            selected_data_path_name=selected_name,
        )
    )
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
