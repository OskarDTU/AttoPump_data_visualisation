"""Persistent application settings (data path, preferences).

Stores per-user preferences in ``~/.attopump_data_visualisation/app_settings.json``
and shared named data-source paths in ``app/data/shared_app_settings.json`` so
other repository users can see them too. Legacy settings from
``app/data/app_settings.json`` are migrated automatically the first time they
are loaded.

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
_SHARED_SETTINGS_PATH = Path(__file__).parent / "shared_app_settings.json"
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


def shared_settings_storage_signature() -> tuple[int, int]:
    """Return a lightweight signature of the shared saved-path store."""
    try:
        stat = _SHARED_SETTINGS_PATH.stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (0, 0)


def _read_json_file(path: Path) -> dict:
    with open(path, "r") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def _load_shared_saved_paths() -> dict[str, str]:
    if not _SHARED_SETTINGS_PATH.exists():
        return {}
    try:
        raw = _read_json_file(_SHARED_SETTINGS_PATH)
    except Exception:
        return {}
    return clean_saved_data_paths(raw.get("saved_data_paths", {}))


def _save_shared_saved_paths(saved_data_paths: dict[str, str]) -> None:
    _SHARED_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_description": "Shared saved data-source paths for repository users.",
        "saved_data_paths": clean_saved_data_paths(saved_data_paths),
    }
    with open(_SHARED_SETTINGS_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def load_settings() -> AppSettings:
    """Load settings from disk, returning defaults if no file exists."""
    shared_saved_paths = _load_shared_saved_paths()

    for path in (_SETTINGS_PATH, _LEGACY_SETTINGS_PATH):
        if not path.exists():
            continue
        try:
            raw = _read_json_file(path)
            raw_data_path = raw.get("data_folder_path", "")
            raw_selected_name = str(raw.get("selected_data_path_name", "")).strip()
            saved_data_paths = (
                shared_saved_paths
                if shared_saved_paths
                else clean_saved_data_paths(raw.get("saved_data_paths", {}))
            )
            selected_name = raw_selected_name if raw_selected_name in saved_data_paths else ""
            settings = AppSettings(
                data_folder_path=clean_data_folder_path(raw_data_path),
                saved_data_paths=saved_data_paths,
                selected_data_path_name=selected_name,
            )
            legacy_saved_paths = clean_saved_data_paths(raw.get("saved_data_paths", {}))
            if legacy_saved_paths and legacy_saved_paths != shared_saved_paths:
                shared_saved_paths = legacy_saved_paths
                _save_shared_saved_paths(shared_saved_paths)
                settings.saved_data_paths = shared_saved_paths
                if settings.selected_data_path_name and settings.selected_data_path_name not in shared_saved_paths:
                    settings.selected_data_path_name = ""
            if selected_name:
                settings.data_folder_path = saved_data_paths[selected_name]
            if path == _LEGACY_SETTINGS_PATH and not _SETTINGS_PATH.exists():
                save_settings(settings)
            elif path == _SETTINGS_PATH and (
                settings.data_folder_path != raw_data_path
                or settings.selected_data_path_name != raw_selected_name
            ):
                save_settings(settings)
            return settings
        except Exception:
            continue

    return AppSettings(saved_data_paths=shared_saved_paths)


def save_settings(settings: AppSettings) -> None:
    """Persist the settings to disk."""
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_saved_paths = clean_saved_data_paths(settings.saved_data_paths)
    _save_shared_saved_paths(cleaned_saved_paths)
    selected_name = settings.selected_data_path_name.strip()
    if selected_name not in cleaned_saved_paths:
        selected_name = ""
    current_path = clean_data_folder_path(settings.data_folder_path)
    if selected_name:
        current_path = cleaned_saved_paths[selected_name]

    payload = asdict(
        AppSettings(
            data_folder_path=current_path,
            saved_data_paths={},
            selected_data_path_name=selected_name,
        )
    )
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
