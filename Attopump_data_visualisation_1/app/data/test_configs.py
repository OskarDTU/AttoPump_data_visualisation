"""Persistent test configuration store.

Provides a JSON-backed store that lets users define the test type and
parameters for each test folder **once** and have them remembered across
sessions and across different machines (the JSON file is committed
alongside the code).

Two kinds of configuration are supported:

1. **Constant-frequency** — the user provides the frequency in Hz.
2. **Frequency sweep** — the user provides start Hz, end Hz, and sweep
   duration in seconds.

The configuration is stored in ``test_configs.json`` next to this file.

Public API
----------
- ``load_test_configs()``   → full dict of configs.
- ``save_test_config(...)`` → upsert a single test's config.
- ``delete_test_config(...)`` → remove a test's config.
- ``get_test_config(...)``  → retrieve a single entry (or ``None``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

_CONFIG_PATH = Path(__file__).parent / "test_configs.json"


# ── Dataclass ──────────────────────────────────────────────────────────────

@dataclass
class TestConfig:
    """User-defined test configuration for a single folder.

    Attributes
    ----------
    test_type : str
        ``"constant"`` or ``"sweep"``.
    frequency_hz : float or None
        For constant-frequency tests only.
    start_hz : float or None
        For sweep tests only: start of sweep range.
    end_hz : float or None
        For sweep tests only: end of sweep range.
    duration_s : float or None
        For sweep tests only: duration of one complete sweep cycle.
    note : str
        Optional free-text note.
    """

    test_type: Literal["constant", "sweep"]
    frequency_hz: float | None = None
    start_hz: float | None = None
    end_hz: float | None = None
    duration_s: float | None = None
    note: str = ""


# ── Load / save helpers ───────────────────────────────────────────────────

def _read_raw() -> dict:
    """Read the JSON file, returning the full dict."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"_description": "User-defined test configurations.", "tests": {}}


def _write_raw(data: dict) -> None:
    """Atomically write the config dict to disk."""
    with open(_CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_test_configs() -> dict[str, TestConfig]:
    """Load all saved test configurations.

    Returns
    -------
    dict[str, TestConfig]
        Mapping from folder name → ``TestConfig``.
    """
    raw = _read_raw()
    configs: dict[str, TestConfig] = {}
    for folder, entry in raw.get("tests", {}).items():
        try:
            configs[folder] = TestConfig(
                test_type=entry.get("test_type", "constant"),
                frequency_hz=entry.get("frequency_hz"),
                start_hz=entry.get("start_hz"),
                end_hz=entry.get("end_hz"),
                duration_s=entry.get("duration_s"),
                note=entry.get("note", ""),
            )
        except Exception:
            continue  # skip malformed entries silently
    return configs


def get_test_config(folder_name: str) -> TestConfig | None:
    """Retrieve the config for a single folder, or ``None``."""
    return load_test_configs().get(folder_name)


def save_test_config(folder_name: str, config: TestConfig) -> None:
    """Upsert a test configuration for *folder_name*."""
    raw = _read_raw()
    raw.setdefault("tests", {})[folder_name] = asdict(config)
    _write_raw(raw)


def delete_test_config(folder_name: str) -> bool:
    """Remove the config for *folder_name*.  Returns True if it existed."""
    raw = _read_raw()
    removed = raw.get("tests", {}).pop(folder_name, None) is not None
    if removed:
        _write_raw(raw)
    return removed
