"""Persistent on-disk cache for processed test payloads and figure specs."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PERSISTENT_CACHE_DIR = _PROJECT_ROOT / ".cache" / "persistent_cache"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("_")
    return slug[:80] or "test"


def _freeze(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _freeze(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_freeze(item) for item in value]
    return value


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(_freeze(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_root() -> Path:
    return _ensure_dir(_PERSISTENT_CACHE_DIR)


def get_persistent_cache_root() -> Path:
    """Return the writable on-disk cache root used by the application."""
    return _cache_root()


def _test_cache_payload(
    *,
    run_name: str,
    data_root_str: str,
    csv_path_str: str,
    csv_signature: tuple[str, int, int],
    metadata_signature: tuple[str, int, int],
    user_patterns_signature: tuple[str, int, int],
    test_config_signature: tuple[str, int, int],
    experiment_log_signature: tuple[str, int, int],
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "data_root_str": data_root_str,
        "csv_path_str": csv_path_str,
        "csv_signature": csv_signature,
        "metadata_signature": metadata_signature,
        "user_patterns_signature": user_patterns_signature,
        "test_config_signature": test_config_signature,
        "experiment_log_signature": experiment_log_signature,
    }


def _test_cache_dir(**context: Any) -> Path:
    payload = _test_cache_payload(**context)
    digest = _stable_hash(payload)[:16]
    path = _cache_root() / "tests" / f"{_slugify(payload['run_name'])}_{digest}"
    _ensure_dir(path)
    meta_path = path / "context.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_pickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _save_pickle(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _figure_settings_hash(settings: Mapping[str, Any]) -> str:
    return _stable_hash({"settings": settings})[:16]


def load_cached_prepared_test_data(**context: Any) -> Any | None:
    """Return a persisted prepared-test payload, if available."""
    path = _test_cache_dir(**context) / "prepared.pkl.gz"
    return _load_pickle(path)


def save_cached_prepared_test_data(payload: Any, **context: Any) -> None:
    """Persist one prepared-test payload."""
    path = _test_cache_dir(**context) / "prepared.pkl.gz"
    _save_pickle(path, payload)


def load_cached_binned_test_data(
    *,
    bin_hz: float,
    **context: Any,
) -> Any | None:
    """Return a persisted binned-test payload, if available."""
    token = f"{float(bin_hz):.6f}".rstrip("0").rstrip(".").replace(".", "_")
    path = _test_cache_dir(**context) / "binned" / f"{token}.pkl.gz"
    return _load_pickle(path)


def save_cached_binned_test_data(
    payload: Any,
    *,
    bin_hz: float,
    **context: Any,
) -> None:
    """Persist one binned-test payload."""
    token = f"{float(bin_hz):.6f}".rstrip("0").rstrip(".").replace(".", "_")
    path = _test_cache_dir(**context) / "binned" / f"{token}.pkl.gz"
    _save_pickle(path, payload)


def _figure_paths(
    *,
    plot_kind: str,
    settings: Mapping[str, Any],
    **context: Any,
) -> tuple[Path, Path]:
    figure_dir = _ensure_dir(_test_cache_dir(**context) / "figures")
    stem = f"{_slugify(plot_kind)}__{_figure_settings_hash(settings)}"
    return figure_dir / f"{stem}.json.gz", figure_dir / f"{stem}.meta.json"


def load_cached_test_figure(
    *,
    plot_kind: str,
    settings: Mapping[str, Any],
    **context: Any,
) -> go.Figure | None:
    """Return a persisted Plotly figure for one test + settings combo."""
    figure_path, _ = _figure_paths(plot_kind=plot_kind, settings=settings, **context)
    if not figure_path.exists():
        return None
    try:
        with gzip.open(figure_path, "rt", encoding="utf-8") as handle:
            return pio.from_json(handle.read())
    except Exception:
        return None


def save_cached_test_figure(
    fig: go.Figure,
    *,
    plot_kind: str,
    settings: Mapping[str, Any],
    **context: Any,
) -> None:
    """Persist a Plotly figure for one test + settings combo."""
    figure_path, meta_path = _figure_paths(
        plot_kind=plot_kind,
        settings=settings,
        **context,
    )
    _ensure_dir(figure_path.parent)
    with gzip.open(figure_path, "wt", encoding="utf-8") as handle:
        handle.write(pio.to_json(fig, pretty=False, remove_uids=True))
    meta = {
        "plot_kind": plot_kind,
        "settings": _freeze(settings),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def get_or_create_cached_test_figure(
    *,
    cache_context: Mapping[str, Any] | None,
    plot_kind: str,
    settings: Mapping[str, Any],
    builder: Callable[[], go.Figure],
) -> tuple[go.Figure, bool]:
    """Load a cached figure or build and persist it on demand."""
    if cache_context:
        cached = load_cached_test_figure(
            plot_kind=plot_kind,
            settings=settings,
            **dict(cache_context),
        )
        if cached is not None:
            return cached, True

    fig = builder()
    if cache_context:
        try:
            save_cached_test_figure(
                fig,
                plot_kind=plot_kind,
                settings=settings,
                **dict(cache_context),
            )
        except Exception:
            pass
    return fig, False
