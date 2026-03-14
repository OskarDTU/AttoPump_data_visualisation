"""Persistent warm-up queue for report-specific per-test caches."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..plots import analysis_plots as ap
from ..plots.plot_generator import (
    downsample_sweep_points,
    plot_sweep_all_points,
    plot_sweep_binned,
    plot_time_series,
)
from .app_settings import clean_data_folder_path, load_settings
from .io_local import list_run_dirs, normalize_root
from .loader import (
    get_test_cache_context,
    load_binned_test_data,
    load_prepared_test_data,
)
from .persistent_cache import (
    get_or_create_cached_test_figure,
    get_persistent_cache_root,
)

REPORT_WARMUP_JOB_KEY = "report_builder"

_TASK_PENDING = "pending"
_TASK_COMPLETED = "completed"
_TASK_ERROR = "error"
_WORKER_IDLE = "idle"
_WORKER_STARTING = "starting"
_WORKER_RUNNING = "running"
_WORKER_PAUSED = "paused"
_WORKER_COMPLETED = "completed"
_WORKER_ERROR = "error"
_WORKER_STOPPED = "stopped"
_WORKER_STALE = "stale"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    return hashlib.sha256(
        json.dumps(
            _freeze(payload),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _jobs_dir() -> Path:
    path = get_persistent_cache_root() / "warmup_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _workers_dir() -> Path:
    path = get_persistent_cache_root() / "warmup_workers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_path(job_key: str) -> Path:
    return _jobs_dir() / f"{job_key}.json"


def _worker_state_path(job_key: str) -> Path:
    return _workers_dir() / f"{job_key}.worker.json"


def _worker_log_path(job_key: str) -> Path:
    return _workers_dir() / f"{job_key}.worker.log"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_warmup_job(job_key: str = REPORT_WARMUP_JOB_KEY) -> dict[str, Any] | None:
    """Load one persisted warm-up job from disk."""
    path = _job_path(job_key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_warmup_job(
    job: Mapping[str, Any],
    job_key: str = REPORT_WARMUP_JOB_KEY,
) -> None:
    """Persist one warm-up job definition and its progress."""
    path = _job_path(job_key)
    payload = dict(job)
    payload["updated_at"] = _now_iso()
    path.write_text(json.dumps(_freeze(payload), indent=2), encoding="utf-8")


def delete_warmup_job(job_key: str = REPORT_WARMUP_JOB_KEY) -> None:
    """Remove one persisted warm-up job."""
    path = _job_path(job_key)
    if path.exists():
        path.unlink()


def load_warmup_worker_state(
    job_key: str = REPORT_WARMUP_JOB_KEY,
) -> dict[str, Any] | None:
    """Load the persisted state for one detached warm-up worker."""
    path = _worker_state_path(job_key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_warmup_worker_state(
    state: Mapping[str, Any],
    job_key: str = REPORT_WARMUP_JOB_KEY,
) -> None:
    """Persist detached warm-up worker metadata and heartbeat information."""
    path = _worker_state_path(job_key)
    payload = dict(state)
    payload["job_key"] = job_key
    payload["updated_at"] = _now_iso()
    path.write_text(json.dumps(_freeze(payload), indent=2), encoding="utf-8")


def delete_warmup_worker_state(job_key: str = REPORT_WARMUP_JOB_KEY) -> None:
    """Remove one persisted detached-worker state record."""
    path = _worker_state_path(job_key)
    if path.exists():
        path.unlink()


def get_warmup_worker_log_path(job_key: str = REPORT_WARMUP_JOB_KEY) -> Path:
    """Return the log-file path for one detached warm-up worker."""
    return _worker_log_path(job_key)


def _pid_is_running(pid: int | None) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def summarize_warmup_worker_state(
    state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return normalized status information for one detached warm-up worker."""
    if not state:
        return {
            "running": False,
            "status": _WORKER_IDLE,
            "stale": False,
            "pid": None,
            "log_path": "",
            "started_at": "",
            "last_heartbeat": "",
            "finished_at": "",
            "error": "",
        }

    raw_pid = state.get("pid")
    try:
        pid = int(raw_pid) if raw_pid not in (None, "") else None
    except (TypeError, ValueError):
        pid = None
    running = _pid_is_running(pid)
    status = str(state.get("status") or _WORKER_IDLE)
    stale = bool(
        pid
        and not running
        and status in {_WORKER_STARTING, _WORKER_RUNNING}
    )
    if stale:
        status = _WORKER_STALE

    return {
        "running": running,
        "status": status,
        "stale": stale,
        "pid": pid,
        "log_path": str(state.get("log_path", "") or ""),
        "started_at": str(state.get("started_at", "") or ""),
        "last_heartbeat": str(state.get("last_heartbeat", "") or ""),
        "finished_at": str(state.get("finished_at", "") or ""),
        "error": str(state.get("error", "") or ""),
    }


def _resolve_warmup_job_data_folder(job: Mapping[str, Any]) -> str:
    stored_path = clean_data_folder_path(
        str(job.get("data_source", {}).get("data_folder_path", "") or "")
    )
    if stored_path:
        return stored_path
    return clean_data_folder_path(load_settings().data_folder_path)


def resolve_warmup_job_inventory(
    *,
    job: Mapping[str, Any] | None = None,
    job_key: str = REPORT_WARMUP_JOB_KEY,
) -> tuple[str, list[Path], list[str]]:
    """Resolve the data source and discovered run folders for one warm-up job."""
    resolved_job = job or load_warmup_job(job_key)
    if not resolved_job:
        raise ValueError("No warm-up job is available.")

    data_folder_str = _resolve_warmup_job_data_folder(resolved_job)
    if not data_folder_str:
        raise ValueError(
            "No data source path is saved for this warm-up job. Select a data source first."
        )

    root = normalize_root(data_folder_str)
    run_dirs = list(list_run_dirs(root))
    run_names = [path.name for path in run_dirs]
    return data_folder_str, run_dirs, run_names


def summarize_warmup_job(job: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return status counters for a warm-up job payload."""
    if not job:
        return {
            "total": 0,
            "pending": 0,
            "completed": 0,
            "errors": 0,
            "active": False,
        }

    tasks = list(job.get("tasks", []))
    pending = sum(1 for task in tasks if task.get("status") == _TASK_PENDING)
    completed = sum(1 for task in tasks if task.get("status") == _TASK_COMPLETED)
    errors = sum(1 for task in tasks if task.get("status") == _TASK_ERROR)
    return {
        "total": len(tasks),
        "pending": pending,
        "completed": completed,
        "errors": errors,
        "active": pending > 0 and not bool(job.get("paused", False)),
        "paused": bool(job.get("paused", False)),
        "complete": pending == 0,
    }


def _make_task(
    *,
    run_name: str,
    kind: str,
    label: str,
    settings: Mapping[str, Any] | None = None,
    plot_kind: str | None = None,
) -> dict[str, Any]:
    settings_dict = dict(settings or {})
    task_id = _stable_hash(
        {
            "run_name": run_name,
            "kind": kind,
            "plot_kind": plot_kind,
            "settings": settings_dict,
        }
    )[:20]
    return {
        "id": task_id,
        "run_name": run_name,
        "kind": kind,
        "label": label,
        "plot_kind": plot_kind,
        "settings": settings_dict,
        "status": _TASK_PENDING,
        "error": "",
    }


def _append_unique_task(job: dict[str, Any], task: Mapping[str, Any]) -> None:
    existing_ids = {item.get("id") for item in job.get("tasks", [])}
    if task.get("id") in existing_ids:
        return
    job.setdefault("tasks", []).append(dict(task))


def start_report_warmup_job(
    *,
    label: str,
    entry_ids: list[str],
    test_names: list[str],
    selection_mode: str,
    profile: Mapping[str, Any],
    data_folder_path: str = "",
    data_source_name: str = "",
    auto_resume: bool = True,
    job_key: str = REPORT_WARMUP_JOB_KEY,
) -> dict[str, Any]:
    """Create a persisted report-cache warm-up job for the selected tests."""
    unique_tests = list(dict.fromkeys(test_names))
    cleaned_data_folder = clean_data_folder_path(data_folder_path)
    job = {
        "job_key": job_key,
        "scope": "report_builder",
        "label": label,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "paused": False,
        "auto_resume": bool(auto_resume),
        "selection": {
            "entry_ids": list(entry_ids),
            "test_names": unique_tests,
            "selection_mode": selection_mode,
        },
        "data_source": {
            "data_folder_path": cleaned_data_folder,
            "selected_name": str(data_source_name).strip(),
        },
        "selection_signature": _stable_hash(
            {
                "entry_ids": entry_ids,
                "test_names": unique_tests,
                "selection_mode": selection_mode,
                "profile": dict(profile),
                "data_folder_path": cleaned_data_folder,
            }
        )[:20],
        "profile": dict(profile),
        "tasks": [],
    }
    for run_name in unique_tests:
        _append_unique_task(
            job,
            _make_task(
                run_name=run_name,
                kind="prepare",
                label=f"{run_name}: prepare reusable test data",
            ),
        )
    save_warmup_job(job, job_key=job_key)
    if not summarize_warmup_worker_state(load_warmup_worker_state(job_key)).get("running"):
        delete_warmup_worker_state(job_key)
    return job


def set_warmup_job_paused(
    paused: bool,
    job_key: str = REPORT_WARMUP_JOB_KEY,
) -> dict[str, Any] | None:
    """Pause or resume one persisted warm-up job."""
    job = load_warmup_job(job_key)
    if not job:
        return None
    job["paused"] = bool(paused)
    save_warmup_job(job, job_key=job_key)
    return job


def retry_failed_warmup_tasks(
    job_key: str = REPORT_WARMUP_JOB_KEY,
) -> dict[str, Any] | None:
    """Reset failed tasks back to pending for one persisted job."""
    job = load_warmup_job(job_key)
    if not job:
        return None
    for task in job.get("tasks", []):
        if task.get("status") == _TASK_ERROR:
            task["status"] = _TASK_PENDING
            task["error"] = ""
    save_warmup_job(job, job_key=job_key)
    return job


def launch_warmup_worker(
    job_key: str = REPORT_WARMUP_JOB_KEY,
) -> dict[str, Any]:
    """Launch the detached background worker that drains one warm-up job."""
    job = load_warmup_job(job_key)
    if not job:
        raise ValueError("Create a warm-up job before starting the background worker.")
    if summarize_warmup_job(job).get("pending", 0) <= 0:
        raise ValueError("This warm-up job has no pending tasks left.")

    current_state = load_warmup_worker_state(job_key)
    current_summary = summarize_warmup_worker_state(current_state)
    if current_summary.get("running"):
        return dict(current_state or {})

    if job.get("paused", False):
        raise ValueError("Resume the warm-up job before starting the background worker.")

    log_path = get_warmup_worker_log_path(job_key)
    command = [
        sys.executable,
        "-m",
        "app.data.cache_warmup_worker",
        "--job-key",
        job_key,
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log_handle:
        proc = subprocess.Popen(
            command,
            cwd=str(_project_root()),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    state = {
        "job_key": job_key,
        "pid": int(proc.pid),
        "status": _WORKER_STARTING,
        "started_at": _now_iso(),
        "last_heartbeat": _now_iso(),
        "finished_at": "",
        "error": "",
        "log_path": str(log_path),
        "command": command,
    }
    save_warmup_worker_state(state, job_key=job_key)
    return state


def _downsample_frame_evenly(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if df.empty or max_points <= 0 or len(df) <= int(max_points):
        return df
    idx = pd.Index(
        pd.Series(
            np.linspace(
                0,
                len(df) - 1,
                num=int(max_points),
                dtype=int,
            )
        ).unique()
    )
    return df.iloc[idx].copy()


def _build_time_series_figure(
    prepared,
    settings: Mapping[str, Any],
):
    plot_df = prepared.const_data if prepared.const_data is not None else prepared.time_series_data
    plot_df = _downsample_frame_evenly(plot_df, int(settings["max_raw_points"]))
    x_col = str(settings["x_col"])
    return plot_time_series(
        plot_df,
        x_col=x_col,
        y_col=prepared.signal_col,
        title="",
        mode=str(settings["mode"]),
        marker_size=int(settings["marker_size"]),
        opacity=float(settings["opacity"]),
    )


def _build_report_sweep_binned_figure(
    run_name: str,
    run_dir: str | Path,
    settings: Mapping[str, Any],
):
    binned = load_binned_test_data(
        run_name,
        run_dir,
        bin_hz=float(settings["bin_hz"]),
    )
    if binned is None:
        fig = go.Figure()
        fig.add_annotation(text="No sweep data available", showarrow=False)
        return fig
    return plot_sweep_binned(
        binned,
        title="",
        mode=str(settings["mode"]),
        marker_size=int(settings["marker_size"]),
        show_error_bars=bool(settings["show_error_bars"]),
    )


def _build_report_per_sweep_figure(prepared, settings: Mapping[str, Any]):
    sweep_df = prepared.sweep_data
    if sweep_df is None:
        return ap.plot_per_test_sweeps(pd.DataFrame(), signal_col=prepared.signal_col)
    return ap.plot_per_test_sweeps(
        sweep_df,
        signal_col=prepared.signal_col,
        bin_hz=float(settings["bin_hz"]),
        title="",
        mode=str(settings["mode"]),
        marker_size=int(settings["marker_size"]),
    )


def _build_report_all_sweeps_figure(prepared, run_name: str, run_dir: str | Path, settings: Mapping[str, Any]):
    sweep_df = prepared.sweep_data
    if sweep_df is None:
        fig = go.Figure()
        fig.add_annotation(text="No sweep data available", showarrow=False)
        return fig
    plot_df = downsample_sweep_points(
        sweep_df,
        max_points=int(settings["max_raw_points"]),
    )
    average_df = load_binned_test_data(
        run_name,
        run_dir,
        bin_hz=float(settings["bin_hz"]),
    )
    overlay = None
    if average_df is not None and not average_df.empty:
        overlay = average_df.rename(columns={"freq_center": "freq"})
        overlay = overlay[[col for col in ["freq", "mean", "std"] if col in overlay.columns]].copy()
    return plot_sweep_all_points(
        plot_df,
        x_col="Frequency",
        y_col=prepared.signal_col,
        color_col="Sweep" if "Sweep" in sweep_df.columns else None,
        title="",
        mode=str(settings["mode"]),
        marker_size=int(settings["marker_size"]),
        opacity=float(settings["opacity"]),
        average_df=overlay,
        show_average_error_bars=bool(settings["show_average_error_bars"]),
    )


def _warm_figure_task(
    *,
    run_name: str,
    run_dir: str | Path,
    plot_kind: str,
    settings: Mapping[str, Any],
) -> None:
    prepared = load_prepared_test_data(run_name, run_dir)
    cache_context = get_test_cache_context(run_name, run_dir)

    def _builder():
        if plot_kind == "time_series":
            return _build_time_series_figure(prepared, settings)
        if plot_kind == "sweep_binned":
            return _build_report_sweep_binned_figure(run_name, run_dir, settings)
        if plot_kind == "sweep_per_sweep_average":
            return _build_report_per_sweep_figure(prepared, settings)
        if plot_kind == "sweep_all_points":
            return _build_report_all_sweeps_figure(prepared, run_name, run_dir, settings)
        raise ValueError(f"Unsupported warm-up plot kind: {plot_kind}")

    get_or_create_cached_test_figure(
        cache_context=cache_context,
        plot_kind=plot_kind,
        settings=settings,
        builder=_builder,
    )


def _enqueue_report_follow_up_tasks(
    job: dict[str, Any],
    *,
    run_name: str,
    run_dir: str | Path,
) -> None:
    prepared = load_prepared_test_data(run_name, run_dir)
    report_profile = dict(job.get("profile", {}).get("report", {}))
    comparisons = set(report_profile.get("comparisons", []))
    plot_modes = dict(report_profile.get("plot_modes", {}))
    sweep_binned_comparisons = {
        "sweep_overlay",
        "sweep_relative",
        "individual_sweeps",
        "global_average",
        "summary_table",
        "std_vs_mean",
        "best_region",
        "correlation",
    }

    if prepared.sweep_data is not None:
        bin_hz = float(report_profile.get("bin_hz", 5.0))
        marker_size = int(report_profile.get("marker_size", 6))
        opacity = float(report_profile.get("opacity", 0.8))
        max_raw_points = int(report_profile.get("max_raw_points", 50_000))
        show_error_bars = bool(report_profile.get("show_error_bars", True))
        show_raw_all_sweeps = bool(report_profile.get("show_raw_all_sweeps", True))

        if comparisons & sweep_binned_comparisons:
            _append_unique_task(
                job,
                _make_task(
                    run_name=run_name,
                    kind="binned",
                    label=f"{run_name}: build binned sweep data",
                    settings={"bin_hz": bin_hz},
                ),
            )
        if "individual_sweeps" in comparisons:
            _append_unique_task(
                job,
                _make_task(
                    run_name=run_name,
                    kind="figure",
                    plot_kind="sweep_binned",
                    label=f"{run_name}: cache report binned sweep figure",
                    settings={
                        "bin_hz": bin_hz,
                        "mode": str(plot_modes.get("individual_binned", "lines+markers")),
                        "marker_size": marker_size,
                        "show_error_bars": show_error_bars,
                    },
                ),
            )
            _append_unique_task(
                job,
                _make_task(
                    run_name=run_name,
                    kind="figure",
                    plot_kind="sweep_per_sweep_average",
                    label=f"{run_name}: cache report per-sweep figure",
                    settings={
                        "bin_hz": bin_hz,
                        "mode": str(plot_modes.get("individual_per_sweep", "lines+markers")),
                        "marker_size": marker_size,
                    },
                ),
            )
            if show_raw_all_sweeps:
                _append_unique_task(
                    job,
                    _make_task(
                        run_name=run_name,
                        kind="figure",
                        plot_kind="sweep_all_points",
                        label=f"{run_name}: cache report raw all-sweeps figure",
                        settings={
                            "bin_hz": bin_hz,
                            "max_raw_points": max_raw_points,
                            "mode": str(
                                plot_modes.get(
                                    "individual_raw_all_sweeps",
                                    "lines+markers",
                                )
                            ),
                            "marker_size": marker_size,
                            "opacity": opacity,
                            "show_average_error_bars": show_error_bars,
                        },
                    ),
                )

    if prepared.const_data is not None and "constant_time_series" in comparisons:
        _append_unique_task(
            job,
            _make_task(
                run_name=run_name,
                kind="figure",
                plot_kind="time_series",
                label=f"{run_name}: cache report constant time-series figure",
                settings={
                    "max_raw_points": int(report_profile.get("max_raw_points", 50_000)),
                    "mode": str(
                        plot_modes.get("constant_time_series", "lines+markers")
                    ),
                    "marker_size": int(report_profile.get("marker_size", 6)),
                    "opacity": float(report_profile.get("opacity", 0.8)),
                    "x_col": str(prepared.const_data.columns[0]),
                },
            ),
        )


def _execute_task(
    job: dict[str, Any],
    task: dict[str, Any],
    run_map: Mapping[str, str | Path],
) -> None:
    run_name = str(task["run_name"])
    run_dir = run_map.get(run_name)
    if run_dir is None:
        raise ValueError("test folder is no longer available in the current data source")

    if task["kind"] == "prepare":
        _enqueue_report_follow_up_tasks(job, run_name=run_name, run_dir=run_dir)
        return
    if task["kind"] == "binned":
        load_binned_test_data(
            run_name,
            run_dir,
            bin_hz=float(task["settings"]["bin_hz"]),
        )
        return
    if task["kind"] == "figure":
        _warm_figure_task(
            run_name=run_name,
            run_dir=run_dir,
            plot_kind=str(task["plot_kind"]),
            settings=dict(task["settings"]),
        )
        return
    raise ValueError(f"Unsupported warm-up task kind: {task['kind']}")


def run_report_warmup_job(
    run_names: list[str] | None = None,
    run_dirs: list[Path] | None = None,
    *,
    max_tasks: int = 1,
    time_budget_s: float = 1.5,
    job_key: str = REPORT_WARMUP_JOB_KEY,
) -> dict[str, Any] | None:
    """Advance a persisted report warm-up job by a small amount of work."""
    job = load_warmup_job(job_key)
    if not job or job.get("paused", False):
        return job

    if run_names is None or run_dirs is None:
        _, resolved_run_dirs, resolved_run_names = resolve_warmup_job_inventory(
            job=job,
            job_key=job_key,
        )
        run_names = resolved_run_names
        run_dirs = resolved_run_dirs

    run_map = {name: path for name, path in zip(run_names, run_dirs)}
    completed_this_run = 0
    started = time.monotonic()

    while completed_this_run < int(max_tasks):
        if time.monotonic() - started >= float(time_budget_s):
            break
        pending_task = next(
            (
                task
                for task in job.get("tasks", [])
                if task.get("status") == _TASK_PENDING
            ),
            None,
        )
        if pending_task is None:
            break
        try:
            _execute_task(job, pending_task, run_map)
            pending_task["status"] = _TASK_COMPLETED
            pending_task["error"] = ""
            pending_task["completed_at"] = _now_iso()
        except Exception as exc:
            pending_task["status"] = _TASK_ERROR
            pending_task["error"] = str(exc)
            pending_task["completed_at"] = _now_iso()
        save_warmup_job(job, job_key=job_key)
        completed_this_run += 1

    return job
