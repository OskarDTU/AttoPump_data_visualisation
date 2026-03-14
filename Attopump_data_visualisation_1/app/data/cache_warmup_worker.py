"""Detached worker that drains a persisted warm-up job to completion."""

from __future__ import annotations

import argparse
import os
import traceback
from datetime import datetime, timezone
from typing import Any

from .cache_warmup import (
    REPORT_WARMUP_JOB_KEY,
    load_warmup_job,
    load_warmup_worker_state,
    resolve_warmup_job_inventory,
    run_report_warmup_job,
    save_warmup_worker_state,
    summarize_warmup_job,
)

_WORKER_RUNNING = "running"
_WORKER_PAUSED = "paused"
_WORKER_COMPLETED = "completed"
_WORKER_ERROR = "error"
_WORKER_STOPPED = "stopped"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _update_worker_state(
    job_key: str,
    *,
    status: str,
    error: str = "",
    finished: bool = False,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = dict(load_warmup_worker_state(job_key) or {})
    state["job_key"] = job_key
    state["pid"] = None if finished else int(os.getpid())
    state["status"] = status
    state["error"] = str(error or "")
    state["last_heartbeat"] = _now_iso()
    state.setdefault("started_at", _now_iso())
    if progress:
        state["progress"] = dict(progress)
    if finished:
        state["finished_at"] = _now_iso()
    else:
        state["finished_at"] = ""
    save_warmup_worker_state(state, job_key=job_key)
    return state


def run_background_warmup_worker(
    *,
    job_key: str = REPORT_WARMUP_JOB_KEY,
) -> int:
    """Drain one persisted warm-up job until complete, paused, or cleared."""
    job = load_warmup_job(job_key)
    if not job:
        _update_worker_state(
            job_key,
            status=_WORKER_STOPPED,
            error="No warm-up job was found.",
            finished=True,
        )
        return 0

    _update_worker_state(job_key, status=_WORKER_RUNNING)
    print(f"[{_now_iso()}] Started warm-up worker for job '{job_key}'.", flush=True)

    while True:
        job = load_warmup_job(job_key)
        if not job:
            _update_worker_state(job_key, status=_WORKER_STOPPED, finished=True)
            print(f"[{_now_iso()}] Warm-up job was cleared.", flush=True)
            return 0

        summary = summarize_warmup_job(job)
        _update_worker_state(job_key, status=_WORKER_RUNNING, progress=summary)

        if summary.get("paused", False):
            _update_worker_state(
                job_key,
                status=_WORKER_PAUSED,
                finished=True,
                progress=summary,
            )
            print(f"[{_now_iso()}] Warm-up job is paused.", flush=True)
            return 0

        if summary.get("pending", 0) <= 0:
            _update_worker_state(
                job_key,
                status=_WORKER_COMPLETED,
                finished=True,
                progress=summary,
            )
            print(
                f"[{_now_iso()}] Warm-up job completed "
                f"({summary.get('completed', 0)} tasks, {summary.get('errors', 0)} errors).",
                flush=True,
            )
            return 0

        data_folder_str, run_dirs, run_names = resolve_warmup_job_inventory(
            job=job,
            job_key=job_key,
        )
        print(
            f"[{_now_iso()}] Processing next task from '{data_folder_str}' "
            f"({summary.get('completed', 0)}/{summary.get('total', 0)} done).",
            flush=True,
        )
        updated_job = run_report_warmup_job(
            run_names,
            run_dirs,
            max_tasks=1,
            time_budget_s=86_400.0,
            job_key=job_key,
        )
        updated_summary = summarize_warmup_job(updated_job)
        _update_worker_state(
            job_key,
            status=_WORKER_RUNNING,
            progress=updated_summary,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drain one saved report warm-up job in the background.",
    )
    parser.add_argument(
        "--job-key",
        default=REPORT_WARMUP_JOB_KEY,
        help="Persisted warm-up job key to process.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point for the detached warm-up worker."""
    args = _parse_args()
    try:
        return run_background_warmup_worker(job_key=str(args.job_key))
    except Exception as exc:
        _update_worker_state(
            str(args.job_key),
            status=_WORKER_ERROR,
            error=str(exc),
            finished=True,
        )
        print(traceback.format_exc(), flush=True)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
