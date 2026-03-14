"""Regression tests for persistent plot/data cache helpers."""

from __future__ import annotations

import sys
import types


def _identity_decorator(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def _wrap(func):
        return func

    return _wrap


class _FakeStreamlit(types.ModuleType):
    session_state: dict = {}

    def __getattr__(self, name: str):
        if name in {"cache_data", "dialog"}:
            return _identity_decorator
        if name == "session_state":
            return self.session_state
        return lambda *args, **kwargs: None


sys.modules.setdefault("streamlit", _FakeStreamlit("streamlit"))

import pandas as pd
import plotly.graph_objects as go

from app.data import cache_warmup, cache_warmup_worker, loader, persistent_cache


def _test_context() -> dict[str, object]:
    return {
        "run_name": "Run A",
        "data_root_str": "/tmp/data",
        "csv_path_str": "/tmp/data/Run A/merged.csv",
        "csv_signature": ("/tmp/data/Run A/merged.csv", 1, 100),
        "metadata_signature": ("meta.json", 1, 10),
        "user_patterns_signature": ("patterns.json", 1, 10),
        "test_config_signature": ("config.json", 1, 10),
        "experiment_log_signature": ("experiment_log.json", 1, 10),
    }


def test_cached_test_figure_is_reused_without_rebuilding(
    monkeypatch,
    tmp_path,
) -> None:
    """A persisted figure should be loaded from disk on the second request."""
    monkeypatch.setattr(
        persistent_cache,
        "_PERSISTENT_CACHE_DIR",
        tmp_path / ".cache" / "persistent_cache",
    )

    build_calls = {"count": 0}

    def _builder() -> go.Figure:
        build_calls["count"] += 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], mode="lines"))
        return fig

    fig1, from_cache_1 = persistent_cache.get_or_create_cached_test_figure(
        cache_context=_test_context(),
        plot_kind="demo_plot",
        settings={"mode": "lines"},
        builder=_builder,
    )
    fig2, from_cache_2 = persistent_cache.get_or_create_cached_test_figure(
        cache_context=_test_context(),
        plot_kind="demo_plot",
        settings={"mode": "lines"},
        builder=_builder,
    )

    assert build_calls["count"] == 1
    assert from_cache_1 is False
    assert from_cache_2 is True
    assert len(fig1.data) == 1
    assert len(fig2.data) == 1


def test_get_test_cache_context_excludes_streamlit_only_run_dir(monkeypatch) -> None:
    """Figure-cache contexts should omit the decorator-only `run_dir_str` key."""
    monkeypatch.setattr(
        loader,
        "_build_test_cache_context",
        lambda run_name, run_dir: {
            "run_name": run_name,
            "run_dir_str": str(run_dir),
            "csv_path_str": "/tmp/test.csv",
        },
    )

    context = loader.get_test_cache_context("Run A", "/tmp/Run A")

    assert context == {
        "run_name": "Run A",
        "csv_path_str": "/tmp/test.csv",
    }


def test_report_warmup_job_expands_prepare_tasks_and_finishes(
    monkeypatch,
    tmp_path,
) -> None:
    """The warm-up queue should persist progress and add follow-up tasks once prepared."""
    monkeypatch.setattr(
        persistent_cache,
        "_PERSISTENT_CACHE_DIR",
        tmp_path / ".cache" / "persistent_cache",
    )

    fake_prepared = types.SimpleNamespace(
        time_series_data=pd.DataFrame({"time_s": [0.0, 1.0], "flow": [1.0, 1.1]}),
        signal_col="flow",
        test_type="sweep",
        sweep_data=pd.DataFrame(
            {
                "Sweep": [0, 0, 1, 1],
                "Frequency": [100.0, 110.0, 100.0, 110.0],
                "flow": [1.0, 1.1, 0.9, 1.05],
            }
        ),
        const_data=None,
        const_freq=None,
    )
    fake_binned = pd.DataFrame(
        {"freq_center": [105.0], "mean": [1.0125], "std": [0.075]}
    )

    monkeypatch.setattr(cache_warmup, "load_prepared_test_data", lambda *args, **kwargs: fake_prepared)
    monkeypatch.setattr(cache_warmup, "load_binned_test_data", lambda *args, **kwargs: fake_binned)
    monkeypatch.setattr(cache_warmup, "get_test_cache_context", lambda *args, **kwargs: _test_context())
    monkeypatch.setattr(cache_warmup, "normalize_root", lambda path: tmp_path)
    monkeypatch.setattr(cache_warmup, "list_run_dirs", lambda root: [tmp_path / "Run A"])
    monkeypatch.setattr(
        cache_warmup,
        "get_or_create_cached_test_figure",
        lambda **kwargs: (kwargs["builder"](), False),
    )

    cache_warmup.start_report_warmup_job(
        label="Warm selected report tests",
        entry_ids=["Pump A"],
        test_names=["Run A"],
        selection_mode="pumps",
        profile={
            "report": {
                "comparisons": ["individual_sweeps"],
                "bin_hz": 5.0,
                "max_raw_points": 5000,
                "marker_size": 6,
                "opacity": 0.8,
                "show_error_bars": True,
                "show_raw_all_sweeps": True,
                "plot_modes": {
                    "individual_binned": "lines+markers",
                    "individual_per_sweep": "lines+markers",
                    "individual_raw_all_sweeps": "markers",
                    "constant_time_series": "lines",
                },
            }
        },
        data_folder_path=str(tmp_path),
    )

    job_after_prepare = cache_warmup.run_report_warmup_job(max_tasks=1, time_budget_s=5.0)
    summary_after_prepare = cache_warmup.summarize_warmup_job(job_after_prepare)

    assert summary_after_prepare["completed"] == 1
    assert summary_after_prepare["pending"] > 0
    assert summary_after_prepare["total"] >= 4

    finished_job = cache_warmup.run_report_warmup_job(max_tasks=10, time_budget_s=5.0)
    finished_summary = cache_warmup.summarize_warmup_job(finished_job)

    assert finished_summary["pending"] == 0
    assert finished_summary["errors"] == 0


def test_summarize_warmup_worker_state_detects_stale_process(monkeypatch) -> None:
    """Detached-worker summaries should flag stale PID records cleanly."""
    monkeypatch.setattr(cache_warmup, "_pid_is_running", lambda pid: False)

    summary = cache_warmup.summarize_warmup_worker_state(
        {
            "pid": 4321,
            "status": "running",
            "log_path": "/tmp/warmup.log",
        }
    )

    assert summary["running"] is False
    assert summary["stale"] is True
    assert summary["status"] == "stale"


def test_launch_warmup_worker_records_process_state(
    monkeypatch,
    tmp_path,
) -> None:
    """Launching the overnight worker should persist detached process metadata."""
    monkeypatch.setattr(
        persistent_cache,
        "_PERSISTENT_CACHE_DIR",
        tmp_path / ".cache" / "persistent_cache",
    )

    cache_warmup.start_report_warmup_job(
        label="Warm selected report tests",
        entry_ids=["Pump A"],
        test_names=["Run A"],
        selection_mode="pumps",
        profile={"report": {"comparisons": []}},
        data_folder_path=str(tmp_path),
    )

    launch_calls: dict[str, object] = {}

    class _FakeProcess:
        pid = 9876

    def _fake_popen(command, **kwargs):
        launch_calls["command"] = list(command)
        launch_calls["kwargs"] = dict(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(cache_warmup.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(cache_warmup, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(cache_warmup.sys, "executable", "/tmp/fake-python")

    state = cache_warmup.launch_warmup_worker()

    assert state["pid"] == 9876
    assert state["status"] == "starting"
    assert launch_calls["command"] == [
        "/tmp/fake-python",
        "-m",
        "app.data.cache_warmup_worker",
        "--job-key",
        cache_warmup.REPORT_WARMUP_JOB_KEY,
    ]
    assert launch_calls["kwargs"]["cwd"] == str(tmp_path)
    assert launch_calls["kwargs"]["start_new_session"] is True


def test_background_worker_marks_completed_job_state(monkeypatch) -> None:
    """The detached worker should mark already-finished jobs as completed."""
    saved_states: list[dict[str, object]] = []
    monkeypatch.setattr(
        cache_warmup_worker,
        "load_warmup_job",
        lambda job_key: {"paused": False, "tasks": [{"status": "completed"}]},
    )
    monkeypatch.setattr(
        cache_warmup_worker,
        "summarize_warmup_job",
        lambda job: {
            "total": 1,
            "pending": 0,
            "completed": 1,
            "errors": 0,
            "paused": False,
            "complete": True,
        },
    )
    monkeypatch.setattr(
        cache_warmup_worker,
        "load_warmup_worker_state",
        lambda job_key: None,
    )
    monkeypatch.setattr(
        cache_warmup_worker,
        "save_warmup_worker_state",
        lambda state, job_key=None: saved_states.append(dict(state)),
    )

    exit_code = cache_warmup_worker.run_background_warmup_worker(job_key="demo")

    assert exit_code == 0
    assert saved_states[-1]["status"] == "completed"
    assert saved_states[-1]["pid"] is None
