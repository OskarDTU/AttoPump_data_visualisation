"""Regression tests for local CSV selection and cloud placeholders."""

from __future__ import annotations

import pytest

from app.data import io_local


def _write_csv(path, content: str = "col_a,col_b\n1,2\n") -> None:
    path.write_text(content, encoding="utf-8")


def test_pick_best_csv_prefers_local_flow_csv_over_auxiliary_events(
    monkeypatch,
    tmp_path,
) -> None:
    """A cloud-only merged.csv should not force the picker onto events.csv."""
    run_dir = tmp_path / "Run A"
    run_dir.mkdir()
    _write_csv(run_dir / "merged.csv")
    _write_csv(run_dir / "flow.csv")
    _write_csv(run_dir / "events.csv", "t_s,freq_set_hz,duty_set\n1,100,0\n")

    monkeypatch.setattr(
        io_local,
        "_is_cloud_placeholder_file",
        lambda path: path.name == "merged.csv",
    )

    pick = io_local.pick_best_csv(run_dir)

    assert pick.csv_path.name == "flow.csv"


def test_pick_best_csv_reports_when_only_cloud_placeholders_are_present(
    monkeypatch,
    tmp_path,
) -> None:
    """A run with only placeholder data files should raise a direct message."""
    run_dir = tmp_path / "Run A"
    run_dir.mkdir()
    _write_csv(run_dir / "merged.csv")
    _write_csv(run_dir / "flow.csv")
    _write_csv(run_dir / "events.csv", "t_s,freq_set_hz,duty_set\n1,100,0\n")

    monkeypatch.setattr(
        io_local,
        "_is_cloud_placeholder_file",
        lambda path: path.name in {"merged.csv", "flow.csv"},
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        io_local.pick_best_csv(run_dir)

    message = str(exc_info.value)
    assert "No locally available data CSV files found" in message
    assert "merged.csv" in message
    assert "flow.csv" in message
    assert "Always Keep on This Device" in message


def test_read_csv_preview_fails_fast_for_cloud_placeholders(
    monkeypatch,
    tmp_path,
) -> None:
    """Preview reads should stop before pandas work when the file is placeholder-only."""
    csv_path = tmp_path / "merged.csv"
    _write_csv(csv_path)

    monkeypatch.setattr(io_local, "_is_cloud_placeholder_file", lambda path: True)

    with pytest.raises(OSError) as exc_info:
        io_local.read_csv_preview(csv_path, nrows=5)

    assert "Always Keep on This Device" in str(exc_info.value)
