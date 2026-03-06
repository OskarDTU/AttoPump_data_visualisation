"""Tests for experiment-log parsing and lookup-backed classification."""

from __future__ import annotations

import pandas as pd

from app.data import data_processor
from app.data.experiment_log import (
    ExperimentLogEntry,
    parse_experiment_log_dataframe,
)
from app.data.test_catalog import rank_test_names


def test_parse_experiment_log_dataframe_splits_rows_and_extracts_metadata() -> None:
    """One spreadsheet row with multiple test IDs becomes multiple entries."""
    df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-03-01")],
            "Time": [pd.Timestamp("2026-03-01 14:31:00")],
            "Pump/BAR ID": ["Pump 3 - Niels"],
            "Test ID": [
                "20260224-1725-Bar3\n20260224-1735-Bar2",
            ],
            "Test type": [
                "500Hz, constant frequency constant voltage, 10 minutes long test.",
            ],
            "Voltage": ["2.5"],
            "Success/fail": ["SUCCESS"],
            "Data note": ["Calibration batch"],
            "Note": ["Stable output"],
            "Explanation": [""],
        }
    )

    entries = parse_experiment_log_dataframe(df)
    by_folder = {entry.folder: entry for entry in entries}

    assert set(by_folder) == {"20260224-1725-Bar3", "20260224-1735-Bar2"}
    entry = by_folder["20260224-1725-Bar3"]
    assert entry.inferred_test_type == "constant"
    assert entry.frequency_hz == 500.0
    assert entry.duration_s == 600.0
    assert entry.pump_bar_id == "Pump 3 - Niels"
    assert entry.date == "2026-03-01"
    assert entry.time == "14:31:00"
    assert "Stable output" in entry.combined_notes


def test_detect_test_type_uses_experiment_log_when_available(monkeypatch) -> None:
    """Experiment-log classification is used before metadata/regex fallback."""
    monkeypatch.setattr(
        data_processor,
        "lookup_experiment_log_entry",
        lambda root, run_name: ExperimentLogEntry(
            folder=run_name,
            normalized_folder="normalized",
            row_number=10,
            inferred_test_type="sweep",
        ),
    )
    monkeypatch.setattr(data_processor, "load_test_metadata", lambda: {})
    monkeypatch.setattr(data_processor, "load_user_patterns", lambda: [])
    monkeypatch.setattr(data_processor, "parse_sweep_spec_from_name", lambda name: None)

    from app.data import test_configs

    monkeypatch.setattr(test_configs, "get_test_config", lambda name: None)

    test_type, method, metadata = data_processor.detect_test_type(
        "20260306-example",
        df=None,
        data_root="/tmp/All_tests",
    )

    assert test_type == "sweep"
    assert method == "experiment_log"
    assert metadata is None


def test_rank_test_names_prefers_exact_and_prefix_matches() -> None:
    """Search ranking should put the obvious match first."""
    run_names = [
        "20260224-1735-Bar2",
        "20260224-1725-Bar3",
        "Test_Pump3_Niels_25022026_1348_20260225_135001_0001",
    ]

    ranked = rank_test_names(run_names, "20260224-1725")

    assert ranked[0] == "20260224-1725-Bar3"
