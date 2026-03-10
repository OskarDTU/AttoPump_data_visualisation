"""Tests for automatic frequency-bin recommendations."""

from __future__ import annotations

import pandas as pd

from app.data.data_processor import recommend_frequency_bin_widths


def test_recommend_frequency_bin_widths_tracks_start_offset_logic() -> None:
    """The recommendation should clear the largest start-point offset."""
    df_a = pd.DataFrame({
        "Frequency": [0, 10, 20, 30, 40, 50],
        "Sweep": [0, 0, 0, 0, 0, 0],
    })
    df_b = pd.DataFrame({
        "Frequency": [12, 22, 32, 42, 52, 62],
        "Sweep": [0, 0, 0, 0, 0, 0],
    })

    recommendation = recommend_frequency_bin_widths([df_a, df_b])

    assert recommendation["start_alignment_gap_hz"] == 12.0
    assert recommendation["test_bin_hz"] >= 12.5
    assert recommendation["average_bin_hz"] >= recommendation["test_bin_hz"]
    assert recommendation["start_spread_hz"] > 0


def test_recommend_frequency_bin_widths_handles_single_sweep() -> None:
    """Single-sweep data should still return bounded defaults."""
    df = pd.DataFrame({
        "Frequency": [1, 2, 3, 4, 5],
        "Sweep": [0, 0, 0, 0, 0],
    })

    recommendation = recommend_frequency_bin_widths([df], min_bin_hz=0.5, max_bin_hz=100.0)

    assert 0.5 <= recommendation["test_bin_hz"] <= 100.0
    assert recommendation["average_bin_hz"] >= recommendation["test_bin_hz"]
