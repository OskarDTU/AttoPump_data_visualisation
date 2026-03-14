"""Regression tests for merged-sweep hold handling and error-bar rendering."""

from __future__ import annotations

import pandas as pd

from app.data.data_processor import bin_by_frequency, prepare_sweep_data
from app.plots.plot_generator import (
    build_sweep_average_trace,
    downsample_sweep_points,
    plot_sweep_all_points,
    plot_sweep_binned,
    plot_sweep_per_sweep_average,
)


def test_prepare_sweep_data_flags_long_constant_frequency_holds() -> None:
    """Long constant-frequency pauses in merged data should be tagged as holds."""
    df = pd.DataFrame(
        {
            "t_s": list(range(100)),
            "flow": [float(i % 20) for i in range(100)],
            "freq_set_hz": ([100] * 10) + ([200] * 10) + ([300] * 10) + ([500] * 60) + ([700] * 10),
        }
    )

    sweep_df = prepare_sweep_data(
        df,
        time_col="t_s",
        signal_col="flow",
        parse_time=False,
        full_df=df,
    )

    hold_rows = sweep_df.loc[sweep_df["IsFrequencyHold"]]
    assert not hold_rows.empty
    assert set(hold_rows["Frequency"]) == {500.0}
    assert hold_rows["FrequencyRunPoints"].min() >= 60


def test_bin_by_frequency_skips_flagged_hold_segments() -> None:
    """Frequency binning should exclude rows already tagged as hold segments."""
    sweep_df = pd.DataFrame(
        {
            "Frequency": [100.0, 100.0, 200.0, 200.0, 500.0, 500.0],
            "flow": [10.0, 12.0, 20.0, 22.0, 999.0, 1001.0],
            "IsFrequencyHold": [False, False, False, False, True, True],
        }
    )

    binned = bin_by_frequency(sweep_df, value_col="flow", bin_hz=25.0)

    assert binned["freq_center"].max() < 500.0


def test_sweep_plots_use_error_bars_and_filter_hold_points() -> None:
    """Sweep plots should render whisker-style std bars and omit hold rows."""
    raw_df = pd.DataFrame(
        {
            "Frequency": [100.0, 150.0, 500.0],
            "flow": [10.0, 15.0, 999.0],
            "Sweep": [0, 0, 0],
            "IsFrequencyHold": [False, False, True],
        }
    )
    fig_raw = plot_sweep_all_points(raw_df, y_col="flow", color_col=None)
    assert list(fig_raw.data[0].x) == [100.0, 150.0]

    binned_df = pd.DataFrame(
        {
            "freq_center": [100.0, 150.0],
            "mean": [10.0, 15.0],
            "std": [1.0, 1.5],
        }
    )
    fig_binned = plot_sweep_binned(binned_df, show_error_bars=True)
    assert len(fig_binned.data) == 1
    assert fig_binned.data[0].error_y.visible is True


def test_downsample_sweep_points_keeps_all_sweeps_visible() -> None:
    """Balanced downsampling should keep at least one point from each sweep."""
    df = pd.DataFrame(
        {
            "Frequency": list(range(15)),
            "flow": [float(i) for i in range(15)],
            "Sweep": ([0] * 5) + ([1] * 5) + ([2] * 5),
        }
    )

    capped = downsample_sweep_points(df, max_points=6)

    assert len(capped) <= 6
    assert set(capped["Sweep"]) == {0, 1, 2}


def test_plot_sweep_per_sweep_average_uses_error_bars_and_filters_holds() -> None:
    """Per-sweep averaged traces should keep separate sweeps and ignore holds."""
    df = pd.DataFrame(
        {
            "Frequency": [100.0, 125.0, 150.0, 175.0, 500.0, 100.0, 125.0, 150.0, 175.0],
            "flow": [10.0, 11.0, 12.0, 13.0, 999.0, 20.0, 21.0, 22.0, 23.0],
            "Sweep": [0, 0, 0, 0, 0, 1, 1, 1, 1],
            "IsFrequencyHold": [False, False, False, False, True, False, False, False, False],
        }
    )

    fig = plot_sweep_per_sweep_average(
        df,
        signal_col="flow",
        bin_hz=25.0,
        show_error_bars=True,
    )

    assert len(fig.data) == 2
    assert [trace.name for trace in fig.data] == ["Sweep 1", "Sweep 2"]
    assert all(trace.error_y.visible is True for trace in fig.data)
    assert all(500.0 not in list(trace.x) for trace in fig.data)


def test_build_sweep_average_trace_averages_sweeps_equally() -> None:
    """The combined average trace should average per-sweep curves, not raw points."""
    df = pd.DataFrame(
        {
            "Frequency": [100.0, 120.0, 140.0, 160.0, 500.0, 100.0, 120.0, 140.0, 160.0],
            "flow": [10.0, 12.0, 14.0, 16.0, 999.0, 20.0, 22.0, 24.0, 26.0],
            "Sweep": [0, 0, 0, 0, 0, 1, 1, 1, 1],
            "IsFrequencyHold": [False, False, False, False, True, False, False, False, False],
        }
    )

    avg_df = build_sweep_average_trace(
        df,
        signal_col="flow",
        bin_hz=50.0,
    )

    assert list(avg_df["freq_center"]) == [125.0, 175.0]
    assert list(avg_df["mean"]) == [17.0, 21.0]
    assert list(avg_df["std"]) == [5.0, 5.0]
