"""Shared data-loading pipeline used by all analysis pages.

Extracts the common load→detect→classify→prepare→bin pipeline that was
previously duplicated across explorer.py, analysis.py, and
report_builder.py into a single, reusable module.

Public API
----------
- ``load_csv_cached(path)``    — Streamlit-cached CSV read.
- ``load_and_classify_tests()``— bulk-load from folder names.
- ``LoadResult``               — structured result container.
- ``resolve_data_path()``      — sidebar widget + persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .config import PLOT_BIN_WIDTH_HZ
from .data_processor import (
    bin_by_frequency,
    detect_constant_frequency,
    detect_test_type,
    detect_time_format,
    guess_signal_column,
    guess_time_column,
    parse_sweep_spec_from_name,
    prepare_sweep_data,
    prepare_time_series_data,
)
from .io_local import (
    list_run_dirs,
    normalize_root,
    pick_best_csv,
    read_csv_full,
)
from .app_settings import (
    clean_data_folder_path,
    load_settings,
    save_settings,
)


# ══════════════════════════════════════════════════════════════════════════
# CACHED CSV LOADER (single definition — no more duplicates)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_csv_cached(csv_path_str: str) -> pd.DataFrame:
    """Load a CSV with 5-minute Streamlit caching."""
    return read_csv_full(csv_path_str)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR DATA-PATH WIDGET (single definition)
# ══════════════════════════════════════════════════════════════════════════

_DATA_PATH_WIDGET_KEY = "data_path"
_DATA_PATH_SELECT_KEY = "data_path_saved_selection"
_DATA_PATH_NAME_KEY = "data_path_save_name"
_DATA_PATH_STATUS_KEY = "_data_path_status"
_MANUAL_PATH_OPTION = "(Current session path)"


def _get_saved_paths() -> dict[str, str]:
    """Return the cleaned saved-path mapping from settings."""
    return load_settings().saved_data_paths


def _find_saved_name_for_path(path: str, saved_paths: dict[str, str]) -> str:
    """Return the saved-path name that matches *path*, if any."""
    for name, saved_path in saved_paths.items():
        if saved_path == path:
            return name
    return ""


def _ensure_data_source_state() -> None:
    """Bootstrap shared session state for the current data source."""
    settings = load_settings()
    saved_paths = settings.saved_data_paths
    current_path = clean_data_folder_path(settings.data_folder_path)
    selected_name = settings.selected_data_path_name

    if selected_name in saved_paths:
        current_path = saved_paths[selected_name]

    if _DATA_PATH_WIDGET_KEY not in st.session_state:
        st.session_state[_DATA_PATH_WIDGET_KEY] = current_path

    if _DATA_PATH_SELECT_KEY not in st.session_state:
        if selected_name in saved_paths:
            st.session_state[_DATA_PATH_SELECT_KEY] = selected_name
        else:
            matched_name = _find_saved_name_for_path(current_path, saved_paths)
            st.session_state[_DATA_PATH_SELECT_KEY] = matched_name or _MANUAL_PATH_OPTION

    if _DATA_PATH_NAME_KEY not in st.session_state:
        selected = st.session_state.get(_DATA_PATH_SELECT_KEY, _MANUAL_PATH_OPTION)
        st.session_state[_DATA_PATH_NAME_KEY] = (
            selected if selected in saved_paths else ""
        )


def _persist_settings_for_current_path() -> None:
    """Persist the current session path and selected saved-path name."""
    settings = load_settings()
    current_path = clean_data_folder_path(st.session_state.get(_DATA_PATH_WIDGET_KEY, ""))
    saved_paths = settings.saved_data_paths
    matched_name = _find_saved_name_for_path(current_path, saved_paths)

    if current_path != st.session_state.get(_DATA_PATH_WIDGET_KEY, ""):
        st.session_state[_DATA_PATH_WIDGET_KEY] = current_path

    if matched_name:
        st.session_state[_DATA_PATH_SELECT_KEY] = matched_name
        settings.selected_data_path_name = matched_name
    else:
        st.session_state[_DATA_PATH_SELECT_KEY] = _MANUAL_PATH_OPTION
        settings.selected_data_path_name = ""

    settings.data_folder_path = current_path
    save_settings(settings)


def _persist_data_path(widget_key: str) -> None:
    """Persist the current widget value into per-user settings."""
    del widget_key  # kept for callback compatibility
    _persist_settings_for_current_path()


def _on_saved_path_selection_change() -> None:
    """Apply the selected saved path to the current session."""
    settings = load_settings()
    saved_paths = settings.saved_data_paths
    selected = st.session_state.get(_DATA_PATH_SELECT_KEY, _MANUAL_PATH_OPTION)

    if selected in saved_paths:
        st.session_state[_DATA_PATH_WIDGET_KEY] = saved_paths[selected]
        st.session_state[_DATA_PATH_NAME_KEY] = selected
        settings.data_folder_path = saved_paths[selected]
        settings.selected_data_path_name = selected
    else:
        settings.data_folder_path = clean_data_folder_path(
            st.session_state.get(_DATA_PATH_WIDGET_KEY, "")
        )
        settings.selected_data_path_name = ""

    save_settings(settings)


def _save_current_path_as_named_source() -> tuple[str, str]:
    """Save the current path under the provided name."""
    name = str(st.session_state.get(_DATA_PATH_NAME_KEY, "")).strip()
    path = clean_data_folder_path(st.session_state.get(_DATA_PATH_WIDGET_KEY, ""))
    if not name:
        return "error", "Enter a name for the saved path."
    if not path:
        return "error", "Enter a data-source path before saving it."

    try:
        normalize_root(path)
    except Exception as exc:
        return "error", f"Cannot save invalid path: {exc}"

    settings = load_settings()
    settings.saved_data_paths[name] = path
    settings.selected_data_path_name = name
    settings.data_folder_path = path
    save_settings(settings)

    st.session_state[_DATA_PATH_SELECT_KEY] = name
    st.session_state[_DATA_PATH_NAME_KEY] = name
    st.session_state[_DATA_PATH_WIDGET_KEY] = path
    return "success", f"Saved data source '{name}'."


def _delete_selected_named_source() -> tuple[str, str]:
    """Delete the currently selected saved path."""
    settings = load_settings()
    selected = st.session_state.get(_DATA_PATH_SELECT_KEY, _MANUAL_PATH_OPTION)
    if selected not in settings.saved_data_paths:
        return "error", "Select a saved path to delete."

    settings.saved_data_paths.pop(selected, None)
    settings.selected_data_path_name = ""
    settings.data_folder_path = clean_data_folder_path(
        st.session_state.get(_DATA_PATH_WIDGET_KEY, "")
    )
    save_settings(settings)

    st.session_state[_DATA_PATH_SELECT_KEY] = _MANUAL_PATH_OPTION
    if st.session_state.get(_DATA_PATH_NAME_KEY) == selected:
        st.session_state[_DATA_PATH_NAME_KEY] = ""
    return "success", f"Deleted saved data source '{selected}'."


def _on_save_path_button_click() -> None:
    """Handle save-path button clicks inside a widget callback."""
    st.session_state[_DATA_PATH_STATUS_KEY] = _save_current_path_as_named_source()


def _on_delete_path_button_click() -> None:
    """Handle delete-path button clicks inside a widget callback."""
    st.session_state[_DATA_PATH_STATUS_KEY] = _delete_selected_named_source()


def render_data_source_sidebar() -> None:
    """Render the global sidebar controls for choosing a data source."""
    _ensure_data_source_state()
    saved_paths = _get_saved_paths()
    options = [_MANUAL_PATH_OPTION] + list(saved_paths)

    selected = st.session_state.get(_DATA_PATH_SELECT_KEY, _MANUAL_PATH_OPTION)
    if selected not in options:
        st.session_state[_DATA_PATH_SELECT_KEY] = _MANUAL_PATH_OPTION

    with st.sidebar:
        st.header("📁 Data Source")
        st.selectbox(
            "Saved data source",
            options=options,
            key=_DATA_PATH_SELECT_KEY,
            help="Choose a named path once for this session, or keep a manual path.",
            on_change=_on_saved_path_selection_change,
        )
        st.text_input(
            "Path to test data folder",
            placeholder="/Users/.../All_tests",
            help="Shared across all pages in this session.",
            key=_DATA_PATH_WIDGET_KEY,
            on_change=_persist_data_path,
            args=(_DATA_PATH_WIDGET_KEY,),
        )
        st.text_input(
            "Save current path as",
            placeholder="Oskars path",
            help="Store the current path under a reusable name for this user.",
            key=_DATA_PATH_NAME_KEY,
        )

        col_save, col_delete = st.columns(2)
        col_save.button(
            "Save Path",
            on_click=_on_save_path_button_click,
        )
        col_delete.button(
            "Delete Path",
            disabled=st.session_state.get(_DATA_PATH_SELECT_KEY, _MANUAL_PATH_OPTION)
            not in saved_paths,
            on_click=_on_delete_path_button_click,
        )

        status = st.session_state.pop(_DATA_PATH_STATUS_KEY, None)
        if status:
            level, message = status
            getattr(st, level)(message)

        st.caption("The selected data source is reused across all pages in this session.")
        st.divider()


def resolve_data_path(
    *,
    key_suffix: str = "",
    render_widget: bool = True,
) -> tuple[str, list[Path], list[str]]:
    """Resolve the current shared data-source path.

    When ``render_widget`` is true, also render the global sidebar controls.
    ``key_suffix`` is retained for call-site compatibility.

    Returns
    -------
    (data_folder_str, run_dirs, run_names)
        The raw path string, list of resolved run directories, and their
        names.  ``run_dirs`` / ``run_names`` are empty if the path is
        invalid.
    """
    del key_suffix  # legacy no-op; all pages should share the same path widget

    _ensure_data_source_state()
    if render_widget:
        render_data_source_sidebar()

    data_folder_str = clean_data_folder_path(st.session_state.get(_DATA_PATH_WIDGET_KEY, ""))
    settings = load_settings()
    if data_folder_str != settings.data_folder_path:
        settings.data_folder_path = data_folder_str
        settings.selected_data_path_name = _find_saved_name_for_path(
            data_folder_str,
            settings.saved_data_paths,
        )
        save_settings(settings)

    run_dirs: list[Path] = []
    run_names: list[str] = []

    if data_folder_str:
        try:
            root = normalize_root(data_folder_str)
            run_dirs = list(list_run_dirs(root))
            run_names = [p.name for p in run_dirs]
        except Exception as e:
            st.sidebar.error(f"❌ Invalid path: {e}")

    return data_folder_str, run_dirs, run_names


# ══════════════════════════════════════════════════════════════════════════
# LOAD RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class LoadResult:
    """Structured result from ``load_and_classify_tests``."""

    all_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    sweep_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    binned_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    const_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    signal_col: str = "flow"
    errors: list[str] = field(default_factory=list)
    test_types: dict[str, str] = field(default_factory=dict)
    const_freqs: dict[str, float] = field(default_factory=dict)

    @property
    def n_sweep(self) -> int:
        return len(self.sweep_data)

    @property
    def n_const(self) -> int:
        return len(self.const_data)

    @property
    def is_empty(self) -> bool:
        return not self.all_data


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def load_and_classify_tests(
    test_names: list[str],
    run_dirs: list[Path],
    run_names: list[str],
    *,
    bin_hz: float = PLOT_BIN_WIDTH_HZ,
    max_raw_points: int | None = None,
) -> LoadResult:
    """Load, classify, and prepare data for a list of test folders.

    This is the single shared implementation of the
    load→detect→classify→prepare→bin pipeline.

    Parameters
    ----------
    test_names : list[str]
        Folder names to load (must exist in *run_names*).
    run_dirs, run_names : list
        Full list of discovered run directories and their names.
    bin_hz : float
        Frequency bin width for sweep analysis.
    max_raw_points : int or None
        If set, cap raw sweep data at this many points.

    Returns
    -------
    LoadResult
        Structured container with all processed data.
    """
    result = LoadResult()
    run_map = {name: path for name, path in zip(run_names, run_dirs)}

    for name in test_names:
        run_dir = run_map.get(name)
        if run_dir is None:
            result.errors.append(f"{name}: not found in data folder")
            continue

        try:
            pick = pick_best_csv(run_dir)
            df = load_csv_cached(str(pick.csv_path))
            if df.empty:
                result.errors.append(f"{name}: empty CSV")
                continue

            time_col = guess_time_column(df)
            sig_col = guess_signal_column(df, "flow")
            if not time_col or not sig_col:
                result.errors.append(f"{name}: cannot detect columns")
                continue

            if result.signal_col == "flow" and sig_col:
                result.signal_col = sig_col

            time_fmt = detect_time_format(df, time_col)
            ts_df = prepare_time_series_data(
                df, time_col, sig_col,
                parse_time=(time_fmt == "absolute_timestamp"),
            )
            result.all_data[name] = ts_df

            # Classify test type
            ttype, _, _ = detect_test_type(name, df, data_root=run_dir.parent)
            result.test_types[name] = ttype
            has_freq = "freq_set_hz" in df.columns

            if ttype == "sweep" or (has_freq and df["freq_set_hz"].dropna().nunique() > 1):
                # ── Sweep path ──
                spec = parse_sweep_spec_from_name(name)
                if has_freq or (spec and spec.duration_s > 0):
                    sweep_df = prepare_sweep_data(
                        ts_df, time_col, sig_col,
                        spec=spec,
                        parse_time=(time_fmt == "absolute_timestamp"),
                        full_df=df if has_freq else None,
                    )
                    if max_raw_points and len(sweep_df) > max_raw_points:
                        result.sweep_data[name] = sweep_df.sample(
                            n=max_raw_points, random_state=42,
                        )
                    else:
                        result.sweep_data[name] = sweep_df

                    try:
                        binned = bin_by_frequency(
                            sweep_df, value_col=sig_col,
                            freq_col="Frequency", bin_hz=bin_hz,
                        )
                        result.binned_data[name] = binned
                    except Exception as be:
                        result.errors.append(f"{name}: binning — {be}")
                else:
                    # Has sweep name but no frequency data → treat as constant
                    result.const_data[name] = ts_df
                    cf = detect_constant_frequency(df, name, data_root=run_dir.parent)
                    if cf:
                        result.const_freqs[name] = cf
            else:
                # ── Constant frequency path ──
                result.const_data[name] = ts_df
                cf = detect_constant_frequency(df, name, data_root=run_dir.parent)
                if cf:
                    result.const_freqs[name] = cf

        except Exception as e:
            result.errors.append(f"{name}: {e}")

    return result
