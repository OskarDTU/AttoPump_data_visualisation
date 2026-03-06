"""Shared Streamlit widget: prompt the user to classify unknown tests.

When the app loads test folders and cannot auto-detect whether a test
is a *constant-frequency* or *frequency-sweep* test, this widget
renders inline forms that ask the user to define the type.  Once saved,
the classification is remembered via ``test_configs.json`` and never
asked again.

Usage
-----
Call ``render_unknown_test_prompt(unknown_names, run_dirs, run_names)``
**before** data loading / analysis.  It returns ``True`` if all tests
are now classified (safe to proceed), or ``False`` if there are still
unclassified tests that need user input.

Call ``classify_tests_quick(names, run_dirs, run_names)`` to get a
dict of ``{name: (test_type, detection_method)}`` and a list of names
that are still "unknown".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ..data.config import DEFAULT_CONSTANT_FREQUENCY_HZ
from ..data.data_processor import (
    detect_test_type,
    parse_sweep_spec_from_name,
)
from ..data.io_local import pick_best_csv, read_csv_full
from ..data.test_configs import (
    TestConfig,
    get_test_config,
    save_test_config,
)


# ── Cached CSV loader (shared with pages) ──────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _load_csv_cached(csv_path_str: str) -> pd.DataFrame:
    return read_csv_full(csv_path_str)


# ── Quick classification scan ──────────────────────────────────────────

def classify_tests_quick(
    names: list[str],
    run_dirs: list[Path],
    run_names: list[str],
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Quickly classify tests and identify unknowns.

    Returns
    -------
    (classifications, unknowns)
        classifications : {name: (test_type, detection_method)}
        unknowns : list of names where type == "unknown"
    """
    classifications: dict[str, tuple[str, str]] = {}
    unknowns: list[str] = []
    data_root = run_dirs[0].parent if run_dirs else None

    for name in names:
        idx = run_names.index(name) if name in run_names else -1
        if idx < 0:
            continue

        # Try detection WITHOUT loading CSV first (fast path)
        ttype, method, _ = detect_test_type(name, df=None, data_root=data_root)
        if ttype != "unknown":
            classifications[name] = (ttype, method)
            continue

        # Need to peek at the CSV for the freq_set_hz column
        try:
            pick = pick_best_csv(run_dirs[idx])
            df = _load_csv_cached(str(pick.csv_path))
            ttype, method, _ = detect_test_type(name, df, data_root=data_root)
        except Exception:
            ttype, method = "unknown", "unknown"

        classifications[name] = (ttype, method)
        if ttype == "unknown":
            unknowns.append(name)

    return classifications, unknowns


# ── Main widget ────────────────────────────────────────────────────────

def render_unknown_test_prompt(
    unknown_names: list[str],
    run_dirs: list[Path],
    run_names: list[str],
    *,
    key_prefix: str = "utp",
) -> bool:
    """Render inline classification forms for unknown tests.

    Shows a warning banner listing unclassified tests and expanders
    with forms to define each one.  When the user saves a classification,
    the page reruns and the test is no longer unknown.

    Parameters
    ----------
    unknown_names : list[str]
        Folder names that need classification.
    run_dirs, run_names : list
        Full list of test folders (for building sweep-spec hints).
    key_prefix : str
        Streamlit key namespace (use different prefixes if this widget
        appears on multiple pages simultaneously, though Streamlit pages
        are never rendered concurrently).

    Returns
    -------
    bool
        ``True`` if there are **no** remaining unknowns (safe to
        proceed).  ``False`` if there are still unclassified tests.
    """
    if not unknown_names:
        return True

    st.warning(
        f"⚠️ **{len(unknown_names)} test(s) could not be auto-classified.**  "
        "The folder name doesn't match any known pattern and no saved "
        "configuration exists.  Please define the test type below so the "
        "app knows how to analyse them."
    )

    any_saved = False

    for i, name in enumerate(unknown_names):
        with st.expander(
            f"❓ {name} — needs classification",
            expanded=(i == 0),  # auto-expand the first one
        ):
            # Try to hint from folder name
            parsed_spec = parse_sweep_spec_from_name(name)
            if parsed_spec:
                st.info(
                    f"💡 Hint: the folder name suggests a sweep "
                    f"({parsed_spec}), but this wasn't confident enough "
                    f"to auto-classify.  Please confirm below."
                )

            col_type, col_save = st.columns([3, 1])
            with col_type:
                cfg_type = st.selectbox(
                    "Test type",
                    options=["constant", "sweep"],
                    format_func=lambda x: (
                        "Constant Frequency" if x == "constant"
                        else "Frequency Sweep"
                    ),
                    key=f"{key_prefix}_type_{i}",
                )

            if cfg_type == "constant":
                cfg_freq = st.number_input(
                    "Frequency (Hz)",
                    value=float(DEFAULT_CONSTANT_FREQUENCY_HZ),
                    min_value=0.1,
                    step=10.0,
                    key=f"{key_prefix}_freq_{i}",
                )
                cfg_start = cfg_end = cfg_dur = None
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    cfg_start = st.number_input(
                        "Start Hz",
                        value=float(parsed_spec.start_hz if parsed_spec else 1.0),
                        min_value=0.0,
                        step=1.0,
                        key=f"{key_prefix}_start_{i}",
                    )
                with c2:
                    cfg_end = st.number_input(
                        "End Hz",
                        value=float(parsed_spec.end_hz if parsed_spec else 1000.0),
                        min_value=0.0,
                        step=10.0,
                        key=f"{key_prefix}_end_{i}",
                    )
                with c3:
                    cfg_dur = st.number_input(
                        "Duration (s)",
                        value=float(parsed_spec.duration_s if parsed_spec else 0.0),
                        min_value=0.0,
                        step=1.0,
                        help="Duration of one sweep cycle. 0 = estimate from data.",
                        key=f"{key_prefix}_dur_{i}",
                    )
                cfg_freq = None

            cfg_note = st.text_input(
                "Note (optional)",
                key=f"{key_prefix}_note_{i}",
            )

            with col_save:
                st.markdown("&nbsp;")  # vertical alignment
                if st.button("💾 Save", key=f"{key_prefix}_save_{i}"):
                    new_cfg = TestConfig(
                        test_type=cfg_type,
                        frequency_hz=cfg_freq,
                        start_hz=cfg_start,
                        end_hz=cfg_end,
                        duration_s=cfg_dur,
                        note=cfg_note,
                    )
                    save_test_config(name, new_cfg)
                    st.success(f"✅ Saved **{cfg_type}** for `{name}`")
                    any_saved = True

    if any_saved:
        st.rerun()

    # Offer to classify all at once if many unknowns
    if len(unknown_names) > 2:
        st.divider()
        st.markdown("**Quick classify all remaining unknowns:**")
        qc1, qc2 = st.columns(2)
        with qc1:
            if st.button(
                f"🔵 Mark all {len(unknown_names)} as **Constant Frequency**",
                key=f"{key_prefix}_bulk_const",
            ):
                for name in unknown_names:
                    save_test_config(name, TestConfig(
                        test_type="constant",
                        frequency_hz=DEFAULT_CONSTANT_FREQUENCY_HZ,
                        note="Bulk-classified as constant",
                    ))
                st.success(f"✅ Classified {len(unknown_names)} tests as constant frequency")
                st.rerun()
        with qc2:
            if st.button(
                f"🟢 Mark all {len(unknown_names)} as **Frequency Sweep**",
                key=f"{key_prefix}_bulk_sweep",
            ):
                for name in unknown_names:
                    spec = parse_sweep_spec_from_name(name)
                    save_test_config(name, TestConfig(
                        test_type="sweep",
                        start_hz=spec.start_hz if spec else 1.0,
                        end_hz=spec.end_hz if spec else 1000.0,
                        duration_s=spec.duration_s if spec else 0.0,
                        note="Bulk-classified as sweep",
                    ))
                st.success(f"✅ Classified {len(unknown_names)} tests as sweep")
                st.rerun()

    return False  # still have unknowns
