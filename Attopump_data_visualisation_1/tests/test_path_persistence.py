"""Regression tests for saved data-path behavior across pages."""

from __future__ import annotations

import json

from app.data import app_settings, loader


class _FakeStreamlit:
    """Small test double for the bits of Streamlit used by resolve_data_path."""

    def __init__(self, *, session_state: dict | None = None) -> None:
        self.session_state = {} if session_state is None else session_state
        self.sidebar = self
        self._pending_inputs: list[str] = []
        self.errors: list[str] = []

    def queue_input(self, value: str) -> None:
        self._pending_inputs.append(value)

    def __enter__(self) -> "_FakeStreamlit":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def header(self, *args, **kwargs) -> None:
        return None

    def error(self, message: str) -> None:
        self.errors.append(message)

    def text_input(self, label: str, **kwargs) -> str:
        key = kwargs["key"]
        if key not in self.session_state:
            self.session_state[key] = ""
        if self._pending_inputs:
            self.session_state[key] = self._pending_inputs.pop(0)
            on_change = kwargs.get("on_change")
            if on_change:
                on_change(*kwargs.get("args", ()))
        return self.session_state[key]


def _patch_settings_paths(monkeypatch, tmp_path) -> tuple:
    settings_dir = tmp_path / "user_settings"
    settings_path = settings_dir / "app_settings.json"
    shared_settings_path = tmp_path / "shared_app_settings.json"
    legacy_path = tmp_path / "legacy_app_settings.json"
    monkeypatch.setattr(app_settings, "_SETTINGS_DIR", settings_dir)
    monkeypatch.setattr(app_settings, "_SETTINGS_PATH", settings_path)
    monkeypatch.setattr(app_settings, "_SHARED_SETTINGS_PATH", shared_settings_path)
    monkeypatch.setattr(app_settings, "_LEGACY_SETTINGS_PATH", legacy_path)
    return settings_path, shared_settings_path, legacy_path


def test_load_settings_migrates_legacy_path_and_cleans_shell_escapes(
    monkeypatch,
    tmp_path,
) -> None:
    """Legacy repo-local settings should migrate into per-user storage."""
    settings_path, shared_settings_path, legacy_path = _patch_settings_paths(monkeypatch, tmp_path)
    legacy_payload = {
        "data_folder_path": "/tmp/AttoPump\\ Project/All\\ tests",
        "saved_data_paths": {"Team path": "/tmp/AttoPump\\ Project/All\\ tests"},
        "selected_data_path_name": "Team path",
    }
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    settings = app_settings.load_settings()

    assert settings.data_folder_path == "/tmp/AttoPump Project/All tests"
    assert settings.saved_data_paths == {"Team path": "/tmp/AttoPump Project/All tests"}
    assert settings.selected_data_path_name == "Team path"
    assert settings_path.exists()
    local_saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert local_saved["data_folder_path"] == "/tmp/AttoPump Project/All tests"
    assert local_saved["saved_data_paths"] == {}
    shared_saved = json.loads(shared_settings_path.read_text(encoding="utf-8"))
    assert shared_saved["saved_data_paths"] == {
        "Team path": "/tmp/AttoPump Project/All tests"
    }


def test_resolve_data_path_shares_one_saved_path_across_pages(
    monkeypatch,
    tmp_path,
) -> None:
    """Entering the path on one page should make it available on the others."""
    settings_path, _, _ = _patch_settings_paths(monkeypatch, tmp_path)
    data_root = tmp_path / "All tests"
    (data_root / "Run A").mkdir(parents=True)

    app_settings.save_settings(
        app_settings.AppSettings(
            data_folder_path=str(data_root).replace(" ", "\\ "),
        )
    )

    shared_state: dict[str, str] = {}
    fake_st = _FakeStreamlit(session_state=shared_state)
    monkeypatch.setattr(loader, "st", fake_st)

    first_path, first_dirs, first_names = loader.resolve_data_path(
        key_suffix="ex",
        render_widget=False,
    )
    second_path, second_dirs, second_names = loader.resolve_data_path(
        key_suffix="an",
        render_widget=False,
    )

    assert first_path == str(data_root)
    assert second_path == str(data_root)
    assert [p.name for p in first_dirs] == ["Run A"]
    assert [p.name for p in second_dirs] == ["Run A"]
    assert first_names == ["Run A"]
    assert second_names == ["Run A"]
    assert shared_state["data_path"] == str(data_root)
    assert shared_state["data_path_saved_selection"] == "(Current session path)"
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved["data_folder_path"] == str(data_root)

    fresh_st = _FakeStreamlit()
    monkeypatch.setattr(loader, "st", fresh_st)
    third_path, third_dirs, third_names = loader.resolve_data_path(
        key_suffix="mg",
        render_widget=False,
    )

    assert third_path == str(data_root)
    assert [p.name for p in third_dirs] == ["Run A"]
    assert third_names == ["Run A"]
    assert fresh_st.session_state["data_path"] == str(data_root)
    assert fresh_st.session_state["data_path_saved_selection"] == "(Current session path)"


def test_named_path_can_be_saved_and_selected_for_session(
    monkeypatch,
    tmp_path,
) -> None:
    """Named saved paths should be reusable without retyping the folder."""
    settings_path, shared_settings_path, _ = _patch_settings_paths(monkeypatch, tmp_path)
    data_root = tmp_path / "All tests"
    (data_root / "Run A").mkdir(parents=True)

    fake_st = _FakeStreamlit(
        session_state={
            "data_path": str(data_root),
            "data_path_save_name": "Oskars path",
            "data_path_saved_selection": "(Current session path)",
        }
    )
    monkeypatch.setattr(loader, "st", fake_st)

    level, message = loader._save_current_path_as_named_source()

    assert level == "success"
    assert "Oskars path" in message
    local_saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert local_saved["saved_data_paths"] == {}
    assert local_saved["selected_data_path_name"] == "Oskars path"
    shared_saved = json.loads(shared_settings_path.read_text(encoding="utf-8"))
    assert shared_saved["saved_data_paths"] == {"Oskars path": str(data_root)}

    fake_st.session_state["data_path"] = ""
    fake_st.session_state["data_path_saved_selection"] = "Oskars path"
    loader._on_saved_path_selection_change()

    assert fake_st.session_state["data_path"] == str(data_root)


def test_load_settings_uses_shared_named_paths_for_other_users(
    monkeypatch,
    tmp_path,
) -> None:
    """A second user should see named data-source paths saved in the repo store."""
    _, shared_settings_path, _ = _patch_settings_paths(monkeypatch, tmp_path)
    shared_settings_path.write_text(
        json.dumps(
            {
                "saved_data_paths": {
                    "Shared lab path": "/tmp/shared/All tests",
                }
            }
        ),
        encoding="utf-8",
    )

    settings = app_settings.load_settings()

    assert settings.saved_data_paths == {"Shared lab path": "/tmp/shared/All tests"}
    assert settings.selected_data_path_name == ""
    assert settings.data_folder_path == ""
