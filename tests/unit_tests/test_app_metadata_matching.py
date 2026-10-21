"""Installed-app metadata matching once the catalog no longer carries siblings."""

import json
import logging
from pathlib import Path

import pytest

from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.apps.sources import local_common_venv as lcv


def _write_metadata(parent: Path, name: str, data: dict) -> None:
    metadata_dir = parent / ".app_metadata"
    metadata_dir.mkdir(exist_ok=True)
    (metadata_dir / f"{name}.json").write_text(json.dumps(data))


def test_find_metadata_by_id_when_siblings_is_null(monkeypatch, tmp_path) -> None:
    # Metadata saved from a catalog entry without a file list must still match
    # on extra.id; a null siblings value must not abort the scan of that file.
    monkeypatch.setattr(lcv, "_get_venv_parent_dir", lambda: tmp_path)
    _write_metadata(
        tmp_path, "my-space", {"id": "owner/robot_dancer", "siblings": None}
    )

    found = lcv._find_metadata_for_entry_point("robot_dancer")

    assert found["id"] == "owner/robot_dancer"


@pytest.mark.asyncio
async def test_install_saves_space_file_list_as_siblings(monkeypatch, tmp_path) -> None:
    # The catalog no longer ships siblings, but the entry-point matcher relies
    # on them, so install must record the space's file list itself.
    saved: dict = {}
    space_files = ["pyproject.toml", "robot_dancer/__init__.py"]

    class _FakeSpaceInfo:
        id = "owner/my-space"
        private = False

    class _FakeHfApi:
        def __init__(self, token=None) -> None:  # type: ignore[no-untyped-def]
            pass

        def space_info(self, repo_id):  # type: ignore[no-untyped-def]
            return _FakeSpaceInfo()

        def list_repo_files(self, repo_id, repo_type, token=None):  # type: ignore[no-untyped-def]
            return list(space_files)

    download_dir = tmp_path / "snapshot"
    download_dir.mkdir()
    (download_dir / "pyproject.toml").write_text("[project]\nname='x'\n")

    async def _fake_running_command(*args, **kwargs):  # type: ignore[no-untyped-def]
        return 0

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    monkeypatch.setattr(lcv, "snapshot_download", lambda **kwargs: str(download_dir))
    monkeypatch.setattr(lcv, "running_command", _fake_running_command)
    monkeypatch.setattr(lcv, "_should_use_separate_venvs", lambda *a, **k: False)
    monkeypatch.setattr(
        lcv, "_save_app_metadata", lambda name, data: saved.update({name: data})
    )

    app = AppInfo(
        name="my-space",
        description="",
        url="https://huggingface.co/spaces/owner/my-space",
        source_kind=SourceKind.HF_SPACE,
        extra={"id": "owner/my-space", "private": False},
    )

    ret = await lcv.install_package(app, logging.getLogger("test"))

    assert ret == 0
    assert saved["my-space"]["id"] == "owner/my-space"
    assert saved["my-space"]["siblings"] == [{"rfilename": f} for f in space_files]
