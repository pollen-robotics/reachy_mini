"""_get_custom_app_url_from_file must resolve copy AND editable installs.

Regression: the relay reaches a running app's /rpc via get_running_app_url ->
_get_custom_app_url_from_file. It used to only probe site-packages/<app>/main.py,
which is absent for an editable (-e) install (the .pth redirects imports), so the
relay returned "no app is running" for every relayed request while apps.status
(daemon-local) still reported running.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from reachy_mini.apps.sources import local_common_venv as lcv

MAIN = 'custom_app_url: str | None = "http://0.0.0.0:7860/"\n'


def _write_app(root: Path, name: str) -> Path:
    pkg = root / name
    pkg.mkdir(parents=True)
    (pkg / "main.py").write_text(MAIN, encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return pkg


def test_copy_install_reads_site_packages(monkeypatch, tmp_path):
    """A regular copy install: main.py physically under site-packages/<app>."""
    _write_app(tmp_path, "some_app")
    monkeypatch.setattr(lcv, "_get_app_site_packages", lambda *a, **k: tmp_path)
    assert lcv._get_custom_app_url_from_file("some_app") == "http://0.0.0.0:7860/"


def test_editable_install_falls_back_to_importlib(monkeypatch, tmp_path):
    """Editable install: nothing under site-packages; importlib spec finds it."""
    pkg = _write_app(tmp_path, "some_app")
    # site-packages exists but has no <app>/main.py (the -e case).
    empty_sp = tmp_path / "site-packages"
    empty_sp.mkdir()
    monkeypatch.setattr(lcv, "_get_app_site_packages", lambda *a, **k: empty_sp)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(pkg / "__init__.py")),
    )
    assert lcv._get_custom_app_url_from_file("some_app") == "http://0.0.0.0:7860/"


def test_missing_everywhere_returns_none(monkeypatch, tmp_path):
    """Neither the file path nor importlib finds it -> None (no crash)."""
    monkeypatch.setattr(lcv, "_get_app_site_packages", lambda *a, **k: tmp_path)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert lcv._get_custom_app_url_from_file("nope_app") is None
