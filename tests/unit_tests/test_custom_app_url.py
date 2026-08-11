"""_get_custom_app_url_from_file must resolve copy AND editable installs.

Regression: the relay reaches a running app's /rpc via get_running_app_url ->
_get_custom_app_url_from_file. It used to only probe site-packages/<app>/main.py,
which is absent for an editable (-e) install (the .pth redirects imports), so the
relay returned "no app is running" for every relayed request while apps.status
(daemon-local) still reported running.

The editable resolution runs the *app's own* python (get_app_python) in a
subprocess, so it works even when the app lives in a separate venv (apps_venv on
wireless) that the daemon interpreter cannot import.
"""

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


def test_editable_install_resolves_via_app_python(monkeypatch, tmp_path):
    """Editable install: the app's own python resolves the origin out-of-process."""
    pkg = _write_app(tmp_path, "some_app")
    empty_sp = tmp_path / "site-packages"
    empty_sp.mkdir()
    monkeypatch.setattr(lcv, "_get_app_site_packages", lambda *a, **k: empty_sp)
    # The app's python (e.g. apps_venv) prints the package origin; the daemon
    # interpreter could not have found this in-process.
    monkeypatch.setattr(
        lcv, "get_app_python", lambda *a, **k: Path("/venvs/apps_venv/bin/python")
    )
    monkeypatch.setattr(
        lcv.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=f"{pkg / '__init__.py'}\n", returncode=0
        ),
    )
    assert (
        lcv._get_custom_app_url_from_file("some_app", wireless_version=True)
        == "http://0.0.0.0:7860/"
    )


def test_missing_everywhere_returns_none(monkeypatch, tmp_path):
    """Neither the file path nor the app-python resolution finds it -> None."""
    monkeypatch.setattr(lcv, "_get_app_site_packages", lambda *a, **k: tmp_path)
    monkeypatch.setattr(
        lcv, "get_app_python", lambda *a, **k: Path("/nonexistent/python")
    )
    monkeypatch.setattr(
        lcv.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="\n", returncode=0),
    )
    assert lcv._get_custom_app_url_from_file("nope_app") is None
