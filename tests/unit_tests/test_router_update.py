"""Unit tests for the update router (install-source, available, info, refs, start)."""

from unittest.mock import MagicMock

import pytest
import requests

from reachy_mini.daemon.app import bg_job_register
from reachy_mini.daemon.app.bg_job_register import JobInfo, JobStatus
from reachy_mini.daemon.app.routers import update


@pytest.fixture(autouse=True)
def _release_lock():
    """Ensure busy_lock starts unlocked for each test."""
    if update.busy_lock.locked():
        update.busy_lock.release()
    yield
    if update.busy_lock.locked():
        update.busy_lock.release()


def test_install_source(monkeypatch, router_app):
    """install-source returns the helper's dict verbatim."""
    src = {"version": "1.2.3", "source": "pypi", "ref": ""}
    monkeypatch.setattr(update, "get_install_source", lambda pkg: src)
    client = router_app(update.router)

    resp = client.get("/update/install-source")

    assert resp.status_code == 200
    assert resp.json() == src


def test_available(monkeypatch, router_app):
    """available reports current/available versions and the update flag."""
    monkeypatch.setattr(update, "get_local_version", lambda pkg: "1.0.0")
    monkeypatch.setattr(update, "is_update_available", lambda pkg, pre: True)
    monkeypatch.setattr(update, "get_pypi_version", lambda pkg, pre: "1.1.0")
    client = router_app(update.router)

    resp = client.get("/update/available")

    assert resp.status_code == 200
    assert resp.json() == {
        "update": {
            "reachy_mini": {
                "is_available": True,
                "current_version": "1.0.0",
                "available_version": "1.1.0",
            }
        }
    }


def test_available_connection_error(monkeypatch, router_app):
    """available degrades to 'unknown' when the PyPI check can't connect."""
    monkeypatch.setattr(update, "get_local_version", lambda pkg: "1.0.0")

    def _boom(pkg, pre):
        raise requests.exceptions.ConnectionError

    monkeypatch.setattr(update, "is_update_available", _boom)
    client = router_app(update.router)

    resp = client.get("/update/available")

    assert resp.status_code == 200
    entry = resp.json()["update"]["reachy_mini"]
    assert entry["is_available"] is False
    assert entry["available_version"] == "unknown"


def test_available_busy(router_app):
    """available refuses while an update holds busy_lock."""
    update.busy_lock.acquire()
    client = router_app(update.router)

    resp = client.get("/update/available")

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Update is in progress"


def test_info_found(monkeypatch, router_app):
    """info returns the job's JobInfo."""
    info = JobInfo(command="update_reachy_mini", status=JobStatus.DONE, logs=["ok"])
    monkeypatch.setattr(bg_job_register, "get_info", lambda jid: info)
    client = router_app(update.router)

    resp = client.get("/update/info", params={"job_id": "abc"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "done"
    assert resp.json()["logs"] == ["ok"]


def test_info_not_found(monkeypatch, router_app):
    """Unknown job id -> 404."""

    def _boom(jid):
        raise ValueError("Job ID not found")

    monkeypatch.setattr(bg_job_register, "get_info", _boom)
    client = router_app(update.router)

    resp = client.get("/update/info", params={"job_id": "nope"})

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Job ID not found"


def _fake_resp(status_code):
    """Build a stand-in requests.Response with a status code."""
    r = MagicMock()
    r.status_code = status_code
    return r


def test_validate_ref_valid(monkeypatch, router_app):
    """A ref GitHub knows -> valid=True."""
    monkeypatch.setattr(update.requests, "get", lambda url, timeout: _fake_resp(200))
    client = router_app(update.router)

    resp = client.get("/update/validate-ref", params={"git_ref": "v1.0.0"})

    assert resp.status_code == 200
    assert resp.json() == {"valid": True, "ref": "v1.0.0"}


def test_validate_ref_invalid(monkeypatch, router_app):
    """A ref GitHub returns 404 for -> valid=False with an error message."""
    monkeypatch.setattr(update.requests, "get", lambda url, timeout: _fake_resp(404))
    client = router_app(update.router)

    resp = client.get("/update/validate-ref", params={"git_ref": "ghost"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert "ghost" in body["error"]


def test_validate_ref_empty(router_app):
    """Blank ref -> 400 before any network call."""
    client = router_app(update.router)

    resp = client.get("/update/validate-ref", params={"git_ref": "  "})

    assert resp.status_code == 400


def test_validate_ref_github_down(monkeypatch, router_app):
    """A RequestException from GitHub -> 503."""

    def _boom(url, timeout):
        raise requests.RequestException("timeout")

    monkeypatch.setattr(update.requests, "get", _boom)
    client = router_app(update.router)

    resp = client.get("/update/validate-ref", params={"git_ref": "v1.0.0"})

    assert resp.status_code == 503


def test_start(monkeypatch, router_app):
    """start schedules the job (no pip) and returns its id."""
    monkeypatch.setattr(update, "is_update_available", lambda pkg, pre: True)
    run = MagicMock(return_value="job-1")
    monkeypatch.setattr(bg_job_register, "run_command", run)
    client = router_app(update.router)

    resp = client.post("/update/start")

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-1"}
    run.assert_called_once()
    assert run.call_args[0][0] == "update_reachy_mini"


def test_start_no_update(monkeypatch, router_app):
    """start refuses when no update is available."""
    monkeypatch.setattr(update, "is_update_available", lambda pkg, pre: False)
    run = MagicMock()
    monkeypatch.setattr(bg_job_register, "run_command", run)
    client = router_app(update.router)

    resp = client.post("/update/start")

    assert resp.status_code == 400
    assert resp.json()["detail"] == "No update available"
    run.assert_not_called()


def test_start_busy(monkeypatch, router_app):
    """start refuses while busy_lock is held."""
    run = MagicMock()
    monkeypatch.setattr(bg_job_register, "run_command", run)
    update.busy_lock.acquire()
    client = router_app(update.router)

    resp = client.post("/update/start")

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Update already in progress"
    run.assert_not_called()


def test_start_from_ref(monkeypatch, router_app):
    """start-from-ref schedules the job (no pip) and returns its id."""
    run = MagicMock(return_value="job-2")
    monkeypatch.setattr(bg_job_register, "run_command", run)
    client = router_app(update.router)

    resp = client.post("/update/start-from-ref", params={"git_ref": "v2.0.0"})

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-2"}
    run.assert_called_once()


def test_start_from_ref_empty(monkeypatch, router_app):
    """start-from-ref refuses a blank ref."""
    run = MagicMock()
    monkeypatch.setattr(bg_job_register, "run_command", run)
    client = router_app(update.router)

    resp = client.post("/update/start-from-ref", params={"git_ref": " "})

    assert resp.status_code == 400
    run.assert_not_called()


def test_start_from_ref_busy(monkeypatch, router_app):
    """start-from-ref refuses while busy_lock is held."""
    run = MagicMock()
    monkeypatch.setattr(bg_job_register, "run_command", run)
    update.busy_lock.acquire()
    client = router_app(update.router)

    resp = client.post("/update/start-from-ref", params={"git_ref": "v2.0.0"})

    assert resp.status_code == 400
    run.assert_not_called()
