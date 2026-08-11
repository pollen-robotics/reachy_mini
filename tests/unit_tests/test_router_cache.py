"""Unit tests for the cache management router (clear-hf, reset-apps)."""

from unittest.mock import MagicMock

import pytest

from reachy_mini.daemon.app.routers import cache


def _patch(monkeypatch, *, exists: bool, rmtree=None):
    """Steer cache.Path existence and stub cache.shutil.rmtree."""
    monkeypatch.setattr(cache, "Path", lambda _p: MagicMock(exists=lambda: exists))
    rmtree = rmtree or MagicMock()
    monkeypatch.setattr(cache.shutil, "rmtree", rmtree)
    return rmtree


ENDPOINTS = [
    ("/cache/clear-hf", "HuggingFace cache cleared", "Cache directory already empty"),
    (
        "/cache/reset-apps",
        "Applications virtual environment removed",
        "Virtual environment directory already empty",
    ),
]


@pytest.mark.parametrize("route,msg_removed,_msg_empty", ENDPOINTS)
def test_path_exists_rmtree_called(monkeypatch, router_app, route, msg_removed, _msg_empty):
    """Existing path -> rmtree runs, 200 with success message."""
    rmtree = _patch(monkeypatch, exists=True)
    client = router_app(cache.router)

    resp = client.post(route)

    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "message": msg_removed}
    rmtree.assert_called_once()


@pytest.mark.parametrize("route,_msg_removed,msg_empty", ENDPOINTS)
def test_path_missing_no_rmtree(monkeypatch, router_app, route, _msg_removed, msg_empty):
    """Missing path -> 200, rmtree not called."""
    rmtree = _patch(monkeypatch, exists=False)
    client = router_app(cache.router)

    resp = client.post(route)

    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "message": msg_empty}
    rmtree.assert_not_called()


@pytest.mark.parametrize("route,_msg_removed,_msg_empty", ENDPOINTS)
def test_rmtree_oserror_returns_500(monkeypatch, router_app, route, _msg_removed, _msg_empty):
    """rmtree raising OSError -> 500 with failure detail."""
    rmtree = _patch(
        monkeypatch, exists=True, rmtree=MagicMock(side_effect=OSError("boom"))
    )
    client = router_app(cache.router)

    resp = client.post(route)

    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]
    rmtree.assert_called_once()
