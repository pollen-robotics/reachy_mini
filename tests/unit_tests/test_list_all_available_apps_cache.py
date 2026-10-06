"""Tests for HF catalog TTL cache and single-flight in AppManager."""

from __future__ import annotations

import asyncio

import pytest

from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.apps.manager import HF_CATALOG_TTL_S, AppManager


def _app(name: str, source: SourceKind, app_id: str) -> AppInfo:
    return AppInfo(name=name, source_kind=source, extra={"id": app_id})


@pytest.mark.asyncio
async def test_list_all_available_apps_uses_cached_hf_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second call within the TTL must not hit HF sources again."""
    mgr = AppManager()
    calls = {"hf_space": 0, "dashboard": 0, "local": 0, "installed": 0}

    async def fake_list(source: SourceKind) -> list[AppInfo]:
        if source == SourceKind.HF_SPACE:
            calls["hf_space"] += 1
            await asyncio.sleep(0)
            return [_app("space-a", SourceKind.HF_SPACE, "owner/space-a")]
        if source == SourceKind.DASHBOARD_SELECTION:
            calls["dashboard"] += 1
            await asyncio.sleep(0)
            return [_app("dash-a", SourceKind.DASHBOARD_SELECTION, "owner/dash-a")]
        if source == SourceKind.LOCAL:
            calls["local"] += 1
            return [_app("local-a", SourceKind.LOCAL, "local-a")]
        if source == SourceKind.INSTALLED:
            calls["installed"] += 1
            return [_app("inst-a", SourceKind.INSTALLED, "inst-a")]
        raise AssertionError(f"unexpected source {source}")

    monkeypatch.setattr(mgr, "list_available_apps", fake_list)

    first = await mgr.list_all_available_apps()
    second = await mgr.list_all_available_apps()

    assert [a.name for a in first] == ["dash-a", "space-a", "local-a", "inst-a"]
    assert [a.name for a in second] == ["dash-a", "space-a", "local-a", "inst-a"]
    assert calls["hf_space"] == 1
    assert calls["dashboard"] == 1
    assert calls["local"] == 2
    assert calls["installed"] == 2


@pytest.mark.asyncio
async def test_list_all_available_apps_single_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent callers must share one in-flight HF catalog fetch."""
    mgr = AppManager()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = {"hf_space": 0, "dashboard": 0}

    async def fake_list(source: SourceKind) -> list[AppInfo]:
        if source == SourceKind.HF_SPACE:
            calls["hf_space"] += 1
            started.set()
            await release.wait()
            return [_app("space-a", SourceKind.HF_SPACE, "owner/space-a")]
        if source == SourceKind.DASHBOARD_SELECTION:
            calls["dashboard"] += 1
            return [_app("dash-a", SourceKind.DASHBOARD_SELECTION, "owner/dash-a")]
        if source in (SourceKind.LOCAL, SourceKind.INSTALLED):
            return []
        raise AssertionError(f"unexpected source {source}")

    monkeypatch.setattr(mgr, "list_available_apps", fake_list)

    t1 = asyncio.create_task(mgr.list_all_available_apps())
    await started.wait()
    t2 = asyncio.create_task(mgr.list_all_available_apps())
    # Give the second caller time to join the in-flight task.
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(t1, t2)

    assert calls["hf_space"] == 1
    assert calls["dashboard"] == 1
    assert [a.name for a in results[0]] == ["dash-a", "space-a"]
    assert [a.name for a in results[1]] == ["dash-a", "space-a"]


@pytest.mark.asyncio
async def test_list_all_available_apps_refetches_after_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the TTL expires, HF sources must be fetched again."""
    mgr = AppManager()
    calls = {"hf_space": 0, "dashboard": 0}

    async def fake_list(source: SourceKind) -> list[AppInfo]:
        if source == SourceKind.HF_SPACE:
            calls["hf_space"] += 1
            return [_app("space-a", SourceKind.HF_SPACE, "owner/space-a")]
        if source == SourceKind.DASHBOARD_SELECTION:
            calls["dashboard"] += 1
            return [_app("dash-a", SourceKind.DASHBOARD_SELECTION, "owner/dash-a")]
        if source in (SourceKind.LOCAL, SourceKind.INSTALLED):
            return []
        raise AssertionError(f"unexpected source {source}")

    monkeypatch.setattr(mgr, "list_available_apps", fake_list)

    await mgr.list_all_available_apps()
    assert calls["hf_space"] == 1
    assert calls["dashboard"] == 1

    mgr._hf_catalog_cache_at -= HF_CATALOG_TTL_S + 1.0
    await mgr.list_all_available_apps()

    assert calls["hf_space"] == 2
    assert calls["dashboard"] == 2


@pytest.mark.asyncio
async def test_failed_hf_fetch_does_not_poison_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed HF fetch must not cache; the next call retries."""
    mgr = AppManager()
    calls = {"hf_space": 0}

    async def fake_list(source: SourceKind) -> list[AppInfo]:
        if source == SourceKind.HF_SPACE:
            calls["hf_space"] += 1
            if calls["hf_space"] == 1:
                raise RuntimeError("hf down")
            return [_app("space-a", SourceKind.HF_SPACE, "owner/space-a")]
        if source == SourceKind.DASHBOARD_SELECTION:
            return []
        if source in (SourceKind.LOCAL, SourceKind.INSTALLED):
            return []
        raise AssertionError(f"unexpected source {source}")

    monkeypatch.setattr(mgr, "list_available_apps", fake_list)

    with pytest.raises(RuntimeError, match="hf down"):
        await mgr.list_all_available_apps()

    apps = await mgr.list_all_available_apps()
    assert [a.name for a in apps] == ["space-a"]
    assert calls["hf_space"] == 2


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_shared_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disconnected caller must not cancel another caller's fetch."""
    mgr = AppManager()
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def load() -> list[AppInfo]:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return [_app("shared", SourceKind.HF_SPACE, "me/shared")]

    monkeypatch.setattr(mgr, "_load_hf_catalog_apps", load)
    first = asyncio.create_task(mgr._get_hf_catalog_apps())
    await started.wait()
    second = asyncio.create_task(mgr._get_hf_catalog_apps())
    await asyncio.sleep(0)
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    release.set()
    result = (await asyncio.gather(second, return_exceptions=True))[0]
    assert isinstance(result, list), repr(result)
    assert result[0].name == "shared"
    assert calls == 1
    await mgr.close()


@pytest.mark.asyncio
async def test_abandoned_fetch_is_owned_until_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close awaits fetch cleanup even when every original waiter disconnected."""
    mgr = AppManager()
    started, cleaned = asyncio.Event(), asyncio.Event()

    async def load() -> list[AppInfo]:
        started.set()
        try:
            await asyncio.Event().wait()
            return []
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    monkeypatch.setattr(mgr, "_load_hf_catalog_apps", load)
    waiter = asyncio.create_task(mgr._get_hf_catalog_apps())
    await started.wait()
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    task = mgr._hf_catalog_task
    assert task is not None
    try:
        assert not task.done(), "request cancellation reached manager-owned work"
        await mgr.close()
        assert task.done() and cleaned.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_cancels_catalog_even_if_app_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """App cleanup failures cannot leak shielded catalog tasks."""
    mgr = AppManager()
    started = asyncio.Event()

    async def load() -> list[AppInfo]:
        started.set()
        await asyncio.Event().wait()
        return []

    async def fail_stop() -> None:
        raise RuntimeError("app stop failed")

    monkeypatch.setattr(mgr, "_load_hf_catalog_apps", load)
    monkeypatch.setattr(mgr, "is_app_running", lambda: True)
    monkeypatch.setattr(mgr, "stop_current_app", fail_stop)
    waiter = asyncio.create_task(mgr._get_hf_catalog_apps())
    await started.wait()
    try:
        with pytest.raises(RuntimeError, match="app stop failed"):
            await mgr.close()
        assert mgr._hf_catalog_task is None or mgr._hf_catalog_task.done()
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        if mgr._hf_catalog_task is not None:
            mgr._hf_catalog_task.cancel()
            await asyncio.gather(mgr._hf_catalog_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_abandoned_failure_can_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fetch that fails after its last caller leaves does not poison retries."""
    mgr = AppManager()
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def load() -> list[AppInfo]:
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
            raise ValueError("late fetch failure")
        return [_app("retry", SourceKind.HF_SPACE, "me/retry")]

    monkeypatch.setattr(mgr, "_load_hf_catalog_apps", load)
    waiter = asyncio.create_task(mgr._get_hf_catalog_apps())
    await started.wait()
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    release.set()
    await asyncio.sleep(0)
    assert [a.name for a in await mgr._get_hf_catalog_apps()] == ["retry"]
    await mgr.close()
    with pytest.raises(RuntimeError, match="closed"):
        await mgr._get_hf_catalog_apps()
