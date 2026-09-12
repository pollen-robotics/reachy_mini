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
