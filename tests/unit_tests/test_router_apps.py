"""Unit tests for ownership-sensitive application lifecycle routes."""

from typing import Any

from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.apps.manager import AppState, AppStatus
from reachy_mini.daemon.app.dependencies import get_app_manager
from reachy_mini.daemon.app.routers import apps


class _AppManager:
    def __init__(self, *, error: str | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def start_app(self, name: str, **kwargs: Any) -> AppStatus:
        self.calls.append((name, kwargs))
        if self.error is not None:
            raise RuntimeError(self.error)
        return AppStatus(
            info=AppInfo(name=name, source_kind=SourceKind.INSTALLED),
            state=AppState.STARTING,
        )


def test_no_evict_start_uses_atomic_app_manager_contract(router_app) -> None:
    """The dedicated route passes the non-evicting manager option atomically."""
    manager = _AppManager()
    client = router_app(
        apps.router,
        overrides={get_app_manager: lambda: manager},
    )

    response = client.post("/apps/start-app/reachy_agent/no-evict")

    assert response.status_code == 200
    assert response.json()["state"] == "starting"
    assert manager.calls == [("reachy_agent", {"evict_remote": False})]


def test_no_evict_start_reports_busy_managed_slot(router_app) -> None:
    """A managed-slot conflict remains a visible HTTP 400 response."""
    manager = _AppManager(error="The robot app slot is already in use")
    client = router_app(
        apps.router,
        overrides={get_app_manager: lambda: manager},
    )

    response = client.post("/apps/start-app/reachy_agent/no-evict")

    assert response.status_code == 400
    assert response.json() == {"detail": "The robot app slot is already in use"}
    assert manager.calls == [("reachy_agent", {"evict_remote": False})]
