"""Unit tests for the JSON-RPC envelope + daemon relay routing.

Covers the three things the relay must get right:
* ``apps.*`` handled locally against the AppManager,
* other namespaces relayed to the app with the caller's ``id`` restored,
* app notifications (no ``id``) fanned out to every client.
"""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reachy_mini.daemon import jsonrpc_relay
from reachy_mini.daemon.jsonrpc_relay import JsonRpcRelay
from reachy_mini.io.jsonrpc import (
    JsonRpcError,
    RpcErrorResponse,
    RpcRequest,
    RpcSuccess,
    looks_like_jsonrpc,
    make_error,
    make_notification,
    make_result,
    parse_inbound,
    parse_request,
)


# ---------------------------------------------------------------- envelope


def test_looks_like_jsonrpc() -> None:
    assert looks_like_jsonrpc({"jsonrpc": "2.0", "method": "x"})
    assert not looks_like_jsonrpc({"type": "set_target"})
    assert not looks_like_jsonrpc("not a dict")


def test_parse_request_notification_vs_call() -> None:
    call = parse_request('{"jsonrpc":"2.0","id":"1","method":"conversation.say"}')
    assert not call.is_notification
    assert call.namespace == "conversation"
    note = parse_request('{"jsonrpc":"2.0","method":"conversation.turn"}')
    assert note.is_notification


def test_error_reason_lives_in_data() -> None:
    # The deliberate spec deviation: the string code the UI branches on is in
    # error.data.reason, error.code stays an int.
    err = make_error(None, message="busy", reason="robot_busy", code=-32000)
    assert err["error"]["code"] == -32000
    assert err["error"]["data"]["reason"] == "robot_busy"


def test_builders_emit_the_expected_wire_shape() -> None:
    # The dict builders are model-backed but must keep their exact JSON shape
    # (including id:null on responses, which the spec requires present).
    assert make_result("1", {"ok": True}) == {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {"ok": True},
    }
    assert make_result(None, 7) == {"jsonrpc": "2.0", "id": None, "result": 7}
    assert make_notification("conversation.turn", {"state": "speaking"}) == {
        "jsonrpc": "2.0",
        "method": "conversation.turn",
        "params": {"state": "speaking"},
    }


def test_parse_inbound_classifies_each_frame() -> None:
    # The TypeAdapter union disambiguates on required fields (result / error /
    # method) instead of hand-rolled dict.get sniffing.
    assert isinstance(
        parse_inbound('{"jsonrpc":"2.0","id":"1","result":{"ok":true}}'), RpcSuccess
    )
    err = parse_inbound(
        '{"jsonrpc":"2.0","id":"1","error":{"code":-32000,"message":"x",'
        '"data":{"reason":"busy"}}}'
    )
    assert isinstance(err, RpcErrorResponse)
    assert err.error.data["reason"] == "busy"
    call = parse_inbound('{"jsonrpc":"2.0","id":"1","method":"conversation.say"}')
    assert isinstance(call, RpcRequest) and not call.is_notification
    note = parse_inbound('{"jsonrpc":"2.0","method":"conversation.turn"}')
    assert isinstance(note, RpcRequest) and note.is_notification


def test_parse_inbound_rejects_malformed() -> None:
    with pytest.raises(JsonRpcError):
        parse_inbound("not json at all")
    with pytest.raises(JsonRpcError):
        parse_inbound('{"jsonrpc":"2.0"}')  # neither result/error nor method


# ------------------------------------------------------------------ fakes


class FakeAppManager:
    """Minimal AppManager stand-in for relay routing tests."""

    def __init__(self, url: str | None = "http://0.0.0.0:7860/") -> None:
        self._url = url
        self.current_app: Any = None
        self.started: list[str] = []
        self.stopped = 0
        self.installed: list[Any] = ["conv"]
        self.wireless_version = True
        self.desktop_app_daemon = False
        self.updated: list[str] = []
        # Updates run in the job thread (own event loop): gate with a thread
        # event, not an asyncio one bound to the test's loop.
        self.update_gate: threading.Event | None = None
        self.update_error: str | None = None

    def is_app_running(self) -> bool:
        return self.current_app is not None

    def get_running_app_url(self) -> str | None:
        return self._url

    async def start_app(self, name: str, keep_remote: bool = False) -> Any:
        self.started.append(name)
        self.last_keep_remote = keep_remote
        self.current_app = SimpleNamespace(
            status=SimpleNamespace(
                state="running",
                error=None,
                info=SimpleNamespace(name=name, description="", url=None),
            )
        )
        return self.current_app.status

    async def stop_current_app(self) -> None:
        self.stopped += 1
        self.current_app = None

    async def list_available_apps(self, kind: Any) -> list[Any]:
        return [SimpleNamespace(name=n) for n in self.installed]

    async def update_app(self, name: str, logger: Any) -> None:
        if self.is_app_running() and self.current_app.status.info.name == name:
            raise RuntimeError("Cannot update while it is running")
        if self.update_gate is not None:
            await asyncio.to_thread(self.update_gate.wait, 5)
        if self.update_error:
            raise RuntimeError(self.update_error)
        self.updated.append(name)


class FakeAppWs:
    """Fake app /rpc connection: records sends, streams injected frames."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self._incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.closed = False

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def close(self) -> None:
        self.closed = True
        await self._incoming.put(None)  # end the reader

    def inject(self, obj: dict[str, Any]) -> None:
        self._incoming.put_nowait(json.dumps(obj))

    def __aiter__(self) -> "FakeAppWs":
        return self

    async def __anext__(self) -> str:
        item = await self._incoming.get()
        if item is None:
            raise StopAsyncIteration
        return item


# ------------------------------------------------------------- apps.* local


@pytest.mark.asyncio
async def test_apps_status_idle() -> None:
    relay = JsonRpcRelay(FakeAppManager(), broadcast=lambda _: None)
    replies: list[dict[str, Any]] = []
    await relay.handle(
        '{"jsonrpc":"2.0","id":"1","method":"apps.status"}', replies.append
    )
    assert replies[0]["id"] == "1"
    assert replies[0]["result"]["state"] == "idle"


@pytest.mark.asyncio
async def test_apps_start_and_stop() -> None:
    apps = FakeAppManager()
    relay = JsonRpcRelay(apps, broadcast=lambda _: None)
    replies: list[dict[str, Any]] = []

    await relay.handle(
        '{"jsonrpc":"2.0","id":"1","method":"apps.start","params":{"name":"conv"}}',
        replies.append,
    )
    assert apps.started == ["conv"]
    assert apps.last_keep_remote is True
    assert replies[-1]["result"]["state"] == "running"
    assert replies[-1]["result"]["info"]["name"] == "conv"

    await relay.handle(
        '{"jsonrpc":"2.0","id":"2","method":"apps.stop"}', replies.append
    )
    assert apps.stopped == 1
    assert replies[-1]["result"] == {"stopped": True}


@pytest.mark.asyncio
async def test_apps_start_requires_name() -> None:
    relay = JsonRpcRelay(FakeAppManager(), broadcast=lambda _: None)
    replies: list[dict[str, Any]] = []
    await relay.handle(
        '{"jsonrpc":"2.0","id":"1","method":"apps.start"}', replies.append
    )
    assert replies[0]["error"]["data"]["reason"] == "invalid_params"


# ------------------------------------------------------------- relay to app


@pytest.mark.asyncio
async def test_relay_forwards_and_restores_id(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeAppWs()

    async def fake_connect(url: str) -> FakeAppWs:
        return ws

    monkeypatch.setattr(jsonrpc_relay, "connect", fake_connect)

    broadcasted: list[str] = []
    relay = JsonRpcRelay(FakeAppManager(), broadcast=broadcasted.append)
    replies: list[dict[str, Any]] = []

    await relay.handle(
        '{"jsonrpc":"2.0","id":"caller-9","method":"conversation.say",'
        '"params":{"text":"hi"}}',
        replies.append,
    )

    # The frame reached the app with a relay-assigned id (not the caller's).
    assert len(ws.sent) == 1
    sent = json.loads(ws.sent[0])
    assert sent["method"] == "conversation.say"
    assert sent["id"] != "caller-9"
    relay_id = sent["id"]

    # App answers on the relay id; the reader restores the caller's id.
    ws.inject({"jsonrpc": "2.0", "id": relay_id, "result": {"accepted": True}})
    await asyncio.sleep(0.05)
    assert replies == [
        {"jsonrpc": "2.0", "id": "caller-9", "result": {"accepted": True}}
    ]

    # A notification (no id) from the app is broadcast to all clients.
    ws.inject(
        {
            "jsonrpc": "2.0",
            "method": "conversation.turn",
            "params": {"state": "speaking"},
        }
    )
    await asyncio.sleep(0.05)
    assert len(broadcasted) == 1
    assert json.loads(broadcasted[0])["method"] == "conversation.turn"

    await relay.aclose()


@pytest.mark.asyncio
async def test_relay_no_app_running() -> None:
    relay = JsonRpcRelay(FakeAppManager(url=None), broadcast=lambda _: None)
    replies: list[dict[str, Any]] = []
    await relay.handle(
        '{"jsonrpc":"2.0","id":"1","method":"conversation.say","params":{"text":"x"}}',
        replies.append,
    )
    assert replies[0]["error"]["data"]["reason"] == "not_running"


@pytest.mark.asyncio
async def test_apps_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """apps.install runs the install-if-missing helper and reports the result."""
    calls: list[str] = []

    async def fake_ensure(apps: Any, name: str) -> bool:
        calls.append(name)
        return name == "conv"

    monkeypatch.setattr(jsonrpc_relay, "ensure_startup_app_installed", fake_ensure)
    relay = JsonRpcRelay(FakeAppManager(), broadcast=lambda _: None)
    replies: list[dict[str, Any]] = []

    await relay.handle(
        '{"jsonrpc":"2.0","id":"1","method":"apps.install","params":{"name":"conv"}}',
        replies.append,
    )
    assert calls == ["conv"]
    assert replies[-1]["result"] == {"installed": True}

    await relay.handle(
        '{"jsonrpc":"2.0","id":"2","method":"apps.install","params":{"name":"nope"}}',
        replies.append,
    )
    assert replies[-1]["error"]["data"]["reason"] == "install_failed"

    await relay.handle(
        '{"jsonrpc":"2.0","id":"3","method":"apps.install"}', replies.append
    )
    assert replies[-1]["error"]["data"]["reason"] == "invalid_params"


@pytest.mark.asyncio
async def test_original_id_recorded_before_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller's id must be recorded before the frame leaves the relay.

    Regression: the reader task runs on this loop and can deliver the app's
    reply *during* ``ws.send``. If the original-id mapping is only written
    after the await, that reply restores id=None and the caller hangs. Assert
    the mapping is already present at send time.
    """
    relay = JsonRpcRelay(FakeAppManager(), broadcast=lambda _: None)
    snapshot: dict[str, bool] = {}

    class RacingWs(FakeAppWs):
        async def send(self, text: str) -> None:
            relay_id = json.loads(text)["id"]
            snapshot["present"] = relay_id in relay._pending_original_id
            await super().send(text)

    ws = RacingWs()

    async def fake_connect(url: str) -> RacingWs:
        return ws

    monkeypatch.setattr(jsonrpc_relay, "connect", fake_connect)

    await relay.handle(
        '{"jsonrpc":"2.0","id":"caller-1","method":"conversation.say"}',
        lambda _r: None,
    )
    assert snapshot["present"] is True
    await relay.aclose()


# ------------------------------------------------------- apps.* management


class FakeManagerWithInstalled(FakeAppManager):
    """FakeAppManager plus an installed list and job coroutines."""

    def __init__(self) -> None:
        super().__init__()
        self.installed = [
            jsonrpc_relay.AppInfo(
                name="conv",
                source_kind=jsonrpc_relay.SourceKind.INSTALLED,
                url="https://huggingface.co/spaces/pollen/conv",
                extra={
                    "id": "pollen/conv",
                    "cardData": {
                        "title": "Conv",
                        "emoji": "🎤",
                        "short_description": "Talk",
                    },
                    "custom_app_url": "http://0.0.0.0:7860/",
                    "siblings": ["huge", "payload"],
                },
            )
        ]
        self.restarted_keep_remote: bool | None = None

    async def list_available_apps(self, source: Any) -> list[Any]:
        return self.installed

    async def restart_current_app(self, keep_remote: bool = False) -> Any:
        if self.current_app is None:
            raise RuntimeError("No app is currently running")
        self.restarted_keep_remote = keep_remote
        return self.current_app.status

    async def install_new_app(self, app: Any, logger: Any) -> None:
        logger.info(f"installing {app.name}")

    async def remove_app(self, name: str, logger: Any) -> None:
        raise RuntimeError("uninstall failed")


async def _call(
    relay: JsonRpcRelay, method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    replies: list[dict[str, Any]] = []
    frame: dict[str, Any] = {"jsonrpc": "2.0", "id": "x", "method": method}
    if params is not None:
        frame["params"] = params
    await relay.handle(json.dumps(frame), replies.append)
    return replies[0]


@pytest.mark.asyncio
async def test_apps_list_installed_is_trimmed() -> None:
    relay = JsonRpcRelay(FakeManagerWithInstalled(), broadcast=lambda _: None)
    reply = await _call(relay, "apps.list_installed")
    assert reply["result"]["apps"] == [
        {
            "name": "conv",
            "description": "Talk",
            "url": "https://huggingface.co/spaces/pollen/conv",
            "space_id": "pollen/conv",
            "title": "Conv",
            "emoji": "🎤",
            "private": False,
            "custom_app_url": "http://0.0.0.0:7860/",
        }
    ]


@pytest.mark.asyncio
async def test_apps_stop_and_restart_when_idle_report_not_running() -> None:
    relay = JsonRpcRelay(FakeManagerWithInstalled(), broadcast=lambda _: None)

    async def stop_raises() -> None:
        raise RuntimeError("No app is currently running")

    relay._apps.stop_current_app = stop_raises  # type: ignore[method-assign]
    assert (await _call(relay, "apps.stop"))["error"]["data"]["reason"] == "not_running"
    assert (await _call(relay, "apps.restart"))["error"]["data"][
        "reason"
    ] == "not_running"


@pytest.mark.asyncio
async def test_apps_restart_keeps_remote() -> None:
    apps = FakeManagerWithInstalled()
    relay = JsonRpcRelay(apps, broadcast=lambda _: None)
    await _call(relay, "apps.start", {"name": "conv"})
    reply = await _call(relay, "apps.restart")
    assert reply["result"]["info"]["name"] == "conv"
    assert apps.restarted_keep_remote is True


async def _wait_final(events: list[dict[str, Any]], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not any(e["params"]["final"] for e in events):
        assert asyncio.get_running_loop().time() < deadline, "job never finished"
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_apps_install_space_runs_a_job_and_streams_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_resolve(space_id: str) -> Any:
        if space_id != "pollen/conv":
            return None
        return jsonrpc_relay.AppInfo(
            name="conv", source_kind=jsonrpc_relay.SourceKind.HF_SPACE
        )

    monkeypatch.setattr(jsonrpc_relay, "get_python_app_info", fake_resolve)
    events: list[dict[str, Any]] = []
    relay = JsonRpcRelay(
        FakeManagerWithInstalled(), broadcast=lambda s: events.append(json.loads(s))
    )

    reply = await _call(relay, "apps.install_space", {"space_id": "pollen/conv"})
    job_id = reply["result"]["job_id"]
    assert reply["result"]["name"] == "conv"
    await _wait_final(events)

    assert all(e["method"] == "apps.job_progress" for e in events)
    assert all(e["params"]["job_id"] == job_id for e in events)
    assert any(e["params"]["line"] == "installing conv" for e in events)
    assert events[-1]["params"]["final"] is True
    assert events[-1]["params"]["status"] == "done"

    status = await _call(relay, "apps.job_status", {"job_id": job_id})
    assert status["result"]["status"] == "done"
    assert "installing conv" in status["result"]["logs"]

    assert (await _call(relay, "apps.install_space", {"space_id": "pollen/nope"}))[
        "error"
    ]["data"]["reason"] == "not_found"
    assert (await _call(relay, "apps.install_space", {"space_id": "bad"}))["error"][
        "data"
    ]["reason"] == "invalid_params"
    assert (await _call(relay, "apps.job_status", {"job_id": "nope"}))["error"]["data"][
        "reason"
    ] == "unknown_job"


@pytest.mark.asyncio
async def test_apps_remove_guards_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    apps = FakeManagerWithInstalled()
    events: list[dict[str, Any]] = []
    relay = JsonRpcRelay(apps, broadcast=lambda s: events.append(json.loads(s)))

    # Refused while running.
    await _call(relay, "apps.start", {"name": "conv"})
    reply = await _call(relay, "apps.remove", {"name": "conv"})
    assert reply["error"]["data"]["reason"] == "app_running"
    apps.current_app = None

    # Refused for the startup app.
    monkeypatch.setattr(
        jsonrpc_relay.startup_app_config, "get_startup_app", lambda: "conv"
    )
    reply = await _call(relay, "apps.remove", {"name": "conv"})
    assert reply["error"]["data"]["reason"] == "startup_app"
    monkeypatch.setattr(
        jsonrpc_relay.startup_app_config, "get_startup_app", lambda: None
    )

    # Busy while another job runs.
    monkeypatch.setattr(jsonrpc_relay.bg_job_register, "has_active_job", lambda: True)
    reply = await _call(relay, "apps.remove", {"name": "conv"})
    assert reply["error"]["data"]["reason"] == "busy"
    monkeypatch.undo()
    monkeypatch.setattr(
        jsonrpc_relay.startup_app_config, "get_startup_app", lambda: None
    )

    # A failing job ends with a final `failed` event.
    reply = await _call(relay, "apps.remove", {"name": "conv"})
    assert "job_id" in reply["result"]
    await _wait_final(events)
    assert events[-1]["params"]["status"] == "failed"


@pytest.mark.asyncio
async def test_apps_remove_clears_a_crashed_app_first() -> None:
    apps = FakeManagerWithInstalled()
    events: list[dict[str, Any]] = []
    relay = JsonRpcRelay(apps, broadcast=lambda s: events.append(json.loads(s)))
    await _call(relay, "apps.start", {"name": "conv"})
    apps.current_app.status.state = "error"

    reply = await _call(relay, "apps.remove", {"name": "conv"})

    assert "job_id" in reply["result"]
    assert apps.stopped == 1
    await _wait_final(events)


@pytest.mark.asyncio
async def test_apps_restart_start_failure_is_not_reported_as_idle() -> None:
    apps = FakeManagerWithInstalled()
    relay = JsonRpcRelay(apps, broadcast=lambda _: None)
    await _call(relay, "apps.start", {"name": "conv"})

    async def restart_fails(keep_remote: bool = False) -> Any:
        raise RuntimeError("An app is already running")

    apps.restart_current_app = restart_fails  # type: ignore[method-assign]
    reply = await _call(relay, "apps.restart")
    assert reply["error"]["data"]["reason"] == "already_running"


@pytest.mark.asyncio
async def test_inline_install_blocks_install_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()

    async def slow_ensure(apps: Any, name: str) -> bool:
        await release.wait()
        return True

    monkeypatch.setattr(jsonrpc_relay, "ensure_startup_app_installed", slow_ensure)
    relay = JsonRpcRelay(FakeManagerWithInstalled(), broadcast=lambda _: None)

    legacy = asyncio.create_task(_call(relay, "apps.install", {"name": "conv"}))
    await asyncio.sleep(0)
    reply = await _call(relay, "apps.install_space", {"space_id": "pollen/other"})
    assert reply["error"]["data"]["reason"] == "busy"

    release.set()
    assert (await legacy)["result"] == {"installed": True}


async def _race_crashed_app_clear_with_install(
    monkeypatch: pytest.MonkeyPatch, verb: str
) -> dict[str, Any]:
    """Run `verb` on a crashed app while another client installs meanwhile."""
    apps = FakeManagerWithInstalled()
    relay = JsonRpcRelay(apps, broadcast=lambda _: None)
    await _call(relay, "apps.start", {"name": "conv"})
    apps.current_app.status.state = "error"
    stop_entered, release_stop = asyncio.Event(), asyncio.Event()
    finish_install = threading.Event()

    async def slow_stop() -> None:
        stop_entered.set()
        await release_stop.wait()
        apps.current_app = None

    async def slow_install(app: Any, logger: Any) -> None:
        await asyncio.to_thread(finish_install.wait, 5)

    async def resolve(space_id: str) -> Any:
        return jsonrpc_relay.AppInfo(
            name="other", source_kind=jsonrpc_relay.SourceKind.HF_SPACE
        )

    finish_first = threading.Event()

    async def slow_job(name: str, logger: Any) -> None:
        # Keep the first job running while the queued install is checked.
        await asyncio.to_thread(finish_first.wait, 5)

    apps.stop_current_app = slow_stop  # type: ignore[method-assign]
    apps.install_new_app = slow_install  # type: ignore[method-assign]
    apps.remove_app = slow_job  # type: ignore[method-assign]
    apps.update_app = slow_job  # type: ignore[method-assign]
    monkeypatch.setattr(jsonrpc_relay, "get_python_app_info", resolve)
    monkeypatch.setattr(
        jsonrpc_relay.startup_app_config, "get_startup_app", lambda: None
    )

    first = asyncio.create_task(_call(relay, f"apps.{verb}", {"name": "conv"}))
    await stop_entered.wait()
    # Another client's install arrives while the crashed app is being
    # cleared: job starts are serialized, so it waits instead of racing.
    install = asyncio.create_task(
        _call(relay, "apps.install_space", {"space_id": "pollen/other"})
    )
    await asyncio.sleep(0.05)
    assert not install.done()
    release_stop.set()
    reply = await first
    queued = await install
    finish_first.set()
    finish_install.set()
    # Exactly one job got the venv: the queued install is refused busy.
    assert queued["error"]["data"]["reason"] == "busy"
    return reply


@pytest.mark.asyncio
async def test_remove_of_crashed_app_races_an_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clearing a crashed app awaits: an install arriving meanwhile must not race."""
    reply = await _race_crashed_app_clear_with_install(monkeypatch, "remove")
    assert "job_id" in reply["result"]


@pytest.mark.asyncio
async def test_update_of_crashed_app_races_an_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same race as remove, through apps.update."""
    reply = await _race_crashed_app_clear_with_install(monkeypatch, "update")
    assert reply["result"]["started"] is True


@pytest.mark.asyncio
async def test_install_space_refuses_a_running_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reinstalling must not swap files under the app's live process."""

    async def resolve(space_id: str) -> Any:
        return jsonrpc_relay.AppInfo(
            name="conv", source_kind=jsonrpc_relay.SourceKind.HF_SPACE
        )

    monkeypatch.setattr(jsonrpc_relay, "get_python_app_info", resolve)
    apps = FakeManagerWithInstalled()
    relay = JsonRpcRelay(apps, broadcast=lambda _: None)
    await _call(relay, "apps.start", {"name": "conv"})

    reply = await _call(relay, "apps.install_space", {"space_id": "pollen/conv"})
    assert reply["error"]["data"]["reason"] == "app_running"

    # A crashed copy is cleared instead, and the reinstall goes ahead.
    apps.current_app.status.state = "error"
    reply = await _call(relay, "apps.install_space", {"space_id": "pollen/conv"})
    assert "job_id" in reply["result"]
    assert apps.stopped == 1


# ------------------------------------------- apps.check_update / apps.update


async def _wait_job_done(job_id: str, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while jsonrpc_relay._job_is_active(job_id):
        assert asyncio.get_running_loop().time() < deadline, "job never finished"
        await asyncio.sleep(0.01)


def _fake_dist(root: Path, dist: str, version: str, entry_point: str | None) -> None:
    info = root / f"{dist}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: {dist}\nVersion: {version}\n")
    if entry_point:
        (info / "entry_points.txt").write_text(
            f"[reachy_mini_apps]\n{entry_point} = pkg.main:App\n"
        )


@pytest.mark.asyncio
async def test_apps_check_update(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """apps.check_update reports the version and HF check of an installed app only."""
    checked: list[str] = []

    async def fake_check(app: Any, **kwargs: Any) -> Any:
        checked.append(app.name)
        return SimpleNamespace(update_available=True)

    monkeypatch.setattr(
        jsonrpc_relay.app_update_checker, "check_app_update", fake_check
    )
    _fake_dist(tmp_path, "conv", "1.1.0", None)
    monkeypatch.setattr(
        jsonrpc_relay.local_common_venv, "get_app_site_packages", lambda *_: tmp_path
    )
    relay = JsonRpcRelay(FakeAppManager(), broadcast=lambda _: None)

    reply = await _call(relay, "apps.check_update", {"name": "conv"})
    assert reply["result"] == {
        "installed": True,
        "version": "1.1.0",
        "update_available": True,
        "checkable": True,
        "updating": False,
        "error": None,
    }
    reply = await _call(relay, "apps.check_update", {"name": "other"})
    assert reply["result"]["installed"] is False
    assert reply["result"]["update_available"] is False
    assert checked == ["conv"]
    reply = await _call(relay, "apps.check_update", {})
    assert reply["error"]["data"]["reason"] == "invalid_params"


@pytest.mark.asyncio
async def test_apps_check_update_finds_version_through_entry_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The app name is the entry-point name; the distribution may be named otherwise."""

    async def unknown(app: Any, **kwargs: Any) -> Any:
        return None  # the source can't be checked

    monkeypatch.setattr(jsonrpc_relay.app_update_checker, "check_app_update", unknown)
    _fake_dist(tmp_path, "reachy_mini_conversation_app", "2.3.0", "conv")
    monkeypatch.setattr(
        jsonrpc_relay.local_common_venv, "get_app_site_packages", lambda *_: tmp_path
    )
    relay = JsonRpcRelay(FakeAppManager(), broadcast=lambda _: None)

    result = (await _call(relay, "apps.check_update", {"name": "conv"}))["result"]
    assert result["version"] == "2.3.0"
    # Not checkable is not "up to date".
    assert result["update_available"] is False
    assert result["checkable"] is False


@pytest.mark.asyncio
async def test_apps_update_stops_the_app_and_runs_as_a_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """apps.update returns at once with a job; check_update follows it."""

    async def no_update(app: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(update_available=False)

    monkeypatch.setattr(jsonrpc_relay.app_update_checker, "check_app_update", no_update)
    events: list[dict[str, Any]] = []
    apps = FakeAppManager()
    apps.update_gate = threading.Event()
    relay = JsonRpcRelay(apps, broadcast=lambda s: events.append(json.loads(s)))
    await _call(relay, "apps.start", {"name": "conv"})

    reply = (await _call(relay, "apps.update", {"name": "conv"}))["result"]
    assert reply["started"] is True
    job_id = reply["job_id"]
    assert apps.stopped == 1  # the manager refuses to update a running app
    again = (await _call(relay, "apps.update", {"name": "conv"}))["result"]
    assert again == {"started": False, "job_id": job_id}
    check = (await _call(relay, "apps.check_update", {"name": "conv"}))["result"]
    assert check["updating"] is True

    apps.update_gate.set()
    await _wait_job_done(job_id)
    result = (await _call(relay, "apps.check_update", {"name": "conv"}))["result"]
    assert result["updating"] is False
    assert result["error"] is None
    assert apps.updated == ["conv"]
    # Progress went out like any app job.
    assert any(e["params"]["job_id"] == job_id for e in events)


@pytest.mark.asyncio
async def test_apps_update_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed background update surfaces through apps.check_update."""

    async def no_update(app: Any, **kwargs: Any) -> Any:
        return None

    monkeypatch.setattr(jsonrpc_relay.app_update_checker, "check_app_update", no_update)
    apps = FakeAppManager()
    apps.update_error = "space not found"
    relay = JsonRpcRelay(apps, broadcast=lambda _: None)

    job_id = (await _call(relay, "apps.update", {"name": "conv"}))["result"]["job_id"]
    await _wait_job_done(job_id)
    result = (await _call(relay, "apps.check_update", {"name": "conv"}))["result"]
    assert result["error"] == "space not found"
    assert result["update_available"] is False


@pytest.mark.asyncio
async def test_concurrent_updates_start_only_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A double tap (two frames dispatched concurrently) must not run two updates."""
    apps = FakeAppManager()
    apps.update_gate = threading.Event()
    relay = JsonRpcRelay(apps, broadcast=lambda _: None)
    await _call(relay, "apps.start", {"name": "conv"})

    async def slow_stop() -> None:
        # Like the real manager: an app being stopped still holds the slot
        # (is_app_running), and a second stop raises.
        status = apps.current_app.status
        if status.state == "stopping":
            raise RuntimeError("No app is currently running")
        status.state = "stopping"
        await asyncio.sleep(0.05)
        apps.stopped += 1
        apps.current_app = None

    apps.stop_current_app = slow_stop  # type: ignore[method-assign]
    first, second = await asyncio.gather(
        _call(relay, "apps.update", {"name": "conv"}),
        _call(relay, "apps.update", {"name": "conv"}),
    )
    # One update runs; the second tap is told so and gets the same job.
    assert first["result"]["started"] is True
    assert second["result"] == {"started": False, "job_id": first["result"]["job_id"]}
    apps.update_gate.set()
    await _wait_job_done(first["result"]["job_id"])
    assert apps.updated == ["conv"]
    assert apps.stopped == 1


@pytest.mark.asyncio
async def test_apps_start_refuses_an_app_being_updated() -> None:
    """Starting an app mid-reinstall would run half-written files."""
    apps = FakeAppManager()
    apps.update_gate = threading.Event()
    relay = JsonRpcRelay(apps, broadcast=lambda _: None)

    job_id = (await _call(relay, "apps.update", {"name": "conv"}))["result"]["job_id"]
    reply = await _call(relay, "apps.start", {"name": "conv"})
    assert reply["error"]["data"]["reason"] == "busy"
    assert apps.started == []

    apps.update_gate.set()
    await _wait_job_done(job_id)
    assert "result" in await _call(relay, "apps.start", {"name": "conv"})


@pytest.mark.asyncio
async def test_relay_jobs_invalidate_the_rest_update_cache() -> None:
    """The dashboard's /check-updates must not show a stale result after an RPC job."""
    from reachy_mini.daemon.app.routers import apps as apps_router

    apps = FakeAppManager()
    relay = JsonRpcRelay(apps, broadcast=lambda _: None)
    apps_router._update_cache = ("stale", "stale")  # type: ignore[assignment]

    job_id = (await _call(relay, "apps.update", {"name": "conv"}))["result"]["job_id"]
    assert apps_router._update_cache is None
    await _wait_job_done(job_id)


# -------------------------------------------------------------- nfc.* local


class FakeNfcReader:
    """Stands in for ``NfcReader``: no serial port, canned answers."""

    def __init__(self, enabled: bool = True) -> None:
        from reachy_mini.nfc import NfcStatus, NfcTag

        self.enabled = enabled
        self.status = NfcStatus(
            connected=True, enabled=enabled, chip_detected=True, driver_available=True
        )
        self.tag = NfcTag(present=True, uid="04A1B2C3", content="badge42")
        self.writes: list[Any] = []
        self.erases: list[Any] = []

    def get_status(self) -> Any:
        return self.status.model_copy(update={"enabled": self.enabled})

    def get_tag(self) -> Any:
        return self.tag

    def is_enabled(self) -> bool:
        return self.enabled

    def start(self) -> None:
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False

    def write(self, request: Any) -> Any:
        from reachy_mini.nfc import NfcWriteResult

        self.writes.append(request)
        return NfcWriteResult(success=True)

    def erase(self, request: Any) -> Any:
        from reachy_mini.nfc import NfcWriteResult

        self.erases.append(request)
        return NfcWriteResult(success=False, error="NO_TAG")


@pytest.mark.asyncio
async def test_nfc_status_and_tag() -> None:
    relay = JsonRpcRelay(
        FakeAppManager(), broadcast=lambda _: None, nfc_reader=FakeNfcReader()
    )  # type: ignore[arg-type]
    status = await _call(relay, "nfc.status")
    assert status["result"]["connected"] is True
    tag = await _call(relay, "nfc.tag")
    assert tag["result"]["uid"] == "04A1B2C3"


@pytest.mark.asyncio
async def test_nfc_without_reader_answers_like_rest() -> None:
    relay = JsonRpcRelay(FakeAppManager(), broadcast=lambda _: None)
    status = await _call(relay, "nfc.status")
    assert status["result"]["enabled"] is False
    assert status["result"]["connected"] is False
    tag = await _call(relay, "nfc.tag")
    assert tag["result"]["present"] is False
    write = await _call(relay, "nfc.write", {"text": "hi"})
    assert write["error"]["data"]["reason"] == "nfc_disabled"


@pytest.mark.asyncio
async def test_nfc_write_and_erase() -> None:
    reader = FakeNfcReader()
    relay = JsonRpcRelay(FakeAppManager(), broadcast=lambda _: None, nfc_reader=reader)  # type: ignore[arg-type]
    write = await _call(relay, "nfc.write", {"uri": "https://huggingface.co"})
    assert write["result"]["success"] is True
    assert reader.writes[0].uri == "https://huggingface.co"
    erase = await _call(relay, "nfc.erase", {"full": True})
    assert erase["result"]["error"] == "NO_TAG"
    assert reader.erases[0].full is True


@pytest.mark.asyncio
async def test_nfc_write_rejects_bad_params() -> None:
    relay = JsonRpcRelay(
        FakeAppManager(), broadcast=lambda _: None, nfc_reader=FakeNfcReader()
    )  # type: ignore[arg-type]
    reply = await _call(relay, "nfc.write", {"text": "a", "uri": "b"})
    assert reply["error"]["data"]["reason"] == "invalid_params"


@pytest.mark.asyncio
async def test_nfc_set_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    saved: list[bool] = []
    monkeypatch.setattr("reachy_mini.nfc.set_nfc_enabled", saved.append)
    reader = FakeNfcReader()
    relay = JsonRpcRelay(FakeAppManager(), broadcast=lambda _: None, nfc_reader=reader)  # type: ignore[arg-type]
    reply = await _call(relay, "nfc.set_enabled", {"enabled": False})
    assert saved == [False]
    assert reply["result"]["enabled"] is False
    # Off, tag operations are refused like the REST routes' 503.
    write = await _call(relay, "nfc.write", {"text": "hi"})
    assert write["error"]["data"]["reason"] == "nfc_disabled"
