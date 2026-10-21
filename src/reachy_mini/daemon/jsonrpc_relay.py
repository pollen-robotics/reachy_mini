"""Daemon-side JSON-RPC relay: the one control surface for apps over WebRTC/WS.

Incoming JSON-RPC frames from any client transport (WebRTC DataChannel,
``/ws/sdk``) are routed here by namespace:

* ``apps.*`` — handled locally against the :class:`AppManager`: lifecycle of
  the single running app (start / stop / restart / status) and the installed
  set (list / install / remove / update, as background jobs whose progress is
  pushed as ``apps.job_progress`` notifications; check_update reports the
  installed version, whether the Hub has a newer one, and a running update). This *is* the "apps API"
  exposed over WebRTC, for clients that can't reach the LAN-only REST API.
* ``nfc.*`` — handled locally against the optional NFC reader, mirroring the
  ``/api/nfc`` routes for clients that only reach the daemon over WebRTC.
* anything else (e.g. ``conversation.*``) — relayed to the running app's
  ``/rpc`` WebSocket. The relay holds one persistent connection to that
  endpoint, which doubles as the daemon's subscription to the app's event
  stream: notifications the app pushes are re-broadcast to every connected
  client.

The relay runs on the daemon's main asyncio loop (where the AppManager and the
app WS client live); transports schedule :meth:`handle` onto it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from importlib.metadata import Distribution, distributions
from typing import TYPE_CHECKING, Any, Callable, Optional
from urllib.parse import urlparse

from pydantic import ValidationError
from websockets.asyncio.client import ClientConnection, connect

from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.apps.manager import AppManager, AppState
from reachy_mini.apps.sources import app_update_checker, local_common_venv
from reachy_mini.apps.sources.hf_space import get_python_app_info
from reachy_mini.daemon import startup_app_config
from reachy_mini.daemon.app import bg_job_register
from reachy_mini.daemon.app.startup_app import ensure_startup_app_installed
from reachy_mini.io.jsonrpc import (
    JsonRpcError,
    RpcErrorResponse,
    RpcRequest,
    RpcSuccess,
    error_from_exc,
    make_error,
    make_notification,
    make_result,
    parse_inbound,
    parse_request,
)

if TYPE_CHECKING:
    from reachy_mini.nfc import NfcReader

Reply = Callable[[dict[str, Any]], None]

_CONNECT_TIMEOUT_S = 5.0

# Notification pushed for every line / the end of an install, remove or update
# job: {job_id, command, status, line, final}.
JOB_PROGRESS_METHOD = "apps.job_progress"


class JsonRpcRelay:
    """Routes JSON-RPC control frames to the daemon or the running app."""

    def __init__(
        self,
        app_manager: AppManager,
        broadcast: Callable[[str], None],
        logger: Optional[logging.Logger] = None,
        nfc_reader: "NfcReader | None" = None,
    ) -> None:
        """Create the relay.

        Args:
            app_manager: source of app lifecycle + the running app URL.
            broadcast: sends a JSON string to *every* connected client (the
                backend's ``broadcast_to_all_clients``). Used to fan app
                notifications out.
            logger: optional logger.
            nfc_reader: the NFC reader, or None when the daemon runs without
                one (``--no-nfc``).

        """
        self._apps = app_manager
        self._nfc = nfc_reader
        self._broadcast = broadcast
        self.logger = logger or logging.getLogger("reachy_mini.jsonrpc_relay")

        self._app_ws: Optional[ClientConnection] = None
        self._app_ws_lock = asyncio.Lock()
        self._reader_task: Optional[asyncio.Task[None]] = None
        # relay-assigned id -> caller reply callback, for correlating app replies,
        # plus the caller's original id to restore on the way back.
        self._pending: dict[str, Reply] = {}
        self._pending_original_id: dict[str, Any] = {}
        self._counter = 0
        # Legacy `apps.install` calls running inline (outside the job register).
        self._inline_installs = 0
        # App behind each install/remove/update job this relay started, so
        # apps.start can refuse an app whose files are being changed.
        self._job_apps: dict[str, str] = {}
        # apps.update: the job updating each app, and the last failure.
        self._update_jobs: dict[str, str] = {}
        self._update_errors: dict[str, str] = {}
        # Serializes job starts (check -> stop/lookup -> re-check -> register).
        # Each frame is its own coroutine, so without it a double tap passes
        # the checks twice while the first one awaits.
        self._job_start_lock = asyncio.Lock()

    # -- entry point -------------------------------------------------------

    async def handle(self, raw: str, reply: Reply) -> None:
        """Route one incoming frame; ``reply`` sends a response to its sender."""
        try:
            req = parse_request(raw)
        except JsonRpcError as e:
            reply(error_from_exc(None, e))
            return

        try:
            if req.namespace == "apps":
                await self._handle_apps(req, reply)
            elif req.namespace == "nfc":
                await self._handle_nfc(req, reply)
            else:
                await self._relay_to_app(req, reply)
        except JsonRpcError as e:
            if not req.is_notification:
                reply(error_from_exc(req.id, e))
        except Exception as e:  # noqa: BLE001 - surface every failure to the caller
            self.logger.warning("relay error on %s: %s", req.method, e)
            if not req.is_notification:
                reply(error_from_exc(req.id, e))

    async def aclose(self) -> None:
        """Close the app connection and fail any pending calls."""
        await self._drop_app_ws()

    # -- apps.* (daemon-local) --------------------------------------------

    async def _handle_apps(self, req: RpcRequest, reply: Reply) -> None:
        verb = req.method.split(".", 1)[1] if "." in req.method else ""
        if verb == "start":
            name = req.params.get("name")
            if not isinstance(name, str) or not name:
                raise JsonRpcError(
                    "apps.start requires a 'name'", reason="invalid_params", code=-32602
                )
            if self._app_has_active_job(name):
                # Its files are being installed, removed or updated.
                raise JsonRpcError(
                    f"{name!r} is being installed or updated", reason="busy"
                )
            try:
                # keep_remote: the requester is this connected client that will
                # drive the app; don't evict its own WebRTC/DataChannel session.
                status = await self._apps.start_app(name, keep_remote=True)
            except RuntimeError as e:
                # start_app raises when an app already holds the robot slot.
                raise JsonRpcError(str(e), reason="already_running") from e
            result: Any = _status_dict(status)
        elif verb == "stop":
            try:
                await self._apps.stop_current_app()
            except RuntimeError as e:
                # stop_current_app raises when nothing is running.
                raise JsonRpcError(str(e), reason="not_running") from e
            result = {"stopped": True}
        elif verb == "restart":
            if not self._apps.is_app_running():
                raise JsonRpcError("No app is currently running", reason="not_running")
            try:
                status = await self._apps.restart_current_app(keep_remote=True)
            except RuntimeError as e:
                # The stop went through but the start did not (another client
                # took the slot meanwhile): the app is now stopped.
                raise JsonRpcError(str(e), reason="already_running") from e
            result = _status_dict(status)
        elif verb == "list_installed":
            installed = await self._apps.list_available_apps(SourceKind.INSTALLED)
            result = {"apps": [_installed_dict(a) for a in installed]}
        elif verb == "install_space":
            space_id = _required_str(req, "space_id")
            async with self._job_start_lock:
                result = await self._start_install_job(space_id)
        elif verb == "remove":
            name = _required_str(req, "name")
            async with self._job_start_lock:
                result = await self._start_remove_job(name)
        elif verb == "check_update":
            result = await self._check_update(_required_str(req, "name"))
        elif verb == "update":
            name = _required_str(req, "name")
            async with self._job_start_lock:
                result = await self._start_update(name)
        elif verb == "job_status":
            job_id = _required_str(req, "job_id")
            try:
                info = bg_job_register.get_info(job_id)
            except ValueError as e:
                raise JsonRpcError(str(e), reason="unknown_job") from e
            result = {"job_id": job_id, **_job_dict(info)}
        elif verb == "install":
            self._ensure_no_active_job()
            name = req.params.get("name")
            if not isinstance(name, str) or not name:
                raise JsonRpcError(
                    "apps.install requires a 'name'",
                    reason="invalid_params",
                    code=-32602,
                )
            # Install-if-missing from the HF catalog (no-op when already
            # installed). Can take minutes on a first install — callers
            # should use a generous timeout. Runs as its own task, so
            # other RPCs (apps.status polling, ...) keep answering.
            self._inline_installs += 1
            try:
                installed_ok = await ensure_startup_app_installed(self._apps, name)
            finally:
                self._inline_installs -= 1
            if not installed_ok:
                raise JsonRpcError(
                    f"could not install app {name!r} (not in the catalog, "
                    "or the install failed — see the daemon logs)",
                    reason="install_failed",
                )
            result = {"installed": True}
        elif verb == "status":
            result = self._current_status_dict()
        else:
            raise JsonRpcError(
                f"unknown method: {req.method}",
                reason="method_not_found",
                code=-32601,
            )
        if not req.is_notification:
            reply(make_result(req.id, result))

    async def _start_install_job(self, space_id: str) -> dict[str, Any]:
        if space_id.count("/") != 1 or not all(space_id.split("/")):
            raise JsonRpcError(
                "space_id must look like 'owner/repo'",
                reason="invalid_params",
                code=-32602,
            )
        self._ensure_no_active_job()
        app = await get_python_app_info(space_id)
        if app is None:
            raise JsonRpcError(
                f"{space_id!r} is not a reachable Reachy Mini Python app",
                reason="not_found",
            )
        # Reinstalling would swap the files under a live process; refuse like
        # remove and update do (and clear a crashed one).
        await self._ensure_not_running(app.name)
        # Again: another install may have started during the awaits above.
        self._ensure_no_active_job()
        job_id = bg_job_register.run_command(
            "install", self._apps.install_new_app, app, on_event=self._on_job_event
        )
        self._job_apps[job_id] = app.name
        _invalidate_rest_update_cache()
        return {"job_id": job_id, "name": app.name}

    async def _start_remove_job(self, name: str) -> dict[str, Any]:
        self._ensure_no_active_job()
        await self._ensure_not_running(name)
        if startup_app_config.get_startup_app() == name:
            # The REST route clears it and re-arms the antenna watcher, which
            # needs the FastAPI app state; refuse rather than leave a watcher
            # that would reinstall the app on the next wake-up.
            raise JsonRpcError(
                f"{name!r} is the startup app; unset it before removing it",
                reason="startup_app",
            )
        # Again, as in update: _ensure_not_running may have awaited.
        self._ensure_no_active_job()
        job_id = bg_job_register.run_command(
            "remove", self._apps.remove_app, name, on_event=self._on_job_event
        )
        self._job_apps[job_id] = name
        _invalidate_rest_update_cache()
        return {"job_id": job_id}

    # -- apps.check_update / apps.update ----------------------------------

    async def _check_update(self, name: str) -> dict[str, Any]:
        """Whether ``name`` is installed, has an update, or is being updated."""
        updating = self._update_job_active(name)
        installed = await self._apps.list_available_apps(SourceKind.INSTALLED)
        app = next((a for a in installed if a.name == name), None)
        info = None
        version = None
        if app is not None and not updating:
            # One HF API call; None when the install source can't be checked.
            info = await app_update_checker.check_app_update(
                app,
                wireless_version=self._apps.wireless_version,
                desktop_app_daemon=self._apps.desktop_app_daemon,
            )
            version = self._installed_version(name)
        return {
            "installed": app is not None,
            "version": version,
            "update_available": bool(info and info.update_available),
            # False when the source can't be checked (not from the Hub, or the
            # Hub call failed), so "no update" is not read as "up to date".
            "checkable": info is not None,
            "updating": updating,
            "error": self._update_errors.get(name),
        }

    def _installed_version(self, name: str) -> str | None:
        """Version of the distribution that provides app ``name``.

        ``name`` is the ``reachy_mini_apps`` entry-point name, which need not
        match the distribution name (a Space can ship a package named
        otherwise), so look the entry point up first.
        """
        site_packages = local_common_venv.get_app_site_packages(
            name, self._apps.wireless_version, self._apps.desktop_app_daemon
        )
        paths = [str(site_packages)] if site_packages else None
        dists: list[Distribution] = list(
            distributions(path=paths) if paths else distributions()
        )
        for dist in dists:
            if any(
                ep.group == "reachy_mini_apps" and ep.name == name
                for ep in dist.entry_points
            ):
                return dist.version
        match = next((d for d in dists if d.metadata["Name"] == name), None)
        return match.version if match else None

    async def _start_update(self, name: str) -> dict[str, Any]:
        """Update ``name`` as a background job; follow it with check_update.

        Runs through the job register like install/remove, so it shares their
        busy guard (no two uv runs in the apps venv) and streams its log as
        ``apps.job_progress``. The app is stopped first if it's the one
        running, since the manager refuses otherwise.
        """
        current = self._update_jobs.get(name)
        if current is not None and self._update_job_active(name):
            return {"started": False, "job_id": current}
        self._ensure_no_active_job()
        running = self._apps.current_app
        if (
            self._apps.is_app_running()
            and running is not None
            and running.status.info.name == name
        ):
            try:
                await self._apps.stop_current_app()
            except RuntimeError:
                # Already stopping (another client, the Talk tab's teardown)
                # or exited meanwhile: the manager refuses to update until the
                # slot is free, so wait for it rather than start a doomed job.
                await self._wait_slot_released(name)
        # Again: the stop awaited, so a double tap or another client may have
        # started a job meanwhile. No await from here to run_command, which
        # registers the job synchronously.
        current = self._update_jobs.get(name)
        if current is not None and self._update_job_active(name):
            return {"started": False, "job_id": current}
        self._ensure_no_active_job()
        self._update_errors.pop(name, None)

        def on_event(
            job_id: str, info: bg_job_register.JobInfo, line: str | None
        ) -> None:
            self._on_job_event(job_id, info, line)
            if line is None and info.status == bg_job_register.JobStatus.FAILED:
                self._update_errors[name] = _job_error(info)

        job_id = bg_job_register.run_command(
            "update", self._apps.update_app, name, on_event=on_event
        )
        self._update_jobs[name] = job_id
        self._job_apps[job_id] = name
        _invalidate_rest_update_cache()
        return {"started": True, "job_id": job_id}

    async def _wait_slot_released(self, name: str, timeout: float = 25.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            app = self._apps.current_app
            if (
                app is None
                or not self._apps.is_app_running()
                or app.status.info.name != name
            ):
                return
            await asyncio.sleep(0.1)
        raise JsonRpcError(f"{name!r} did not stop", reason="app_running")

    def _update_job_active(self, name: str) -> bool:
        job_id = self._update_jobs.get(name)
        return job_id is not None and _job_is_active(job_id)

    def _app_has_active_job(self, name: str) -> bool:
        return any(
            app == name and _job_is_active(job_id)
            for job_id, app in list(self._job_apps.items())
        )

    def _ensure_no_active_job(self) -> None:
        # Two uv/pip runs in the shared apps venv at once can corrupt it. The
        # register also holds jobs started over REST, so an RPC job never
        # overlaps one (the REST routes themselves do not check), and the
        # legacy inline `apps.install` is counted separately.
        if bg_job_register.has_active_job() or self._inline_installs:
            raise JsonRpcError(
                "another install/remove/update is in progress", reason="busy"
            )

    async def _ensure_not_running(self, name: str) -> None:
        app = self._apps.current_app
        if app is None or not self._apps.is_app_running():
            return
        if app.status.info.name != name:
            return
        if app.status.state == AppState.ERROR:
            # Crashed: no process left, but it still holds the slot (and
            # update_app refuses while it does). Clear it instead of leaving
            # the user stuck between "stop it first" and nothing to stop.
            await self._apps.stop_current_app()
            return
        raise JsonRpcError(f"{name!r} is running; stop it first", reason="app_running")

    def _on_job_event(
        self, job_id: str, info: bg_job_register.JobInfo, line: str | None
    ) -> None:
        # Called from the job thread; broadcast_to_all_clients is thread-safe
        # (the WebRTC self-update streams its progress the same way).
        params = {
            "job_id": job_id,
            "command": info.command,
            "status": info.status.value,
            "line": line,
            "final": line is None,
        }
        self._broadcast(json.dumps(make_notification(JOB_PROGRESS_METHOD, params)))

    # -- nfc.* (daemon-local) ---------------------------------------------

    async def _handle_nfc(self, req: RpcRequest, reply: Reply) -> None:
        # Imported here: the reader module pulls in serial I/O the relay does
        # not otherwise need.
        from reachy_mini.nfc import (
            NfcEnableRequest,
            NfcEraseRequest,
            NfcStatus,
            NfcWriteRequest,
            driver_available,
            no_tag,
            set_nfc_enabled,
        )

        verb = req.method.split(".", 1)[1] if "." in req.method else ""
        reader = self._nfc
        try:
            if verb == "status":
                status = (
                    reader.get_status()
                    if reader is not None
                    else NfcStatus(
                        connected=False,
                        enabled=False,
                        chip_detected=False,
                        driver_available=driver_available(),
                        error="NFC reader disabled",
                    )
                )
                result: Any = status.model_dump(mode="json")
            elif verb == "tag":
                tag = reader.get_tag() if reader is not None else no_tag()
                result = tag.model_dump(mode="json")
            elif verb == "set_enabled":
                enable = NfcEnableRequest.model_validate(req.params)
                if reader is None:
                    raise _nfc_disabled()
                set_nfc_enabled(enable.enabled)
                # stop() joins the reader thread: keep it off the event loop.
                await asyncio.to_thread(reader.start if enable.enabled else reader.stop)
                result = reader.get_status().model_dump(mode="json")
            elif verb == "erase":
                erase = NfcEraseRequest.model_validate(req.params)
                if reader is None or not reader.is_enabled():
                    raise _nfc_disabled()
                # Blocks until a tag is presented: keep it off the event loop.
                outcome = await asyncio.to_thread(reader.erase, erase)
                result = outcome.model_dump(mode="json")
            elif verb == "write":
                write = NfcWriteRequest.model_validate(req.params)
                if reader is None or not reader.is_enabled():
                    raise _nfc_disabled()
                outcome = await asyncio.to_thread(reader.write, write)
                result = outcome.model_dump(mode="json")
            else:
                raise JsonRpcError(
                    f"unknown method: {req.method}",
                    reason="method_not_found",
                    code=-32601,
                )
        except ValidationError as e:
            raise JsonRpcError(
                f"invalid params for {req.method}: {e.errors()[0]['msg']}",
                reason="invalid_params",
                code=-32602,
            ) from e
        if not req.is_notification:
            reply(make_result(req.id, result))

    def _current_status_dict(self) -> dict[str, Any]:
        app = self._apps.current_app
        if app is None or not self._apps.is_app_running():
            return {"state": "idle", "info": None, "error": None}
        return _status_dict(app.status)

    # -- relay to the running app's /rpc ----------------------------------

    async def _relay_to_app(self, req: RpcRequest, reply: Reply) -> None:
        ws = await self._ensure_app_ws()

        if req.is_notification:
            # Fire-and-forget: forward verbatim, nothing to correlate.
            await ws.send(req.model_dump_json())
            return

        self._counter += 1
        relay_id = f"relay-{self._counter}"
        # Record both correlation entries BEFORE the await: the reader task runs
        # on this same loop and can process the app's reply *during*
        # ``ws.send`` (a fast reply on the loopback WS). If `original_id` were
        # set after the await, that reply would restore id=None and the caller
        # would hang until timeout.
        self._pending[relay_id] = reply
        self._pending_original_id[relay_id] = req.id
        # Forward the request under a relay-assigned id (model_copy keeps every
        # other field intact); the reader restores `original_id` on the reply.
        outgoing = req.model_copy(update={"id": relay_id})
        try:
            await ws.send(outgoing.model_dump_json())
        except Exception as e:
            self._pending.pop(relay_id, None)
            self._pending_original_id.pop(relay_id, None)
            raise JsonRpcError(f"app unavailable: {e}", reason="app_unavailable") from e

    async def _ensure_app_ws(self) -> ClientConnection:
        async with self._app_ws_lock:
            if self._app_ws is not None:
                return self._app_ws

            url = self._apps.get_running_app_url()
            if not url:
                raise JsonRpcError(
                    "no app is running", reason="not_running", code=-32000
                )
            ws_url = _rpc_ws_url(url)
            try:
                self._app_ws = await asyncio.wait_for(
                    connect(ws_url), timeout=_CONNECT_TIMEOUT_S
                )
            except Exception as e:
                raise JsonRpcError(
                    f"cannot reach app /rpc at {ws_url}: {e}",
                    reason="app_unavailable",
                ) from e
            self._reader_task = asyncio.create_task(self._read_app(self._app_ws))
            self.logger.info("relay connected to app /rpc at %s", ws_url)
            return self._app_ws

    async def _read_app(self, ws: ClientConnection) -> None:
        try:
            async for message in ws:
                text = message if isinstance(message, str) else message.decode()
                self._on_app_frame(text)
        except Exception as e:
            self.logger.info("app /rpc reader ended: %s", e)
        finally:
            await self._drop_app_ws()

    def _on_app_frame(self, text: str) -> None:
        try:
            msg = parse_inbound(text)
        except JsonRpcError:
            return  # unparseable frame from the app — drop it
        # A correlated response (success or error) carries a relay-assigned id
        # (always a `relay-N` str) we're waiting on; restore the caller's id and
        # hand it back.
        if (
            isinstance(msg, (RpcSuccess, RpcErrorResponse))
            and isinstance(msg.id, str)
            and msg.id in self._pending
        ):
            reply = self._pending.pop(msg.id)
            original_id = self._pending_original_id.pop(msg.id, None)
            reply(msg.model_copy(update={"id": original_id}).model_dump())
        else:
            # A notification/event (or an uncorrelated response) — fan it out to
            # every client verbatim.
            self._broadcast(text)

    async def _drop_app_ws(self) -> None:
        ws, self._app_ws = self._app_ws, None
        task, self._reader_task = self._reader_task, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        # Fail every in-flight call so callers don't hang.
        pending, self._pending = self._pending, {}
        originals, self._pending_original_id = self._pending_original_id, {}
        for rid, reply in pending.items():
            reply(
                make_error(
                    originals.get(rid),
                    message="app connection lost",
                    reason="app_unavailable",
                )
            )


def _rpc_ws_url(app_url: str) -> str:
    """Turn an app's ``custom_app_url`` into its ``/rpc`` WebSocket URL.

    ``http://0.0.0.0:7860/`` -> ``ws://127.0.0.1:7860/rpc``. Binding on
    ``0.0.0.0`` means "listen on all interfaces"; the daemon reaches it on the
    loopback.
    """
    p = urlparse(app_url)
    host = "127.0.0.1" if p.hostname in (None, "0.0.0.0") else p.hostname
    scheme = "wss" if p.scheme == "https" else "ws"
    port = f":{p.port}" if p.port else ""
    return f"{scheme}://{host}{port}/rpc"


def _job_is_active(job_id: str) -> bool:
    try:
        status = bg_job_register.get_info(job_id).status
    except ValueError:
        return False
    return status in (
        bg_job_register.JobStatus.PENDING,
        bg_job_register.JobStatus.IN_PROGRESS,
    )


def _job_error(info: bg_job_register.JobInfo) -> str:
    """Return the failure message of a finished job, from its last error line."""
    marker = "failed with error: "
    for line in reversed(info.logs):
        if marker in line:
            return line.split(marker, 1)[1]
    return info.logs[-1] if info.logs else "update failed"


def _invalidate_rest_update_cache() -> None:
    """Drop the REST /apps/check-updates cache, as the REST job routes do.

    Without it, the dashboard can show a stale "update available" for up to
    its 5 min TTL after an app changed over the relay.
    """
    from reachy_mini.daemon.app.routers import apps as apps_router

    apps_router._update_cache = None


def _required_str(req: RpcRequest, key: str) -> str:
    value = req.params.get(key)
    if not isinstance(value, str) or not value:
        raise JsonRpcError(
            f"{req.method} requires a '{key}'", reason="invalid_params", code=-32602
        )
    return value


def _installed_dict(app: AppInfo) -> dict[str, Any]:
    """Serialize an installed app, without the bulky raw Space metadata."""
    extra = app.extra or {}
    card = extra.get("cardData")
    card = card if isinstance(card, dict) else {}
    space_id = extra.get("id")
    return {
        "name": app.name,
        "description": app.description or card.get("short_description") or "",
        "url": app.url,
        "space_id": space_id if isinstance(space_id, str) else None,
        "title": card.get("title"),
        "emoji": card.get("emoji"),
        "private": bool(extra.get("private")),
        "custom_app_url": extra.get("custom_app_url"),
    }


def _job_dict(info: bg_job_register.JobInfo) -> dict[str, Any]:
    return {"command": info.command, "status": info.status.value, "logs": info.logs}


def _nfc_disabled() -> JsonRpcError:
    """Refuse a tag operation while the reader is off (REST answers 503)."""
    return JsonRpcError("NFC reader disabled", reason="nfc_disabled")


def _status_dict(status: Any) -> dict[str, Any]:
    """Serialize an ``AppStatus`` to a plain JSON dict."""
    info = status.info
    return {
        "state": str(
            status.state.value if hasattr(status.state, "value") else status.state
        ),
        "error": status.error,
        "info": {"name": info.name, "description": info.description, "url": info.url},
    }
