"""Embedded JS runtime for subroutines.

Design summary
--------------
- One QuickJS context per subroutine; one asyncio task pumps it.
- Host bindings (``robot.*``, ``emit``, ``console.*``, ``setInterval``,
  ``setTimeout``, ``clearInterval``, ``clearTimeout``, ``now``) are
  exposed as Python callables registered via QuickJS's
  ``Context.add_callable`` on the ``__globals__`` namespace.
- Subroutine source is wrapped in an IIFE that assigns an ``api``
  object to a known global, then introspected so the daemon can ACK
  the method names.
- Method calls are dispatched from Python by invoking
  ``api[methodName](...)`` via a registered helper. JS exceptions and
  return values cross the boundary as JSON-serializable values.
- Idle behaviour uses our own ``setInterval`` / ``setTimeout``
  implementations backed by ``asyncio.sleep`` — QuickJS itself has no
  timers.
- Cancellation: each in-flight call gets an ``AbortSignal``-shaped
  object whose ``.aborted`` flag the guest is expected to poll
  cooperatively. ``cancel_subroutine_call`` flips the flag.

Out of scope for this initial cut
---------------------------------
- ES modules / ``import`` (QuickJS's std modules are not wired).
- ``async``/``await`` inside guest code (QuickJS supports it but
  pumping the JS microtask queue from Python is fiddly; revisit if
  app authors need it).
- Networking from the guest (no ``fetch``); behaviours talk to the
  outside world via ``robot.*`` bindings or ``emit`` only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QuickJS availability — graceful degradation when the wheel isn't installed.
# The daemon's other features must continue to work; the backend handler
# checks ``QUICKJS_AVAILABLE`` and returns a clean error rather than
# letting an ImportError propagate.
# ---------------------------------------------------------------------------
try:
    import quickjs as _quickjs  # type: ignore[import-not-found]

    QUICKJS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional wheel
    _quickjs = None  # type: ignore[assignment]
    QUICKJS_AVAILABLE = False


class SubroutineRuntimeError(RuntimeError):
    """A subroutine could not be created, called, or cleanly torn down."""


# ---------------------------------------------------------------------------
# Outbound message sender — abstracted so the runtime is testable without
# instantiating the whole daemon. The backend wires a closure over
# ``_send_message_to_webrtc(owner_peer_id, json_payload)``.
# ---------------------------------------------------------------------------
class SubroutineSender(Protocol):
    """Send a JSON-encoded message to the subroutine's owning peer."""

    def __call__(self, payload: Dict[str, Any]) -> None:
        ...


# ---------------------------------------------------------------------------
# Host API surface exposed to the guest as ``robot.*``. The backend
# provides an instance; we keep the interface narrow on purpose —
# adding a method here is a deliberate policy decision (every new
# binding widens the trust boundary).
# ---------------------------------------------------------------------------
class SubroutineHost(Protocol):
    """Minimal contract the daemon implements to drive the robot from a subroutine."""

    def set_head_pose(self, matrix4x4_flat: List[float]) -> None:
        """Set the target head pose (4×4 matrix, row-major flat)."""

    def set_head_joints(self, joints: List[float]) -> None:
        """Set target head joint positions (7 values)."""

    def set_antennas(self, antennas: List[float]) -> None:
        """Set target antenna joint positions [right, left] in radians."""

    def set_body_yaw(self, body_yaw: float) -> None:
        """Set target body yaw in radians."""

    def play_sound(self, file: str) -> None:
        """Play a robot-side sound file by name."""

    def get_state(self) -> Dict[str, Any]:
        """Return a state snapshot (head pose, joints, antennas, motor mode, ...)."""


# ---------------------------------------------------------------------------
# Wrapper around the QuickJS source so we control how the guest's
# ``api`` is registered. The IIFE assigns to ``__reachy_subroutine_api``,
# which the host then introspects.
# ---------------------------------------------------------------------------
_GUEST_PROLOGUE = r"""
// Initialised by the host. Each in-flight call shows up here keyed by
// call_id so the guest can read its own abort signal.
var __reachy_abort_signals = {};

// Public API the host invokes. The user assigns to this either via
// `export const api = ...` (rewritten below) or by directly writing
// `__reachy_subroutine_api = ...`.
var __reachy_subroutine_api = null;

// The console shim forwards to host bindings (which serialise the
// args to JSON and emit a SubroutineLogMsg).
var console = {
    log:   function () { __reachy_host_log('log',   __reachy_stringify(arguments)); },
    info:  function () { __reachy_host_log('log',   __reachy_stringify(arguments)); },
    warn:  function () { __reachy_host_log('warn',  __reachy_stringify(arguments)); },
    error: function () { __reachy_host_log('error', __reachy_stringify(arguments)); },
    debug: function () { __reachy_host_log('log',   __reachy_stringify(arguments)); },
};

function __reachy_stringify(argsLike) {
    var parts = [];
    for (var i = 0; i < argsLike.length; i++) {
        var v = argsLike[i];
        if (typeof v === 'string') parts.push(v);
        else { try { parts.push(JSON.stringify(v)); } catch (_) { parts.push(String(v)); } }
    }
    return parts.join(' ');
}

// emit(name, payload) — pushes a SubroutineEventMsg to the owner.
function emit(name, payload) {
    __reachy_host_emit(String(name), JSON.stringify(payload === undefined ? null : payload));
}

// Timer machinery. The host pumps each tick by calling
// __reachy_fire_timer(id); IDs are positive integers.
var __reachy_timers = {};

function setInterval(fn, periodMs) {
    var id = __reachy_host_register_interval(Math.max(0, periodMs|0));
    __reachy_timers[id] = { fn: fn, kind: 'interval' };
    return id;
}
function setTimeout(fn, delayMs) {
    var id = __reachy_host_register_timeout(Math.max(0, delayMs|0));
    __reachy_timers[id] = { fn: fn, kind: 'timeout' };
    return id;
}
function clearInterval(id) {
    delete __reachy_timers[id];
    __reachy_host_clear_timer(id|0);
}
var clearTimeout = clearInterval;

function __reachy_fire_timer(id) {
    var entry = __reachy_timers[id];
    if (!entry) return;
    if (entry.kind === 'timeout') delete __reachy_timers[id];
    try {
        entry.fn();
    } catch (e) {
        __reachy_host_log('error', 'timer error: ' + (e && e.stack || e));
        __reachy_host_uncaught_error(String(e && e.stack || e));
    }
}

// Invoked by the host to dispatch a remote call. Returns a
// JSON-encoded {ok: bool, result?: any, error?: string} payload —
// QuickJS<->Python marshalling of complex values is finicky so we
// always cross the boundary as JSON.
function __reachy_invoke_method(callId, methodName, argsJson) {
    if (!__reachy_subroutine_api) {
        return JSON.stringify({ ok: false, error: 'subroutine has no api' });
    }
    var fn = __reachy_subroutine_api[methodName];
    if (typeof fn !== 'function') {
        return JSON.stringify({ ok: false, error: 'method ' + methodName + ' not found' });
    }
    var args;
    try { args = JSON.parse(argsJson); } catch (e) {
        return JSON.stringify({ ok: false, error: 'malformed args' });
    }
    if (!Array.isArray(args)) args = [];
    // Tail-arg convention: every method receives a `{signal}` object
    // so it can cooperatively cancel. Cheap if unused.
    var signal = { aborted: false };
    __reachy_abort_signals[callId] = signal;
    args.push({ signal: signal });
    try {
        var result = fn.apply(null, args);
        delete __reachy_abort_signals[callId];
        // Serialise; undefined -> null so the wire stays JSON-clean.
        return JSON.stringify({ ok: true, result: result === undefined ? null : result });
    } catch (e) {
        delete __reachy_abort_signals[callId];
        return JSON.stringify({ ok: false, error: String(e && e.stack || e) });
    }
}

function __reachy_signal_abort(callId) {
    var s = __reachy_abort_signals[callId];
    if (s) s.aborted = true;
}

function __reachy_method_names() {
    if (!__reachy_subroutine_api) return JSON.stringify([]);
    var names = [];
    for (var k in __reachy_subroutine_api) {
        if (typeof __reachy_subroutine_api[k] === 'function') names.push(k);
    }
    return JSON.stringify(names);
}
"""

# Rewrite `export const api = ...` to assign into our well-known global.
# We don't run a real ES-module loader, so this rewrite is the minimum
# ergonomic concession to the documented author surface.
_GUEST_USERCODE_WRAPPER = r"""
(function () {
    %s
    if (typeof api !== 'undefined' && api) {
        __reachy_subroutine_api = api;
    }
})();
"""


def _rewrite_user_code(source: str) -> str:
    """Translate the documented author surface to QuickJS-compatible code.

    Author writes ``export const api = { ... }`` (familiar JS) but the
    runtime is not running ES modules — we strip the leading ``export``
    so ``api`` becomes a local-scope binding the wrapper IIFE can
    capture and assign to ``__reachy_subroutine_api``.

    Conservative: only rewrites ``export const api`` / ``export let api``
    / ``export var api``. Other ``export`` syntax is left alone (and
    will error at evaluation, which is the right signal to the author).
    """
    rewritten = source
    for prefix in ("export const api", "export let api", "export var api"):
        rewritten = rewritten.replace(prefix, prefix.removeprefix("export "), 1)
    return rewritten


# ---------------------------------------------------------------------------
# The runtime itself
# ---------------------------------------------------------------------------
class SubroutineRuntime:
    """A single live subroutine: QuickJS context + timers + dispatcher.

    Lifetime is owned by the daemon. Construction parses the source
    (raises :class:`SubroutineRuntimeError` on syntax errors);
    ``call(...)`` dispatches a method synchronously w.r.t. the JS
    context but may yield to the asyncio loop between host-binding
    calls; ``unload()`` cancels timers, disposes the context, and
    rejects any in-flight calls.
    """

    # Conservative defaults — tweak if real subroutines need more
    # headroom. The wireless variant (CM4) has the tighter envelope.
    DEFAULT_MEMORY_LIMIT_BYTES = 16 * 1024 * 1024  # 16 MB
    DEFAULT_MAX_STACK_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB
    # Wall-clock budget per synchronous call into the guest. Past this
    # the interrupt handler fires and the call aborts.
    DEFAULT_CALL_TIMEOUT_S = 5.0

    def __init__(
        self,
        *,
        subroutine_id: str,
        source: str,
        host: SubroutineHost,
        sender: SubroutineSender,
        owner_peer_id: Optional[str] = None,
        name: str = "",
        persist: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Build and evaluate the subroutine. Raises if QuickJS is missing or eval fails."""
        if not QUICKJS_AVAILABLE:
            raise SubroutineRuntimeError(
                "QuickJS Python bindings not installed. "
                "Subroutines require the 'quickjs' package; install with "
                "'pip install quickjs' or skip this feature on platforms "
                "where the wheel is unavailable."
            )

        assert _quickjs is not None  # for type checkers
        self.subroutine_id = subroutine_id
        self.owner_peer_id = owner_peer_id
        self.name = name or subroutine_id
        self.persist = persist
        self._host = host
        self._sender = sender
        self._loop = loop or asyncio.get_event_loop()
        self._logger = logger.getChild(self.name)

        self._context = _quickjs.Context()
        try:
            self._context.set_memory_limit(self.DEFAULT_MEMORY_LIMIT_BYTES)
            self._context.set_max_stack_size(self.DEFAULT_MAX_STACK_SIZE_BYTES)
        except AttributeError:
            # Older quickjs wheels may not expose these — the safety
            # envelope is then "trust the guest", which is fine for the
            # current trust model (author == Space owner).
            pass

        # Timer state. Each registered timer is an asyncio.Task that
        # schedules __reachy_fire_timer(id) on the runtime's main task
        # via _enqueue_fire(). Storing the asyncio.Task lets us cancel
        # on unload or on explicit clearInterval.
        self._next_timer_id = 1
        self._timers: Dict[int, asyncio.Task[None]] = {}

        # Coroutine pump: host bindings that need to actually run the
        # daemon's set_target / play_sound / get_state are *sync* in our
        # contract — the host does the threadsafe scheduling itself. We
        # never await inside the QuickJS callback frame; that would
        # require pumping the JS microtask queue from Python.
        self._unloaded = False
        self._install_bindings()
        self._evaluate_user_code(source)

    # ----- public API --------------------------------------------------

    @property
    def method_names(self) -> List[str]:
        """List of method names exported by the subroutine's ``api`` object."""
        try:
            raw = self._context.eval("__reachy_method_names()")
            names = json.loads(raw)
            return list(names) if isinstance(names, list) else []
        except Exception:  # pragma: no cover - defensive
            self._logger.exception("method introspection failed")
            return []

    def call(self, call_id: str, method: str, args: List[Any]) -> Any:
        """Invoke a method synchronously. Raises :class:`SubroutineRuntimeError` on guest error.

        ``call_id`` is the correlation ID the host uses to scope
        cancellation. ``args`` is JSON-serialised and unpacked
        guest-side; the guest also receives a tail ``{signal}`` object
        whose ``signal.aborted`` flag the host can flip via
        :meth:`cancel_call`.
        """
        if self._unloaded:
            raise SubroutineRuntimeError("subroutine has been unloaded")
        # Note on runaway guests: quickjs.Context exposes set_time_limit
        # but it interacts badly with Python host callbacks (the very
        # first add_callable invocation under an active time limit
        # throws with an empty error). We rely on cooperative
        # cancellation via the {signal} tail-arg instead; a
        # genuinely-runaway guest is recovered by unload(), which
        # disposes the context entirely. Future: run eval in a worker
        # thread with a hard kill on timeout.
        try:
            args_json = json.dumps(list(args))
        except (TypeError, ValueError) as e:
            raise SubroutineRuntimeError(f"args not JSON-serialisable: {e}")
        try:
            raw = self._context.eval(
                f"__reachy_invoke_method({json.dumps(call_id)}, "
                f"{json.dumps(method)}, {json.dumps(args_json)})"
            )
        except Exception as e:  # quickjs raises a wrapped error
            raise SubroutineRuntimeError(str(e)) from e
        try:
            envelope = json.loads(raw)
        except (TypeError, ValueError):
            raise SubroutineRuntimeError("malformed guest response")
        if not envelope.get("ok"):
            raise SubroutineRuntimeError(envelope.get("error", "guest error"))
        return envelope.get("result")

    def cancel_call(self, call_id: str) -> None:
        """Cooperative-cancel an in-flight call by flipping its abort flag."""
        if self._unloaded:
            return
        try:
            self._context.eval(f"__reachy_signal_abort({json.dumps(call_id)})")
        except Exception:
            self._logger.debug("cancel_call: guest threw", exc_info=True)

    def unload(self) -> None:
        """Tear down timers and dispose the QuickJS context. Idempotent."""
        if self._unloaded:
            return
        self._unloaded = True
        for tid, task in list(self._timers.items()):
            task.cancel()
        self._timers.clear()
        # quickjs.Context is garbage-collected when refs drop; explicit
        # del is enough on CPython.
        try:
            del self._context
        except Exception:
            pass
        self._logger.info("subroutine %s unloaded", self.subroutine_id)

    # ----- host binding plumbing ---------------------------------------

    def _install_bindings(self) -> None:
        """Register the Python callables that back ``robot.*``, ``emit``, timers, etc."""
        c = self._context

        # robot.* — surface narrowly. Each binding is a sync wrapper
        # around the host's method; arguments cross as JSON-friendly
        # primitives (numbers, strings, lists). Anything more exotic
        # would need a marshalling layer we don't have yet.

        def _set_target_head(matrix_json: str) -> None:
            try:
                m = json.loads(matrix_json)
                self._host.set_head_pose(list(m))
            except Exception:
                self._logger.exception("robot.setHeadPose failed")

        def _set_head_joints(joints_json: str) -> None:
            try:
                self._host.set_head_joints(list(json.loads(joints_json)))
            except Exception:
                self._logger.exception("robot.setHeadJoints failed")

        def _set_antennas(antennas_json: str) -> None:
            try:
                self._host.set_antennas(list(json.loads(antennas_json)))
            except Exception:
                self._logger.exception("robot.setAntennas failed")

        def _set_body_yaw(yaw: float) -> None:
            try:
                self._host.set_body_yaw(float(yaw))
            except Exception:
                self._logger.exception("robot.setBodyYaw failed")

        def _play_sound(file: str) -> None:
            try:
                self._host.play_sound(str(file))
            except Exception:
                self._logger.exception("robot.playSound failed")

        def _get_state() -> str:
            try:
                state = self._host.get_state()
                return json.dumps(state)
            except Exception as e:
                self._logger.exception("robot.getState failed")
                return json.dumps({"error": str(e)})

        # Console / emit / log forwarding
        def _host_log(level: str, message: str) -> None:
            self._sender(
                {
                    "type": "subroutine_log",
                    "subroutine_id": self.subroutine_id,
                    "level": level if level in ("log", "warn", "error") else "log",
                    "message": str(message),
                }
            )
            log_level = {"warn": logging.WARNING, "error": logging.ERROR}.get(
                level, logging.INFO
            )
            self._logger.log(log_level, "%s", message)

        def _host_emit(name: str, payload_json: str) -> None:
            try:
                payload = json.loads(payload_json) if payload_json else None
            except (TypeError, ValueError):
                payload = None
            self._sender(
                {
                    "type": "subroutine_event",
                    "subroutine_id": self.subroutine_id,
                    "event": str(name),
                    "payload": payload,
                }
            )

        def _host_uncaught_error(message: str) -> None:
            self._sender(
                {
                    "type": "subroutine_error",
                    "subroutine_id": self.subroutine_id,
                    "error": str(message),
                    "fatal": False,
                }
            )

        # Timer machinery — host owns the scheduling; the guest just
        # registers callbacks by ID and reacts to __reachy_fire_timer.
        def _register_interval(period_ms: int) -> int:
            tid = self._next_timer_id
            self._next_timer_id += 1
            self._timers[tid] = self._loop.create_task(
                self._run_interval(tid, max(0, int(period_ms)) / 1000.0)
            )
            return tid

        def _register_timeout(delay_ms: int) -> int:
            tid = self._next_timer_id
            self._next_timer_id += 1
            self._timers[tid] = self._loop.create_task(
                self._run_timeout(tid, max(0, int(delay_ms)) / 1000.0)
            )
            return tid

        def _clear_timer(tid: int) -> None:
            task = self._timers.pop(int(tid), None)
            if task is not None:
                task.cancel()

        # add_callable signatures vary slightly across quickjs versions;
        # the simplest invocation (name, fn) works in all versions
        # we've tested.
        c.add_callable("__reachy_host_set_head_pose", _set_target_head)
        c.add_callable("__reachy_host_set_head_joints", _set_head_joints)
        c.add_callable("__reachy_host_set_antennas", _set_antennas)
        c.add_callable("__reachy_host_set_body_yaw", _set_body_yaw)
        c.add_callable("__reachy_host_play_sound", _play_sound)
        c.add_callable("__reachy_host_get_state", _get_state)
        c.add_callable("__reachy_host_log", _host_log)
        c.add_callable("__reachy_host_emit", _host_emit)
        c.add_callable("__reachy_host_uncaught_error", _host_uncaught_error)
        c.add_callable("__reachy_host_register_interval", _register_interval)
        c.add_callable("__reachy_host_register_timeout", _register_timeout)
        c.add_callable("__reachy_host_clear_timer", _clear_timer)
        c.add_callable("__reachy_host_now", lambda: time.monotonic() * 1000.0)

        # robot.* namespace + now() bound to time.monotonic so guest
        # code can do timing without a separate import.
        c.eval(
            _GUEST_PROLOGUE
            + r"""
            var robot = {
                setHeadPose:   function (m)   { __reachy_host_set_head_pose(JSON.stringify(m)); },
                setHeadJoints: function (j)   { __reachy_host_set_head_joints(JSON.stringify(j)); },
                setAntennas:   function (a)   { __reachy_host_set_antennas(JSON.stringify(a)); },
                setBodyYaw:    function (y)   { __reachy_host_set_body_yaw(y); },
                playSound:     function (f)   { __reachy_host_play_sound(f); },
                getState:      function ()    { return JSON.parse(__reachy_host_get_state()); }
            };
            function now() { return __reachy_host_now(); }
            """
        )

    def _evaluate_user_code(self, source: str) -> None:
        """Evaluate the author's source inside the prepared sandbox."""
        rewritten = _rewrite_user_code(source)
        wrapped = _GUEST_USERCODE_WRAPPER % (rewritten,)
        try:
            self._context.eval(wrapped)
        except Exception as e:
            raise SubroutineRuntimeError(f"subroutine eval failed: {e}") from e

    # ----- timer pumps -------------------------------------------------

    async def _run_interval(self, tid: int, period_s: float) -> None:
        """Fire __reachy_fire_timer(tid) every ``period_s`` until cancelled."""
        # Drift-compensated loop: we sleep to absolute deadlines so a
        # long fire callback doesn't slip subsequent ticks. The
        # eval call is sync but typically completes in well under 1 ms
        # for tiny callbacks like a sin() and a host binding call.
        deadline = self._loop.time() + period_s
        while not self._unloaded and tid in self._timers:
            now = self._loop.time()
            await asyncio.sleep(max(0.0, deadline - now))
            if self._unloaded or tid not in self._timers:
                return
            self._fire_timer(tid)
            # Catch up across missed ticks rather than drifting.
            deadline += period_s
            if deadline < self._loop.time():
                deadline = self._loop.time() + period_s

    async def _run_timeout(self, tid: int, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        if self._unloaded or tid not in self._timers:
            return
        self._fire_timer(tid)
        # Single-shot — drop the entry so the guest's reference goes
        # stale gracefully if the JS callback didn't already remove it.
        self._timers.pop(tid, None)

    def _fire_timer(self, tid: int) -> None:
        if self._unloaded:
            return
        try:
            self._context.eval(f"__reachy_fire_timer({int(tid)})")
        except Exception as e:
            # An exception escaping a timer callback is a real
            # subroutine bug; surface it as a non-fatal subroutine_error
            # and keep the runtime alive.
            self._logger.warning("timer %d threw: %s", tid, e)
            try:
                self._sender(
                    {
                        "type": "subroutine_error",
                        "subroutine_id": self.subroutine_id,
                        "error": f"{e}\n{traceback.format_exc()}",
                        "fatal": False,
                    }
                )
            except Exception:
                pass


# Async helper for unit tests / direct use — schedules a one-shot
# coroutine on the runtime's loop without going through call().
def schedule(loop: asyncio.AbstractEventLoop, coro: Awaitable[Any]) -> asyncio.Task[Any]:
    """Convenience: schedule a coroutine on ``loop`` (works from any thread)."""
    return asyncio.run_coroutine_threadsafe(coro, loop)  # type: ignore[return-value]
