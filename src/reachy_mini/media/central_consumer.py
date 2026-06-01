"""Python aiortc consumer of the Reachy Mini HF central signaling relay.

Counterpart to :mod:`reachy_mini.media.central_signaling_relay` — the
daemon-side *producer* — this module is the *consumer*: a hardware-free
client that subscribes to the central HF Space at
``https://pollen-robotics-reachy-mini-central.hf.space``, negotiates
WebRTC with one robot, and exposes:

* decoded **video** frames as numpy arrays (:meth:`latest_frame`);
* bidirectional **audio** — the robot mic decoded to mono float32 via an
  ``on_pcm`` callback, and an outbound ``out_track`` whose audio plays out
  the robot speaker (the daemon advertises audio as ``sendrecv``);
* a thread-safe ``send_command`` for the ``data`` channel the daemon offers,
  plus an ``on_command_ready`` hook for one-shot setup once it opens.

All audio/command extras are optional — with none of them set this behaves
exactly as the original video-only consumer.

Use this from any cloud backend (HF Space, Cloud Run, etc.) that needs to
read a Reachy Mini's camera and/or drive the robot via the visitor's HF
token, without instantiating :class:`reachy_mini.ReachyMini` (which
requires daemon access and the full hardware stack).

Per-process / per-session: instantiate one per visitor (so the visitor's
short-lived HF Bearer token scopes the listener to robots they actually
own). The instance maintains a long-lived SSE channel to central plus a
single :class:`~aiortc.RTCPeerConnection` to one robot.

Example::

    from reachy_mini.media.central_consumer import ReachyCentralConsumer

    consumer = ReachyCentralConsumer(
        hf_token=visitor_token,
        robot_peer_id=visitor_robot_peer_id,
        consumer_label="my-cloud-backend/visitor-1",
    )
    await consumer.start()

    # Pull the freshest frame whenever you need it (e.g. in your
    # inference loop). ``frame_id`` is a monotonically increasing
    # counter so you can dedupe without comparing arrays.
    snap = consumer.latest_frame()
    if snap is not None:
        frame_id, rgb = snap
        ...

    # Send a JSON command on the robot's "data" data channel (e.g.
    # ``set_full_target`` / ``goto_target``). Thread-safe — schedules
    # the actual ``RTCDataChannel.send`` on aiortc's event loop.
    consumer.send_command({"type": "goto_target", "head": [...], "duration": 0.4})

    await consumer.stop()

Protocol summary (matches ``reachy-mini-sdk.js``):

* ``GET {central}/events`` — Server-Sent Events. We receive ``welcome``,
  ``list``, ``peer`` (SDP offer + remote ICE), ``endSession``.
* ``POST {central}/send`` — JSON envelopes. We send ``setPeerStatus``,
  ``startSession``, ``peer`` (SDP answer), ``endSession``.

The robot creates the offer; we answer. aiortc gathers all local ICE
candidates before ``setLocalDescription`` returns (no out-trickle), so
the answer SDP is sent once; trickled remote candidates arriving after
the offer are buffered and replayed once the remote description is set.

Other niceties baked in:

* Heartbeat ``setPeerStatus`` every ~10 s (refreshed from the welcome
  lease) so central doesn't evict us as a stale listener.
* Force-SSE-reconnect when central returns ``"Connect to /events first"``
  on a POST — that's the signature of a stale relay-side peer record.
* ``send_command`` marshalled via ``loop.call_soon_threadsafe`` so
  callers running in worker threads (e.g. GPU pipelines) can safely
  drive the robot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Awaitable, Callable, List, Optional, Tuple

import aiohttp
import av
import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import candidate_from_sdp

logger = logging.getLogger(__name__)


DEFAULT_CENTRAL_URL = "https://pollen-robotics-reachy-mini-central.hf.space"
# Sample rate incoming mic audio is resampled to before ``on_pcm`` (mono
# float32). 24 kHz suits most speech models; override via ``audio_sample_rate``.
DEFAULT_AUDIO_SAMPLE_RATE = 24000


def _patch_aiortc_dtls_ciphers() -> None:
    """Append a cipher the robot's GStreamer ``webrtcsink`` accepts.

    aiortc's default DTLS cipher list shares no cipher with the robot
    daemon's GStreamer ``webrtcsink``, so DTLS never completes and the
    peer connection never reaches "connected". We wrap
    ``RTCCertificate._create_ssl_context`` to add
    ``ECDHE-RSA-AES128-GCM-SHA256``.

    Upstream fix tracked in aiortc PR #1392
    (https://github.com/aiortc/aiortc/pull/1392); remove this shim once a
    released aiortc negotiates a compatible cipher by default.
    """
    try:
        from aiortc.rtcdtlstransport import RTCCertificate
    except Exception as e:  # pragma: no cover
        logger.warning("[reachy] dtls-patch: aiortc import failed: %r", e)
        return
    if getattr(RTCCertificate, "_reachy_cipher_patched", False):
        return
    try:
        _orig = RTCCertificate._create_ssl_context
    except AttributeError as e:
        logger.warning("[reachy] dtls-patch: _create_ssl_context not found: %r", e)
        return
    ciphers = (
        b"ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-ECDSA-CHACHA20-POLY1305:"
        b"ECDHE-ECDSA-AES128-SHA:ECDHE-ECDSA-AES256-SHA:"
        b"ECDHE-RSA-AES128-GCM-SHA256"
    )

    def _patched(self, srtp_profiles):
        ctx = _orig(self, srtp_profiles)
        try:
            ctx.set_cipher_list(ciphers)
        except Exception as e:  # pragma: no cover
            logger.warning("[reachy] dtls-patch: set_cipher_list failed: %r", e)
        return ctx

    RTCCertificate._create_ssl_context = _patched
    RTCCertificate._reachy_cipher_patched = True
    logger.info("[reachy] DTLS cipher list extended (+ECDHE-RSA-AES128-GCM-SHA256)")


# Applied once at import: harmless and idempotent, and required for the
# robot leg to complete DTLS.
_patch_aiortc_dtls_ciphers()


class ReachyCentralConsumer:
    """Single-robot WebRTC consumer of the HF central signaling relay."""

    def __init__(
        self,
        hf_token: str,
        central_url: str = DEFAULT_CENTRAL_URL,
        robot_peer_id: Optional[str] = None,
        robot_name: str = "reachymini",
        ice_servers_provider: Optional[
            Callable[[], Awaitable[List[RTCIceServer]]]
        ] = None,
        consumer_label: str = "reachy-mini-consumer",
        on_pcm: Optional[Callable[[np.ndarray], None]] = None,
        out_track: Optional[MediaStreamTrack] = None,
        on_command_ready: Optional[Callable[[], None]] = None,
        audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    ):
        """Args (audio additions; all optional — video behaviour unchanged):

        on_pcm: called with each decoded chunk of the robot's microphone as a
            mono float32 ndarray (shape ``(n,)``) resampled to
            ``audio_sample_rate``. When ``None``, inbound audio is ignored.
        out_track: an :class:`aiortc.MediaStreamTrack` (``kind == "audio"``)
            whose frames are sent to the robot speaker. Added to the peer
            connection on the robot's sendrecv audio m-line before
            ``createAnswer``. When ``None``, we send no audio.
        on_command_ready: called (no args) once the robot ``data`` channel
            opens, so callers can send one-shot setup commands — e.g.
            ``send_command({"type": "set_wobbling", "enabled": True})``.
        audio_sample_rate: rate the mic audio is resampled to for ``on_pcm``.
        """
        if not hf_token:
            raise ValueError("hf_token is required for ReachyCentralConsumer")
        self._hf_token = hf_token
        self._central_url = central_url.rstrip("/")
        self._robot_name = robot_name
        self._consumer_label = consumer_label
        self._ice_servers_provider = ice_servers_provider
        self._on_pcm = on_pcm
        self._out_track = out_track
        self._out_track_added = False
        self._on_command_ready = on_command_ready
        self._audio_sample_rate = int(audio_sample_rate)

        # ``_target_peer_id_pinned`` is the value supplied at construction
        # (or via env). When set we never auto-rediscover. When unset we
        # auto-pick from each fresh ``list`` event.
        self._target_peer_id_pinned: Optional[str] = robot_peer_id
        self._target_peer_id: Optional[str] = robot_peer_id

        self._http: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self._discovery_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._stopping = False
        # The live SSE response, so other tasks can force a reconnect by
        # closing it (e.g. when central has deregistered our peer).
        self._sse_resp = None
        self._heartbeat_interval = 10.0  # refreshed from welcome lease

        self._self_peer_id: Optional[str] = None
        self._session_id: Optional[str] = None
        self._starting_session = False
        self._pc: Optional[RTCPeerConnection] = None
        self._pending_remote_ice: List[dict] = []
        self._remote_desc_set = False
        self._pc_connected = False

        # Snapshot of (monotonic id, ndarray). Stored as a single tuple so a
        # reader gets a consistent pair under the GIL — assigning a tuple is
        # an atomic reference swap. The id lets pollers detect new frames
        # without relying on object identity (CPython recycles ids).
        self._latest: Optional[Tuple[int, np.ndarray]] = None
        self._frame_counter = 0
        # (width, height) of the source frames as the daemon's webrtcsink
        # emits them — None until the first frame arrives.
        self._source_size: Optional[Tuple[int, int]] = None

        # Inbound-audio diagnostics: decoded mic frames + total samples
        # handed to ``on_pcm`` (at ``audio_sample_rate``, mono).
        self._audio_frames = 0
        self._pcm_samples = 0

        # Robot-side data channel ("data"), captured via pc.on("datachannel")
        # when the daemon's offer arrives. Sendrecv; we only send commands.
        # aiortc's RTCDataChannel.send() is not thread-safe — we marshal
        # every send through ``_cmd_loop.call_soon_threadsafe`` so worker
        # threads can use ``send_command`` safely.
        self._cmd_channel = None  # type: ignore[assignment]
        self._cmd_channel_open = False
        self._cmd_loop: Optional[asyncio.AbstractEventLoop] = None
        self._cmd_sent = 0
        self._cmd_send_errors = 0

    # ------------------------------------------------------------------ API

    def is_connected(self) -> bool:
        return self._pc_connected

    def is_command_ready(self) -> bool:
        return self._cmd_channel_open and self._cmd_channel is not None

    def send_command(self, envelope: dict) -> bool:
        """JSON-encode ``envelope`` and write it to the robot's data
        channel. Thread-safe: schedules the actual ``send()`` on the
        event loop that owns the aiortc peer connection. Returns True
        if the send was queued, False if the channel isn't open."""
        ch = self._cmd_channel
        loop = self._cmd_loop
        if ch is None or not self._cmd_channel_open or loop is None:
            return False
        payload = json.dumps(envelope)

        def _do_send():
            try:
                ch.send(payload)
            except Exception as e:
                self._cmd_send_errors += 1
                if self._cmd_send_errors <= 3:
                    print(f"[{self._consumer_label}] send_command failed: {e!r}")

        try:
            loop.call_soon_threadsafe(_do_send)
            self._cmd_sent += 1
            return True
        except RuntimeError:
            # Loop closed mid-send; treat as channel closed.
            return False

    def _on_cmd_ready(self) -> None:
        """Robot ``data`` channel just opened — fire the caller's hook so it
        can send one-shot setup (e.g. enable head wobbling)."""
        if self._on_command_ready is not None:
            try:
                self._on_command_ready()
            except Exception as e:
                logger.warning("[reachy] on_command_ready hook failed: %r", e)

    def latest_frame(self) -> Optional[Tuple[int, np.ndarray]]:
        """Return ``(frame_id, ndarray)`` or ``None`` if no frame yet.

        ``frame_id`` is a monotonically increasing counter — pollers can
        compare it to the last id they processed to detect new frames.
        """
        return self._latest

    def status(self) -> dict:
        pc_state = self._pc.connectionState if self._pc is not None else None
        return {
            "connected": self._pc_connected,
            "self_peer_id": self._self_peer_id,
            "robot_peer_id": self._target_peer_id,
            "session_id": self._session_id,
            "pc_state": pc_state,
            "frames": self._frame_counter,
            "audio_frames": self._audio_frames,
            "pcm_samples": self._pcm_samples,
        }

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._http = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._run_forever(), name="reachy-consumer")

    async def stop(self) -> None:
        self._stopping = True
        for attr in ("_heartbeat_task", "_discovery_task", "_task"):
            t = getattr(self, attr)
            if t is not None:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
                setattr(self, attr, None)
        await self._teardown_session()
        if self._http is not None:
            await self._http.close()
            self._http = None

    # --------------------------------------------------------------- runner

    async def _run_forever(self) -> None:
        backoff = 1.0
        while not self._stopping:
            try:
                await self._connect_once()
                # Normal SSE return (server closed the stream) → retry.
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "[reachy] connection error: %r; reconnecting in %.1fs",
                    e, backoff,
                )
            await self._teardown_session()
            if self._stopping:
                return
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff * 2.0, 30.0)

    async def _connect_once(self) -> None:
        url = f"{self._central_url}/events"
        headers = {
            "Authorization": f"Bearer {self._hf_token}",
            "Accept": "text/event-stream",
        }
        # No total timeout (long-lived); sock_read covers stalls.
        timeout = aiohttp.ClientTimeout(total=None, sock_read=60.0)
        assert self._http is not None
        async with self._http.get(url, headers=headers, timeout=timeout) as resp:
            if resp.status != 200:
                txt = (await resp.text())[:200]
                raise RuntimeError(f"SSE /events HTTP {resp.status}: {txt!r}")
            logger.info("[reachy] SSE connected to %s", url)
            self._sse_resp = resp
            try:
                async for event in self._iter_sse_events(resp):
                    if self._stopping:
                        return
                    await self._handle_event(event)
            finally:
                self._sse_resp = None
                self._self_peer_id = None

    def _force_sse_reconnect(self, why: str) -> None:
        """Break the current SSE read loop so _run_forever reconnects.

        Used when central has deregistered our peer (a /send returns
        "Connect to /events first") but our SSE socket is a half-open
        zombie that never errors on its own.
        """
        resp = self._sse_resp
        if resp is not None:
            print(f"[reachy] forcing SSE reconnect ({why})")
            try:
                resp.close()  # breaks readline() -> _connect_once returns
            except Exception:
                pass

    @staticmethod
    async def _iter_sse_events(resp):
        """Yield decoded JSON dicts from an SSE response."""
        data_lines: List[str] = []
        while True:
            raw = await resp.content.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "":
                if data_lines:
                    payload = "\n".join(data_lines)
                    data_lines = []
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        logger.debug("[reachy] non-json SSE data: %r", payload[:200])
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            # Other SSE fields (event:, id:, retry:) are ignored.

    # --------------------------------------------------------------- events

    async def _handle_event(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "welcome":
            self._self_peer_id = msg.get("peerId")
            lease = msg.get("lease_seconds")
            rec = msg.get("recommended_heartbeat_interval_seconds")
            if isinstance(rec, (int, float)) and rec > 0:
                self._heartbeat_interval = max(2.0, float(rec))
            elif isinstance(lease, (int, float)) and lease > 0:
                self._heartbeat_interval = max(2.0, float(lease) / 3.0)
            print(f"[reachy] welcome peerId={self._self_peer_id} "
                  f"(heartbeat {self._heartbeat_interval:.1f}s)")
            await self._post({
                "type": "setPeerStatus",
                "roles": ["listener"],
                "meta": {"name": self._consumer_label},
            })
            # Keep our listener peer registered: central evicts peers with
            # no inbound /send within the lease (SSE pings don't refresh
            # it), which otherwise leaves a zombie SSE + startSession 400s.
            self._ensure_heartbeat_task()
            # Discovery is handled solely by the HTTP-poll loop
            # (_discovery_loop) — central's SSE `list` push is unreliable
            # for already-online producers. Having two paths call
            # startSession races and causes central to kill one session.
            self._ensure_discovery_task()
            if self._target_peer_id is not None:
                # Only when explicitly pinned via REACHY_ROBOT_PEER_ID.
                await self._start_session()
        elif t == "list":
            producers = msg.get("producers") or []
            names = [(p.get("meta") or {}).get("name") for p in producers]
            print(f"[reachy] list event (informational): {len(producers)} producers names={names!r}")
        elif t == "peer":
            sdp = msg.get("sdp")
            ice = msg.get("ice")
            if sdp:
                await self._handle_remote_sdp(sdp)
            if ice:
                await self._handle_remote_ice(ice)
        elif t == "endSession":
            reason = msg.get("reason")
            sid = msg.get("sessionId")
            # Central also pushes endSession for *stale* sessions — e.g.
            # a previous consumer instance (same HF identity) that died
            # without cleaning up. Those carry a sessionId that doesn't
            # match anything in this process, so we just acknowledge and
            # keep the SSE open instead of tearing down and reconnecting.
            if self._session_id is not None and sid == self._session_id:
                print(f"[reachy] endSession for our session {sid} (reason={reason!r})")
                raise RuntimeError(f"endSession: {reason}")
            print(
                f"[reachy] ignoring stale endSession sessionId={sid} "
                f"reason={reason!r} (we hold session={self._session_id!r})"
            )
        elif t == "sessionRejected":
            reason = msg.get("reason")
            logger.warning("[reachy] sessionRejected (reason=%s)", reason)
            raise RuntimeError(f"sessionRejected: {reason}")
        else:
            logger.debug("[reachy] unhandled event type=%r", t)

    @staticmethod
    def _canonical_name(name: Optional[str]) -> str:
        return (name or "").lower().replace("_", "").replace("-", "").replace(" ", "")

    def _auto_pick_producer(self, producers: list) -> None:
        target = self._canonical_name(self._robot_name)
        # Canonicalised name match — "reachy_mini", "reachy-mini",
        # "ReachyMini" all match "reachymini".
        for p in producers:
            meta = p.get("meta") or {}
            if self._canonical_name(meta.get("name")) == target:
                self._target_peer_id = p.get("id")
                logger.info(
                    "[reachy] selected robot peerId=%s name=%r transport=%r",
                    self._target_peer_id, meta.get("name"), meta.get("transport"),
                )
                return
        # Fallback: if there's only one producer for this token, use it
        # regardless of how it named itself. Pin REACHY_ROBOT_PEER_ID if
        # this guess is ever wrong.
        if len(producers) == 1:
            only = producers[0]
            meta = only.get("meta") or {}
            self._target_peer_id = only.get("id")
            logger.info(
                "[reachy] only one producer visible (name=%r) and name didn't match "
                "%r — picking it anyway. Set REACHY_ROBOT_PEER_ID to pin a specific id.",
                meta.get("name"), self._robot_name,
            )
            return
        names = [(p.get("meta") or {}).get("name") for p in producers]
        logger.info(
            "[reachy] no producer matched name=%r (canonical=%r) among %d producers %r; "
            "waiting for next list",
            self._robot_name, target, len(producers), names,
        )

    # ----------------------------------------------------------- heartbeat

    def _ensure_heartbeat_task(self) -> None:
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="reachy-heartbeat"
        )

    async def _heartbeat_loop(self) -> None:
        """Refresh our peer lease so central doesn't evict the listener.

        Central only refreshes ``last_seen`` on inbound /send (SSE pings
        don't count), so an idle listener gets swept after the lease and
        its SSE becomes a zombie. A periodic setPeerStatus keeps us
        registered.
        """
        while not self._stopping and self._sse_resp is not None:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                if self._stopping or self._sse_resp is None:
                    return
                await self._post({
                    "type": "setPeerStatus",
                    "roles": ["listener"],
                    "meta": {"name": self._consumer_label},
                })
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[reachy] heartbeat error: {e!r}")

    # ----------------------------------------------------------- discovery

    def _ensure_discovery_task(self) -> None:
        """Start the HTTP-poll discovery loop once per SSE connection."""
        if self._target_peer_id is not None:
            return
        if self._discovery_task is not None and not self._discovery_task.done():
            return
        self._discovery_task = asyncio.create_task(
            self._discovery_loop(), name="reachy-discovery"
        )

    async def _robot_status(self) -> list:
        """GET {central}/api/robot-status — the reliable producer list.

        The SSE `list` push only re-fires on producer status *changes*, so
        a robot that was already online before we connected can be absent
        from the SSE stream entirely. This HTTP endpoint always returns the
        current set, so we use it for discovery.
        """
        assert self._http is not None
        url = f"{self._central_url}/api/robot-status"
        headers = {"Authorization": f"Bearer {self._hf_token}"}
        async with self._http.get(
            url, headers=headers, timeout=aiohttp.ClientTimeout(total=10.0),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        return payload.get("robots") or []

    @staticmethod
    def _normalize_robot_status(robots: list) -> list:
        """Map /api/robot-status entries to the SSE `list` producer shape
        ({id, meta}) so `_auto_pick_producer` can consume either source."""
        out = []
        for r in robots:
            meta = r.get("meta") or {}
            if "name" not in meta and r.get("robotName"):
                meta = {**meta, "name": r.get("robotName")}
            out.append({"id": r.get("peerId") or r.get("id"), "meta": meta})
        return out

    async def _discovery_loop(self) -> None:
        """Poll /api/robot-status until we hold a session.

        Loops on `_session_id is None` (not `_target_peer_id`) so that a
        failed startSession doesn't strand us: on failure we clear the
        (unpinned) target and re-discover on the next pass.
        """
        delay = 2.0
        while not self._stopping and self._session_id is None:
            try:
                if self._target_peer_id is None:
                    robots = await self._robot_status()
                    names = [(r.get("meta") or {}).get("name")
                             or r.get("robotName") for r in robots]
                    print(f"[reachy] robot-status poll: {len(robots)} robots names={names!r}")
                    if robots:
                        self._auto_pick_producer(self._normalize_robot_status(robots))
                if self._target_peer_id is not None and self._session_id is None:
                    await self._start_session()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[reachy] discovery error: {e!r}")
                # Re-discover next pass unless the robot id was pinned.
                if self._target_peer_id_pinned is None:
                    self._target_peer_id = None
            if self._session_id is not None:
                return
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 10.0)

    # --------------------------------------------------------------- POSTs

    async def _post(self, body: dict) -> Optional[dict]:
        url = f"{self._central_url}/send"
        headers = {
            "Authorization": f"Bearer {self._hf_token}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=15.0)
        assert self._http is not None
        try:
            async with self._http.post(
                url, headers=headers, json=body, timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    txt = (await resp.text())[:200]
                    print(f"[reachy] POST /send type={body.get('type')} "
                          f"-> HTTP {resp.status}: {txt}")
                    # Central deregistered our peer but our SSE is a zombie.
                    # Force a fresh SSE connection (new welcome re-registers).
                    if resp.status == 400 and "Connect to /events" in txt:
                        self._force_sse_reconnect("peer deregistered")
                    return None
                try:
                    return await resp.json()
                except Exception:
                    return None
        except Exception as e:
            logger.warning("[reachy] POST /send type=%s failed: %r",
                           body.get("type"), e)
            return None

    async def _start_session(self) -> None:
        if self._target_peer_id is None:
            return
        # Guard set synchronously (before any await) so concurrent callers
        # can't both pass the check and double-start — that races on
        # central and gets one session killed.
        if self._session_id is not None or self._starting_session:
            return
        self._starting_session = True
        try:
            resp = await self._post({
                "type": "startSession",
                "peerId": self._target_peer_id,
            })
            if resp is None:
                raise RuntimeError("startSession: no response from central")
            t = resp.get("type")
            if t == "sessionRejected":
                raise RuntimeError(
                    f"startSession rejected: {resp.get('reason')} "
                    f"(activeApp={resp.get('activeApp')})"
                )
            if t != "sessionStarted":
                raise RuntimeError(f"startSession: unexpected response {resp!r}")
            self._session_id = resp.get("sessionId")
            print(f"[reachy] sessionStarted sessionId={self._session_id}")
            # The robot will now push an SDP offer to us via SSE.
        finally:
            self._starting_session = False

    # ------------------------------------------------------------ WebRTC

    async def _build_pc(self) -> None:
        ice_servers = await self._fetch_ice_servers()
        # Log the URL kinds we got — invaluable for diagnosing TURN issues.
        kinds = []
        for s in ice_servers:
            urls = s.urls if isinstance(s.urls, list) else [s.urls]
            for u in urls:
                scheme = (u or "").split(":", 1)[0]
                kinds.append(scheme + (" (auth)" if s.username else ""))
        print(f"[reachy] building pc with ice servers: {kinds}")
        config = RTCConfiguration(iceServers=ice_servers)
        pc = RTCPeerConnection(configuration=config)

        @pc.on("track")
        def _on_track(track):
            print(f"[reachy] received remote track kind={track.kind}")
            if track.kind == "video":
                asyncio.create_task(self._consume_video(track))
            elif track.kind == "audio" and self._on_pcm is not None:
                asyncio.create_task(self._consume_audio(track))

        @pc.on("datachannel")
        def _on_datachannel(channel):
            # The robot daemon offers a "data" channel for JSON command
            # envelopes (set_full_target / goto_target / ...). We capture
            # it and expose send_command(); we don't read telemetry yet.
            if channel.label != "data":
                print(f"[reachy] ignoring unexpected data channel: {channel.label!r}")
                return
            self._cmd_channel = channel
            # ``on("datachannel")`` fires inside aiortc's event loop, so
            # capturing here gives us the right loop for thread-safe sends.
            try:
                self._cmd_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._cmd_loop = asyncio.get_event_loop()
            print(f"[reachy] robot data channel attached (state={channel.readyState})")
            if channel.readyState == "open":
                self._cmd_channel_open = True
                self._on_cmd_ready()

            @channel.on("open")
            def _open():
                self._cmd_channel_open = True
                print("[reachy] robot data channel open")
                self._on_cmd_ready()

            @channel.on("close")
            def _close():
                self._cmd_channel_open = False

            @channel.on("message")
            def _msg(_m):
                # Telemetry from the robot (state snapshots). Ignored for now.
                pass

        @pc.on("connectionstatechange")
        async def _on_state():
            st = pc.connectionState
            print(f"[reachy] pc state: {st}")
            self._pc_connected = (st == "connected")
            if st == "failed":
                raise RuntimeError("pc connection failed")

        @pc.on("iceconnectionstatechange")
        async def _on_ice_state():
            print(f"[reachy] ice state: {pc.iceConnectionState}")

        @pc.on("icegatheringstatechange")
        async def _on_ice_gather_state():
            print(f"[reachy] ice gathering: {pc.iceGatheringState}")

        self._pc = pc
        self._remote_desc_set = False
        self._out_track_added = False
        self._pending_remote_ice = []

    async def _fetch_ice_servers(self) -> List[RTCIceServer]:
        if self._ice_servers_provider is not None:
            try:
                servers = await self._ice_servers_provider()
                if servers:
                    return servers
            except Exception as e:
                logger.warning("[reachy] ice_servers_provider failed: %r", e)
        # Fallback to the same STUN the JS SDK uses.
        return [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]

    async def _handle_remote_sdp(self, sdp_msg: dict) -> None:
        sdp_type = sdp_msg.get("type")
        sdp_text = sdp_msg.get("sdp")
        if not sdp_text or sdp_type != "offer":
            logger.warning("[reachy] unexpected sdp envelope: type=%r", sdp_type)
            return
        if self._pc is None:
            await self._build_pc()
        assert self._pc is not None
        offer = RTCSessionDescription(sdp=sdp_text, type=sdp_type)
        await self._pc.setRemoteDescription(offer)
        self._remote_desc_set = True
        # Attach our outbound audio track to the audio transceiver the robot's
        # (sendrecv) offer just created — after setRemoteDescription and before
        # createAnswer so it lands on the existing m-line instead of spawning a
        # second one.
        if self._out_track is not None and not self._out_track_added:
            try:
                self._pc.addTrack(self._out_track)
                self._out_track_added = True
                print("[reachy] added outbound audio track")
            except Exception as e:
                logger.warning("[reachy] addTrack(out) failed: %r", e)
        # Drain any ICE candidates that arrived before the offer.
        pending, self._pending_remote_ice = self._pending_remote_ice, []
        for c in pending:
            await self._add_remote_ice(c)
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        # aiortc has finished local ICE gathering at this point; the answer
        # SDP carries all of our candidates. We don't trickle outbound ICE.
        local_sdp = self._pc.localDescription.sdp
        self._log_candidate_types("local", local_sdp)
        self._remote_sdp_for_log = sdp_text
        self._log_candidate_types("remote", sdp_text)
        await self._post({
            "type": "peer",
            "sessionId": self._session_id,
            "sdp": {
                "type": self._pc.localDescription.type,
                "sdp": local_sdp,
            },
        })
        logger.info("[reachy] sent SDP answer for session %s", self._session_id)

    @staticmethod
    def _log_candidate_types(label: str, sdp: str) -> None:
        import re
        from collections import Counter
        # Candidate line: "candidate:<foundation> <comp> <proto> <prio> <ip> <port> typ <type> ..."
        types = Counter(re.findall(r"candidate:[^\r\n]* typ (\w+)", sdp or ""))
        n = sum(types.values())
        print(f"[reachy] {label} ICE candidates: {n} total by type {dict(types) or '{}'}")

    async def _handle_remote_ice(self, ice: dict) -> None:
        if not self._remote_desc_set or self._pc is None:
            self._pending_remote_ice.append(ice)
            return
        await self._add_remote_ice(ice)

    async def _add_remote_ice(self, ice: dict) -> None:
        cand_str = (ice.get("candidate") or "").strip()
        if not cand_str:
            return  # end-of-candidates marker; aiortc doesn't need it
        try:
            sdp_cand = cand_str
            prefix = "candidate:"
            if sdp_cand.startswith(prefix):
                sdp_cand = sdp_cand[len(prefix):]
            candidate = candidate_from_sdp(sdp_cand)
            candidate.sdpMid = ice.get("sdpMid")
            mline = ice.get("sdpMLineIndex")
            if mline is not None:
                candidate.sdpMLineIndex = int(mline)
            assert self._pc is not None
            await self._pc.addIceCandidate(candidate)
            # Log the type so we can see what the robot offers us.
            import re
            m = re.search(r" typ (\w+)", cand_str)
            print(f"[reachy] +remote ICE {m.group(1) if m else '?'}: {cand_str[:80]}")
        except Exception as e:
            logger.warning("[reachy] addIceCandidate failed: %r (cand=%r)",
                           e, cand_str)

    async def _consume_video(self, track) -> None:
        try:
            while True:
                try:
                    frame = await track.recv()
                except MediaStreamError:
                    print("[reachy] video track ended (MediaStreamError)")
                    return
                try:
                    arr = frame.to_ndarray(format="rgb24")
                except Exception as e:
                    print(f"[reachy] frame decode failed: {e!r}")
                    continue
                # Atomic reference swap of the (id, ndarray) tuple — readers
                # see either the prior pair or this one, never a mismatch.
                self._frame_counter += 1
                self._latest = (self._frame_counter, arr)
                # Source resolution as the daemon's webrtcsink emits it
                # (h, w) — consumers read this to pick the right camera
                # crop_scale when computing look-at geometry.
                if (
                    self._source_size is None
                    or self._source_size != (arr.shape[1], arr.shape[0])
                ):
                    self._source_size = (arr.shape[1], arr.shape[0])
                    print(
                        f"[reachy] source frame size: "
                        f"{self._source_size[0]}x{self._source_size[1]}"
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[reachy] video consumer error: {e!r}")

    async def _consume_audio(self, track) -> None:
        """Decode the robot mic track to mono float32 at ``audio_sample_rate``
        and hand each chunk to ``on_pcm``.

        aiortc delivers Opus as 48 kHz s16 ``av.AudioFrame``s; the resampler
        downmixes to mono and converts to ``flt`` at the requested rate.
        """
        resampler = av.audio.resampler.AudioResampler(
            format="flt", layout="mono", rate=self._audio_sample_rate,
        )
        try:
            while True:
                try:
                    frame = await track.recv()
                except MediaStreamError:
                    print("[reachy] audio track ended (MediaStreamError)")
                    return
                try:
                    for out in resampler.resample(frame):
                        # "flt"/mono → ndarray shape (1, samples); flatten.
                        pcm = out.to_ndarray().reshape(-1).astype(
                            np.float32, copy=False,
                        )
                        if pcm.size == 0:
                            continue
                        self._audio_frames += 1
                        self._pcm_samples += int(pcm.size)
                        if self._on_pcm is not None:
                            self._on_pcm(pcm)
                except Exception as e:
                    print(f"[reachy] audio frame decode failed: {e!r}")
                    continue
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[reachy] audio consumer error: {e!r}")

    # ----------------------------------------------------------- teardown

    async def _teardown_session(self) -> None:
        # Best-effort endSession to central before tearing down the pc.
        # Only if we actually held a session AND the SSE is still alive
        # — otherwise central rejects with HTTP 400 "Connect to /events
        # first" and that's just noise in the logs.
        if (
            self._session_id is not None
            and self._http is not None
            and not self._http.closed
        ):
            try:
                resp = await self._post({
                    "type": "endSession",
                    "sessionId": self._session_id,
                })
                if resp is None:
                    # POST already logged its own warning, no need to echo.
                    pass
            except Exception:
                pass
        self._session_id = None
        self._starting_session = False
        self._remote_desc_set = False
        self._pending_remote_ice = []
        self._pc_connected = False
        if self._pc is not None:
            try:
                await self._pc.close()
            except Exception:
                pass
            self._pc = None
        # If the robot peerId wasn't pinned via env, force a fresh pick on
        # the next list event in case the robot reconnected with a new id.
        if self._target_peer_id_pinned is None:
            self._target_peer_id = None


def from_env(
    ice_servers_provider: Optional[Callable[[], Awaitable[List[RTCIceServer]]]] = None,
) -> Optional[ReachyCentralConsumer]:
    """Build a consumer from environment variables.

    Returns ``None`` if ``HF_TOKEN`` is not set — the caller is expected to
    surface that as a configuration error to the user.
    """
    hf_token = (os.getenv("HF_TOKEN") or "").strip()
    if not hf_token:
        logger.warning(
            "[reachy] HF_TOKEN is not set; ReachyCentralConsumer cannot start"
        )
        return None
    central_url = (
        os.getenv("REACHY_CENTRAL_URL") or DEFAULT_CENTRAL_URL
    ).strip()
    robot_peer_id = (os.getenv("REACHY_ROBOT_PEER_ID") or "").strip() or None
    robot_name = (os.getenv("REACHY_ROBOT_NAME") or "reachymini").strip()
    return ReachyCentralConsumer(
        hf_token=hf_token,
        central_url=central_url,
        robot_peer_id=robot_peer_id,
        robot_name=robot_name,
        ice_servers_provider=ice_servers_provider,
    )
