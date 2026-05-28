# Cloud-Backend Consumer

The Reachy Mini daemon publishes its camera + a JSON command channel
through the HF *central signaling relay* at
`https://pollen-robotics-reachy-mini-central.hf.space`. That's what the
JS SDK (`reachy-mini-sdk.js`) uses to let a browser app talk to a robot
over WebRTC without any direct LAN connection.

`reachy_mini.media.central_consumer.ReachyCentralConsumer` is the
**Python counterpart**: a hardware-free aiortc client that lets a
cloud backend (HF Space, Cloud Run, any Python service) consume a
robot's camera stream and drive the robot the same way the browser
SDK does.

It is the natural opposite of
`reachy_mini.media.central_signaling_relay` (which the daemon runs as a
*producer*).

## When to use it

- You're building an HF Space (or any cloud service) that needs to see
  what a visitor's Reachy Mini sees and react to it — e.g. RF-DETR
  inference, MediaPipe, an LLM with a vision tool.
- You want the cloud backend to **own** the WebRTC peer to the robot
  (rather than the visitor's browser). This avoids dual-peer races —
  the central relay only allows one consumer per robot at a time, and
  having two of them fight ends in `sessionRejected: robot busy`.
- You don't have, or don't want, the daemon or GStreamer installed on
  the backend container.

If your code runs on the same machine as the daemon, prefer
`ReachyMini(media_backend="local")` or the WebRTC client — they're
faster (no central round-trip) and richer (audio + DoA, etc.).

## Quick start

```python
import asyncio
import numpy as np
from reachy_mini.media.central_consumer import ReachyCentralConsumer


async def main():
    consumer = ReachyCentralConsumer(
        hf_token=hf_bearer_token,         # visitor's short-lived HF token
        robot_peer_id=visitor_robot_peer, # pin to a known robot
        consumer_label="my-space/visitor-1",
    )
    await consumer.start()
    try:
        # Pull the latest frame whenever you need it.
        snap = consumer.latest_frame()
        if snap is not None:
            frame_id, rgb = snap  # rgb: HxWx3 uint8
            ...

        # Drive the robot. Wire format is the same as `set_full_target`
        # / `goto_target` accepted by the daemon's WebRTC data channel —
        # exactly what `ReachyMini.set_target` / `goto_target` send.
        consumer.send_command({
            "type": "goto_target",
            "head": [...16-float row-major pose matrix...],
            "duration": 0.4,
            "body_yaw": None,
            "antennas": None,
        })
    finally:
        await consumer.stop()


asyncio.run(main())
```

The full runnable demo (with brightness logging + a head nod) lives at
[`examples/central_consumer_cloud_backend.py`](../../../examples/central_consumer_cloud_backend.py).

## Per-visitor isolation

A typical HF Space serving many visitors should build **one consumer
per visitor session**. Each visitor logs into HF via the
[SDK host shell](javascript-sdk.md#mounthost) and the iframe forwards
`{hfToken, robotPeerId}` to your backend's session-open endpoint.
The backend constructs a `ReachyCentralConsumer` pinned to that
visitor's robot — the visitor's token authorizes the central relay,
so visitors only see their own robots.

## What it handles for you

- **SSE channel** to `GET /events` (with auto-reconnect on close).
- **Session negotiation**: `setPeerStatus` → `startSession` →
  `peer` (offer ⇄ answer) → trickle-ICE replay. Robot creates the
  offer; we answer. aiortc gathers all our local ICE candidates
  before `setLocalDescription` returns, so the answer SDP is complete
  on the first send.
- **Heartbeat**: refreshes `setPeerStatus` every ~10 s (interval
  derived from central's `welcome` lease) so the relay doesn't evict
  the listener.
- **Stale-relay recovery**: if a POST returns `"Connect to /events
  first"` (the signature of an eviction), the consumer force-closes
  the SSE response so the reader reopens it. The session continues.
- **Robot data channel capture**: the daemon offers a `"data"`
  bidirectional channel; `ReachyCentralConsumer` attaches and exposes
  `send_command(envelope)` for outbound JSON.
- **Thread-safe `send_command`**: callers running in worker threads
  (e.g. a GPU pipeline) can call it directly. The actual
  `RTCDataChannel.send` is marshalled onto aiortc's event loop via
  `loop.call_soon_threadsafe`.

## TURN configuration

By default the consumer asks Google's public STUN server, which is
enough whenever the robot daemon already offers a TURN-relay
candidate of its own (see the [Media Server's
`_apply_turn_servers`](../../../src/reachy_mini/media/media_server.py)).
If you need the consumer to allocate its own relay (e.g. when neither
side can hole-punch), pass an `ice_servers_provider` returning
`aiortc.RTCIceServer` objects with TURN credentials. The HF-hosted
Cloudflare TURN proxy at `https://turn.fastrtc.org/credentials` is a
reasonable default for HF Spaces:

```python
async def _ice():
    async with aiohttp.ClientSession() as s:
        r = await s.get(
            "https://turn.fastrtc.org/credentials",
            headers={"Authorization": f"Bearer {space_secret_hf_token}"},
            params={"ttl": 600},
        )
        payload = await r.json()
    return [RTCIceServer(urls=s["urls"], username=s.get("username"),
                         credential=s.get("credential"))
            for s in payload.get("iceServers", [])]


consumer = ReachyCentralConsumer(
    hf_token=visitor_token,
    robot_peer_id=visitor_robot_peer,
    ice_servers_provider=_ice,
)
```

> **Heads up**: aiortc's TURN client (`CHANNEL_BIND`) is currently
> broken against Cloudflare's TURN. The recommended setup is to leave
> the consumer on STUN-only and let the **robot daemon** offer a
> relay candidate — that path goes through GStreamer's TURN client,
> which works. The patched daemon in `reachy_mini_wt_turn` already
> does this in `media_server._apply_turn_servers`.

## Sending commands

`send_command(envelope)` accepts any of the JSON envelopes the daemon's
WebRTC data channel accepts, defined in
`reachy_mini.io.protocol`. Most useful for cloud backends:

- `{"type": "set_full_target", "head": [...], "antennas": [...], "body_yaw": ...}` —
  immediate (no daemon-side interpolation). Suitable for ~50–100 Hz
  streaming control.
- `{"type": "goto_target", "head": [...], "duration": ..., ...}` —
  smooth daemon-side interpolation; the right choice for one-shot
  movements (e.g. "look at this object").

The `head` field is a flat 16-float row-major 4×4 pose matrix.
`reachy_mini.ReachyMini.set_target` and `goto_target` build the same
matrix internally — see `reachy_mini.reachy_mini.look_at_world` for
how to construct one from a target point in world coordinates.
