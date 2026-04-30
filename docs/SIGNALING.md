# Reachy Mini Signaling Lifecycle

> Single source of truth for **how a Reachy Mini instance announces itself**,
> stays announced, and disappears across the central server, BLE, mDNS and
> loopback HTTP discovery channels.
>
> If you change anything in `central_signaling_relay.py`, the central server
> (`reachy_mini_central/app.py`) or the mobile/desktop client's discovery
> aggregator, **read this first** and update it after.

## Three layers, three jobs

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer       Owner of                                Lives in        │
├──────────────────────────────────────────────────────────────────────┤
│  Daemon      "Am I a healthy robot, what can I do?"  reachy_mini     │
│  Central     "Who is online RIGHT NOW for user X?"   reachy_mini_central │
│  Client      "Show one row per physical robot"       mobile / tray / web │
└──────────────────────────────────────────────────────────────────────┘
```

The daemon never decides what a client should display. The central never
decides whether a robot is healthy. The client never invents fields.
Each layer is allowed to add new fields - never to override another layer's.

## The `meta` payload (stable contract, additive only)

The daemon sends this with every `setPeerStatus(roles=["producer"])`. Central
forwards it verbatim to listeners. Client reads it without re-interpreting.

```jsonc
{
  "schema_version": 1,        // bump only on breaking field semantics

  // Identity (set once per install, survives reboots)
  "name": "reachy_mini",      // user-facing label, mutable via UI rename
  "install_id": "639d5558364f41d6b39ffe5be1466793",
  "kind": "robot",            // "robot" | "tray"  - what runs the daemon

  // Health (refreshed every status tick, debounced 1 s)
  "health": "ok",             // "ok" | "degraded" | "error"
  "error_code": null,         // short stable taxonomy when health != "ok",
                              //   e.g. "motor_comm" | "no_backend" | "media"

  // Optional capabilities the client can light up against
  "capabilities": ["motion", "audio", "camera"],
  "wireless_version": true,
  "version": "1.7.0"
}
```

Rules:

- **Add fields freely** - older clients ignore unknown keys.
- **Never repurpose a field**. If semantics change, bump `schema_version` and
  add a new field; clients then branch on the version.
- **Never leak transient data** (current motor angles, RTT, etc.). Heartbeat
  density belongs to ping/pong, not `meta`.

### `health` semantics

| Value | When | Client policy |
|-------|------|---------------|
| `ok` | Backend ready, motors enabled, no error in `_status`. | Listed normally. Auto-connect allowed. |
| `degraded` | Daemon up, backend up, but a non-critical subsystem is down (camera failed, audio failed). | Listed with a yellow badge. Auto-connect allowed; client surfaces a warning if the missing capability matters to its current screen. |
| `error` | `state == ERROR` or `backend_status is None` or `backend.ready` not set after grace period. | Listed with a red badge. **Never auto-connect.** Tap shows the error code so the user can act (replug USB, check power, retry). |

## Lifecycle - cooperative path (the daemon is alive)

Daemon transitions emit explicit central messages.

```
boot                                  shutdown
 │                                     ▲
 ├─ start_central_relay()              │
 ├─ welcome from central               │
 ├─ setPeerStatus(roles=["producer"],  │
 │     meta={..., health: H1})         │
 │                                     │
 ├─ on health change (debounced 1s):   │
 │     setPeerStatus(roles=["producer"], meta={..., health: H2})
 │                                     │
 ├─ on tray + no_backend > 30s:        │
 │     setPeerStatus(roles=[])  ─→ central drops from producers list
 │     (peer object kept on central, can re-register)
 │                                     │
 ├─ on backend recovery (tray):        │
 │     setPeerStatus(roles=["producer"], meta={..., health: "ok"})
 │                                     │
 └─────────────── SIGTERM / SIGINT / lifespan finally ───┘
                  │
                  ├─ withdraw():       setPeerStatus(roles=[])  // best-effort, 500ms timeout
                  ├─ stop():           close central WebSocket
                  └─ central observes WS close → disconnect_peer()
```

Two distinct primitives:

- `relay.withdraw()` - send `setPeerStatus(roles=[])`, wait for ack with
  short timeout, **leave the WebSocket open**. Used for "I'm here but no
  longer want to be listed" (backend lost, hardware unplugged, transient
  condition that may resolve).
- `relay.stop()` - send `roles=[]` THEN close the WebSocket. Used for
  shutdown.

## Lifecycle - uncooperative path (daemon dies without warning)

`SIGKILL`, panic, kernel oops, WiFi yanked, Pi power loss. The daemon gets
no chance to send `roles=[]`. Coverage falls to **central-side TTL**, driven
by an explicit application-level heartbeat.

### Why server-pushed keepalives are NOT a liveness signal

A naive design refreshes `last_seen` whenever the central yields a keepalive
ping on the SSE channel. **That doesn't work** behind HTTP/2 reverse proxies
(HF Spaces, Cloudflare, etc.): the local TCP send buffer absorbs writes
silently on a half-open socket, so `yield` returns without raising for
several minutes after the peer's network has died. With a touch on every
yield, the sweeper would never evict the zombie.

### Heartbeat protocol

- The daemon re-emits `setPeerStatus(roles=["producer"], meta=…)` every
  `self._heartbeat_interval_seconds` **even if the meta payload is
  identical**. This arrives at central as a real `POST /send`, which is the
  only authoritative liveness signal we accept.
- Meta deltas (health flips from `ok` to `degraded`, name change, …) are
  emitted immediately on the daemon's 1 Hz status tick, not gated on the
  heartbeat. So health transitions still propagate within 1 s.
- Central's `signaling.touch(peer_id)` is called **only** on:
  - `POST /send` arriving from the peer (covers heartbeat + applicative),
  - a session message successfully delivered through its message queue.
- The SSE keepalive yield (`{"event": "ping"}`) is purely a proxy
  heartbeat to keep the HTTP/2 connection from being culled by the
  reverse proxy. It does **not** touch `last_seen`.

### Heartbeat negotiation (server-driven)

The heartbeat interval is **negotiated at handshake** so server operators
can tune the liveness contract from one place (`LEASE_SECONDS` env var on
the central Space) and every connected daemon auto-aligns on its next
reconnect, with **zero coordinated daemon redeploy**.

- Central's `welcome` SSE frame carries:
  ```json
  {
    "type": "welcome",
    "peerId": "...",
    "username": "...",
    "lease_seconds": 15,
    "recommended_heartbeat_interval_seconds": 5
  }
  ```
- The daemon stores the recommended value into
  `self._heartbeat_interval_seconds`, clamped to
  `[MIN_HEARTBEAT_INTERVAL_SECONDS, MAX_HEARTBEAT_INTERVAL_SECONDS]`
  (`[1.0, 60.0]`) as a defence against a malformed welcome.
- If the field is missing (older central, partial deploy), the daemon
  keeps the module-level default
  (`HEARTBEAT_INTERVAL_SECONDS`, currently `5.0`). Full backwards
  compatibility.
- `recommended_heartbeat_interval_seconds = LEASE_SECONDS / 3` by
  convention so a daemon tolerates two consecutive missed POSTs before
  hitting the lease.

### Eviction guarantees

- Background sweeper scans every `SWEEPER_INTERVAL_SECONDS` (default
  3 s). Any peer with `now - last_seen > LEASE_SECONDS` (default 15 s)
  is `disconnect_peer()`'d (= removed from `peers`, `producers`,
  `token_to_peer`, sessions ended).
- `request.is_disconnected()` inside the SSE generator stays as a
  fast-path: if the OS already noticed the dead socket (FIN/RST), we evict
  in <30 s without waiting for the lease.
- Worst case for an uncooperative death: `LEASE_SECONDS + sweeper_interval`
  ≈ 18 s. With heartbeat=5 s and lease=15 s we tolerate two consecutive
  missed heartbeats before eviction kicks in.

### Tuning the contract

To shrink the staleness window further (or open it to absorb more
network instability), change **only** `LEASE_SECONDS` on the central
Space, then redeploy the central. Daemons re-derive their heartbeat
interval on the next reconnect; nothing else needs to change. Keep the
ratio `lease : heartbeat ≥ 3 : 1` so a single missed POST never causes
eviction.

### Cooperative withdraw on graceful network loss

Some user-initiated actions (`POST /wifi/forget`, `POST /wifi/forget_all`)
are about to take the daemon offline by wiping its only network. The daemon
calls `notify_withdraw(timeout=1s)` **while still on WiFi** so central
removes the producer instantly, instead of leaving the user staring at a
ghost row for ~55 s. Best-effort: if the withdraw POST fails for any
reason we still proceed with the WiFi reset and rely on TTL eviction.

## install_id - the anchor across channels

Every channel publishes `install_id`:

| Channel | Field | Example |
|---------|-------|---------|
| Central | `meta.install_id` | full UUID4 hex |
| BLE GATT | TLV in advertisement payload | first 8 bytes (16 hex) |
| mDNS TXT | `install_id=` | full UUID4 hex |
| Loopback HTTP | `/api/daemon/identity` JSON | full UUID4 hex |

Client merges by **prefix match on the truncated form** when comparing a BLE
sighting with a central row, by **full match** otherwise.

### Collision policy

A daemon flashed onto a different robot keeps its `install_id`. A daemon
re-installed on the same robot generates a new `install_id`. So **two
producers with the same `install_id` and the same HF user can only mean
one of them is stale**.

Central applies **last-writer-wins**: when a new producer registers with an
`install_id` already held by another producer of the same user, central
evicts the older one (`disconnect_peer`) before accepting the new one. Any
session of the old producer is ended with `reason="install_id_takeover"`.

## What lives where (responsibility cheat-sheet)

| Question | Answer |
|----------|--------|
| Is this robot reachable on the LAN right now? | Client (probes BLE / mDNS / loopback). |
| Is this robot's hardware healthy? | Daemon (`peer_health.compute()`) → `meta.health`. |
| Should we hide a robot row from the user? | Client (gates on `meta.health` + presence on at least one channel). |
| Is the central listing stale? | Central (TTL sweeper). |
| Two listings for the same physical robot? | Client (dedup by `install_id`). |
| Two daemons fighting over one `install_id`? | Central (last-writer-wins eviction). |

When in doubt, ask: "if a different team owned this layer, would they
expect to see this code?" If no, it's in the wrong layer.

## Observability (debugging the staleness window)

Operators can introspect the running contract without redeploying:

| Endpoint | Auth | What it tells you |
|----------|------|-------------------|
| `GET /health` | none | Active `lease_seconds`, `sweeper_interval_seconds`, `recommended_heartbeat_interval_seconds`, raw counts. Cheap, public. |
| `GET /api/robot-status` | HF Bearer | Per producer: `last_seen_age_seconds` (seconds since the last inbound POST). Plus `lease_seconds` mirrored at the top level. Caller-filtered. |
| `GET /api/debug/peers` | HF Bearer | Owner-filtered dump of *every* peer (producers AND consumers): role, connected, session, full meta, `last_seen`, `last_seen_age_seconds`. Use this when a robot does not show up where expected. |

On the daemon side:

| Surface | What it tells you |
|---------|-------------------|
| `get_relay_status()` (Python) / `/api/daemon/status` | `state`, `is_connected`, plus the negotiated `heartbeat_interval_seconds` and `central_lease_seconds` so you can confirm the contract the daemon is currently honouring. |
| Central relay startup log | `[Central Relay] central welcome received peer_id=… heartbeat=5.0s lease=15.0s; …` — pin the active interval to a wall-clock at handshake time. |

Recommended client guard for "is this row really fresh?":

```ts
const STALE_GRACE = 0.6; // tighter than the server-side sweeper
const isFresh = robot.last_seen_age_seconds < lease_seconds * STALE_GRACE;
```

This lets the mobile/desktop UI hide a row 40% earlier than the
server-side eviction would, without depending on the sweeper interval.

## Versioning & migration

- `schema_version` is incremented when **field semantics** break. Adding a
  field never bumps it.
- Central never inspects `schema_version`; it forwards verbatim. Only the
  client branches on it.
- When bumping, support `version - 1` in clients for at least one release
  cycle so out-of-date daemons remain usable.

## Tests we keep green

- `peer_health.compute(...)` truth table (24 cases).
- Central: `setPeerStatus(roles=[])` removes from `producers`, broadcasts
  `peerStatusChanged` to other listeners of the same user.
- Central: install_id collision evicts older, ends its session.
- Central: TTL sweeper purges a peer that stops sending after
  `LEASE_SECONDS`.
- Central: SSE keepalive ping does NOT refresh `last_seen` (regression
  guard for the half-open zombie bug).
- Central: `POST /send` (carrying e.g. a heartbeat-shaped
  `setPeerStatus`) DOES refresh `last_seen`.
- Daemon relay: `update_producer_meta()` re-emits with identical meta
  once the heartbeat interval elapses.
- Mobile: hides any row whose only present channel is central + `health=error`.
- Mobile: keeps a row visible if BLE or mDNS sees it locally even when
  central marks it `error`.
