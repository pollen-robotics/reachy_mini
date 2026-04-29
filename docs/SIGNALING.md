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

`SIGKILL`, panic, kernel oops, WiFi cable yanked, Pi power loss. The daemon
gets no chance to send `roles=[]`. Coverage falls to **central-side TTL**:

- Central tracks `last_seen: float` per peer, refreshed by:
  - any incoming `POST /send` from that peer, or
  - SSE heartbeat round-trip (server `ping` -> client TCP ack reaches OS).
- A background sweeper task scans every 10 s. Any peer with
  `now - last_seen > LEASE_SECONDS` (default 45 s) is `disconnect_peer()`'d
  (= removed from `peers`, `producers`, `token_to_peer`, sessions ended).
- `request.is_disconnected()` inside the SSE generator stays as a
  fast-path: if the OS already noticed the dead socket, we evict in <30 s
  without waiting for the lease.

Result: under any failure mode, a peer disappears from `/api/robot-status`
within `LEASE_SECONDS`.

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
- Mobile: hides any row whose only present channel is central + `health=error`.
- Mobile: keeps a row visible if BLE or mDNS sees it locally even when
  central marks it `error`.
