# libnice abort on session reuse (consumer peer ID stickiness)

- **Status**: open, no fix shipped
- **Severity**: high (kills the daemon process every few sessions on Wi-Fi/central path)
- **Affects**: every WebRTC client going through HF central (mobile app, web SDK, conversation app on Spaces, any future client). USB/loopback path is immune.
- **First observed**: 2026-04-30, on `reachy-mini.local` (192.168.1.19), `libnice10 0.1.22-1`, `gstreamer1.0-nice 0.1.22-1` (Debian 12 arm64).
- **Triage author**: investigation captured during a chat session, see `agent-transcripts` for the live trace.

## Symptom

From the user's point of view: the iPhone app connects to a Wi-Fi robot the first 1-3 times, then a connection attempt to **the same robot, shortly after disconnecting**, fails with the conversation engine reporting `Session stopped` and a flood of `webrtc.proxy fetch.no_dc` warnings. The robot disappears from the central list for ~5-10 seconds, then comes back and the next attempt may succeed.

USB connections to the same robot (or to the Mac tray daemon) are never affected.

## Diagnostic

The daemon process aborts in the middle of ICE negotiation:

```
libnice:ERROR:../agent/conncheck.c:987:priv_conn_check_tick_stream_nominate:
  assertion failed: (p->state == NICE_CHECK_SUCCEEDED)
Bail out!
reachy-mini-daemon.service: Main process exited, code=exited, status=134/n/a
```

`SIGABRT` from a `g_assert_cmpint` inside libnice's connectivity check nomination loop. systemd restarts the daemon, which forces a brand new central-relay peer ID. The next session works again, until the next accumulation cycle.

### Reproducer (on `reachy-mini.local`, 2026-04-30)

```
10:12:58 -> 10:13:13   session 1   consumer afe0fd7c   OK
10:14:16 -> 10:14:29   session 2   consumer afe0fd7c   OK
10:16:44 -> 10:16:59   session 3   consumer afe0fd7c   OK
10:17:19 -> 10:17:20   session 4   consumer afe0fd7c   abort  <-- libnice crash
10:18:04 -> 10:18:32   session 5   consumer 3988be23   OK     (post-restart, fresh consumer)
```

The pattern is consistent: every session enqueued under the **same GStreamer signaling consumer peer ID** eventually trips the libnice assertion. After systemd restarts the daemon, the central relay reconnects and gets a new consumer ID, resetting the counter.

The exact ICE candidate that triggered the crash on session 4:

```
candidate:1592251543 1 udp 1677732095
  2a01:cb11:482:9d00:9dc7:c52e:301:d90e 56290 typ srflx
  raddr 2a01:cb11:482:9d00:9dc7:c52e:301:d90e rport 56290
```

That is a self-mapped IPv6 srflx (`raddr == candidate addr`), valid by spec but rare; libnice's nomination logic walks a check pair that is in `NICE_CHECK_FROZEN`/`NICE_CHECK_IN_PROGRESS` instead of the expected `NICE_CHECK_SUCCEEDED`. The bug is **state pollution carried across sessions** under the same consumer, not the candidate per se.

## Root cause (hypothesis)

`gst-plugin-webrtc-rs` keeps the GST signaling consumer registration alive across sessions for the same `central_signaling_relay` connection, and the `webrtcbin`/`niceagent` lifecycle in `reachy_mini.media.media_server` does not fully tear down libnice's per-stream state between consumer-added/consumer-removed cycles. Some `NiceComponent`/`NiceStream` handles or check-pair entries leak forward into the next session, and the nomination tick eventually walks a stale entry.

This is consistent with:

- Only the central path is affected (loopback path uses `aiortc`, no libnice).
- A fresh consumer (after systemd restart) always works on session 1.
- Failures cluster after multiple back-to-back sessions, not from cold start.

## Mitigations, ranked by where the fix lives

| # | Where | Effort | Pérennité | Notes |
|---|---|---|---|---|
| 1 | mobile app SDK shim | XS | weak | rotate the signaling websocket and force a new central peer ID before every `startSession()`. Hides the bug for the mobile app only; every other client (Python SDK, web SDK, HF Space conv app) stays vulnerable. Useful as a stop-gap. |
| 2 | daemon `media_server.py` | M | strong | strict per-session lifecycle: explicit `set_state(NULL)` + `unref` of `webrtcbin` + drop the `niceagent` on `consumer removed`, and instantiate brand new objects on `consumer added`. This is the *real* fix. |
| 3 | daemon process supervision | M | strong (defense-in-depth) | run the GStreamer pipeline in a dedicated subprocess supervised by the daemon. A libnice abort kills only that subprocess, not the API/BLE/relay/lock. Recovery is ~200 ms. Doesn't fix the root cause but bounds the blast radius for the entire class of C-side bugs. |
| 4 | OS package | L | external | upgrade `libnice10` past 0.1.22 (the upstream `priv_conn_check_tick_stream_nominate` codepath has churned in master). Heavy: needs a custom Debian build for arm64 and a daemon-image update flow. |

The **honest** robust answer is **2 + 3 together**, in the daemon repo, `reachy_mini/src/reachy_mini/media/`. 2 fixes the cause; 3 prevents the next equivalent native-side bug from taking down the daemon.

The mobile-app workaround (option 1) is acceptable as an interim mitigation, but should be tagged with a `TODO` referencing this document so it gets removed once 2 lands.

## Suggested implementation outline (option 2)

In `reachy_mini/src/reachy_mini/media/media_server.py`, audit the consumer-added / consumer-removed handlers:

1. Confirm a new `webrtcbin` is created per session (currently appears to be the case per the GStreamer logs).
2. Verify the `webrtcbin` is set to `Gst.State.NULL` and **removed from the pipeline** on `consumer removed`, before any new consumer is accepted on the same registration.
3. Drop any cached `Gst.WebRTCBin`, `Gst.WebRTCRTPTransceiver`, audio/video pads, and queue elements created for the session. Run with `GST_DEBUG=GST_REFCOUNTING:5` once to make sure refcount drops to 0 (no leaked pad / probe / pipeline-level reference holding `niceagent` alive).
4. If refcounts can't be brought to zero cleanly (custom probes, pad handlers), tear down and recreate the parent `pipeline` element on every consumer-removed. Slower (~50-100 ms cold start for GStreamer elements), but bullet-proof.

Reproduction harness: on a Wi-Fi-connected robot, use a Python script that opens a `ReachyMini` session via central, sleeps 200 ms, closes, and loops. The current build aborts within 4-10 iterations.

## Suggested implementation outline (option 3)

Move the GStreamer media path into a dedicated `media-pipeline` subprocess (multiprocessing or `asyncio.subprocess`). The main daemon (FastAPI + BLE + central relay + robot lock) speaks to it over a Unix socket / pipe. Crashes in the subprocess are caught by the supervisor, which restarts it, re-attaches to the local GST signaling server, and emits a `central.relay.state` transition through `connecting` so the app shows a brief "reconnecting" toast instead of "robot offline".

This also gives us a clean isolation boundary for any future native dependency we pull in (rust audio plugins, custom encoders, etc.).

## Stop-gap (option 1) for the mobile app

Until the daemon-side fix is shipped, in `reachy_mini_mobile_app/src/conversation/useReachySdk.ts` (or in the vendored `reachy-mini.js`):

- Before each `startSession()`, force the SDK to reconnect its central WebSocket so the central server assigns a new consumer peer ID.
- Mark the patch `// TODO(known-issue: libnice-session-reuse-crash) remove once daemon fix lands`.

This does not change behavior on USB/loopback (no central peer ID involved) and only adds ~200 ms to the second-and-onward connection.

## References

- libnice tracker: https://gitlab.freedesktop.org/libnice/libnice/-/issues (search for `priv_conn_check_tick_stream_nominate`).
- GStreamer plugin: `gst-plugins-rs/net/webrtc` (the producer/consumer signaling we use).
- Daemon entry points relevant to this bug:
  - `reachy_mini/src/reachy_mini/media/media_server.py` (`consumer added` / `consumer removed` handlers)
  - `reachy_mini/src/reachy_mini/media/central_signaling_relay.py` (central session mapping)
- App-side observability: `[ReachyMini trace] dc.open` / `webrtc.proxy fetch.no_dc` in `useReachySdk.ts` and `webrtcClient.ts`.
- Live evidence captured 2026-04-30: see chat transcript [libnice abort on Wi-Fi reuse](fb11cb70-58e2-4ce2-bba1-b6d7fd7944c9).
