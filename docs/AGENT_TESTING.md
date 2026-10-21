# Testing daemon changes (for agents and contributors)

This guide is for changes to the **daemon itself**: the REST API, the
JSON-RPC relay, the app manager, media and the signalling relay. Testing an
app you built *on* Reachy Mini is a different job, covered in
[`skills/testing-apps.md`](../skills/testing-apps.md) and
[`skills/debugging.md`](../skills/debugging.md).

It covers the loop end to end:
- unit tests;
- putting a branch on a real robot;
- watching the robot;
- driving the paths that need a remote client.

Everything here was used to test the Python-apps relay (#1477) on a
Reachy Mini Wireless.

The phone side (Android build, driving the mobile app from an agent) lives
in the mobile app repo:
[`reachy_mini_mobile_app/docs/AGENT_TESTING.md`](https://github.com/pollen-robotics/reachy_mini_mobile_app/blob/main/docs/AGENT_TESTING.md).

## 1. Unit tests

```bash
uv run pytest                                        # everything that runs on this machine
uv run pytest tests/unit_tests/test_jsonrpc_relay.py # one file
uv run pytest -k "update and race"                   # by name
```

Markers select hardware-bound tests (`pyproject.toml`): `audio`,
`respeaker`, `loopback`, `video`, `wireless`, `webrtc`. CI runs
`-m 'not audio and not video and not wireless and not webrtc'`, plus
separate audio/video and webrtc jobs. Run the matching marker yourself when
you touch that hardware path.

Before claiming a fix, do two things:
- **Prove the regression test bites.** Stash the source fix, run the new
  test and watch it fail, then restore:
  ```bash
  git stash push -- src/reachy_mini/daemon/jsonrpc_relay.py
  uv run pytest tests/unit_tests/test_jsonrpc_relay.py -k concurrent
  git stash pop
  ```
- **Fakes must model the real states.** The relay tests use a
  `FakeAppManager`. A race fix passed against a fake whose
  `stop_current_app` returned instantly. It failed on the robot within
  seconds, because there an app being stopped is `stopping`: it still
  holds the slot (`is_app_running()` is true) and a second stop raises
  `RuntimeError`. When a test is about concurrency, give the fake the same
  intermediate states and errors as `AppManager`, then confirm on a robot.

Lint and types, as CI runs them:

```bash
uv run ruff check src/ && uv run ruff format --check src/ && uv run mypy src/
```

## 2. Putting a branch on a robot

The robot runs the daemon from a venv as a systemd service. To test a
branch:
1. Push it, since pip pulls from GitHub.
2. Install it into that venv and restart the service.

No wheel to build and no backup to take.

```bash
ssh pollen@reachy-mini.local
source /venvs/mini_daemon/bin/activate
pip install "git+https://github.com/pollen-robotics/reachy_mini.git@<branch>"
sudo systemctl restart reachy-mini-daemon
```

Checks:

```bash
SP=/venvs/mini_daemon/lib/python3.12/site-packages
cat $SP/reachy_mini-*.dist-info/direct_url.json   # requested_revision + commit_id
curl -s localhost:8000/api/daemon/status          # "state": "running"
curl -s localhost:8000/api/hf-auth/relay-status   # "is_connected": true (remote access up)
```

Gotchas:
- **Same version, no reinstall.** A branch with the same version string
  as what's installed (any branch off the same `main`) can be silently
  skipped by pip. Compare `commit_id` in `direct_url.json` after the
  install. If it didn't change, re-run with
  `pip install --force-reinstall --no-deps "git+…@<branch>"`.
- **Restarting is enough.** `sudo systemctl restart` works after a pip
  install; the service doesn't need re-creating.
- **Sign-in survives.** The robot's Hugging Face sign-in (remote access)
  survives reinstalls. It lives in
  `~/.cache/huggingface/reachy_mini_daemon_credentials.json`. If
  `relay-status` says it's not connected because there's no token, sign
  the robot in again:
  1. `POST /api/hf-auth/oauth/device/start`, which returns a `user_code`.
  2. The account owner approves it at <https://hf.co/oauth/device>.
- **Keep the robot quiet** while testing:
  `curl -s -X POST localhost:8000/api/volume/set -H 'Content-Type: application/json' -d '{"volume": 0}'`.
- **Someone else's daemon.** The robot may be running another person's
  branch or work in progress. Check `direct_url.json` and say what you're
  replacing.

## 3. Watching the robot

Logs:

```bash
journalctl -u reachy-mini-daemon -f -o short-iso | grep -v -E "Pong|Ping|setPeerStatus"
```

Useful greps:

| What | Pattern |
|---|---|
| App lifecycle | `manager.runner - INFO` (Starting / Stopping / App stopped) |
| App stdout/stderr | `manager.runner - WARNING` / `ERROR` (the app's own output) |
| Install / remove / update jobs | `logs_job_<id>`, `Job '<command>' completed successfully` / `failed` |
| Robot slot ownership | `RobotAppLock:` (`remote_session`, `local_app`, `free`) |
| Remote sessions | `[Central Relay]` (`Received session request`, `Session ended from central/local`) |
| WebRTC health | `[watchdog]`, `connection-state ->`, `Data channel opened/closed` |
| Daemon restarts | `systemd`: `Started/Stopping reachy-mini-daemon`; `systemctl show reachy-mini-daemon -p NRestarts` |

REST probes (LAN only, behind `LocalNetworkGuardMiddleware`):

| Probe | Endpoint |
|---|---|
| Daemon state, version | `GET /api/daemon/status` |
| Who holds the robot | `GET /api/daemon/robot-app-lock-status` |
| Running app | `GET /api/apps/current-app-status` (`null`, or `{state: starting/running/stopping/error/done}`) |
| Installed apps | `GET /api/apps/list-available/installed` |
| A job | `GET /api/apps/job-status/{job_id}` |
| Remote access | `GET /api/hf-auth/relay-status`, `GET /api/hf-auth/central-robot-status` |

Things that look like bugs but aren't yours:
- **Wi-Fi stalls.** The robot's Wi-Fi stalls about 1–2 s every 30 s
  (power save / roaming scans). Longer gateway losses come from the access
  point. Clients see these as re-dials. Rule them out with an on-robot
  `ping -D -i 0.2 192.168.1.1` before blaming your change.
- **uv on stderr.** uv prints `Using Python … environment` on stderr, and
  the job logger records every stderr line as `ERROR`.
- **numpy pin conflict.** `pip check` flags numpy above the pins of
  `pollen-bmi088-imu-library` and `gst-signalling`. It's long-standing and
  harmless so far.
- **libnice abort.** A rare libnice assertion
  (`conncheck.c:987 priv_conn_check_tick_stream_nominate`) aborts the
  daemon during ICE, and systemd restarts it. Note the time; it's not
  caused by app code.
- **Two clocks.** The journal is in the robot's timezone and only keeps
  the current boot. A phone's clock can be an hour off. Convert before
  correlating.

## 4. Driving the robot from a test

- **REST:** `curl` from the robot (`localhost:8000`) or from a machine on
  the same LAN.
- **JSON-RPC relay (`apps.*`, `conversation.*`, …):** reachable **only over
  the WebRTC data channel**, not over REST or `/ws/sdk`. You need a remote
  client:
  - **The mobile app**, driven over Chrome DevTools: call `rpcCall` on the
    live SDK instance. See the mobile guide, which has a script for it.
  - **Any JS SDK client** (`@pollen-robotics/reachy-mini-sdk`) connected
    through central: `await robot.rpcCall('apps.status')`.

  A small headless Node client for agents would remove the need for a
  phone. It doesn't exist yet.
- **Python apps on the robot:**
  - **Apps venv:** `/venvs/apps_venv`. uv lives at `/opt/uv/uv` (it isn't
    on a non-interactive SSH `PATH`).
  - **Simulate a crash:** `pkill -9 -f "apps_venv/bin/python -u -m <app>"`.
    The app goes to `error`, with `Process exited with code -9`.
  - **Simulate an orphan** (running, but no longer installed):
    `/opt/uv/uv pip uninstall --python /venvs/apps_venv/bin/python <app>`
    while the app runs.
  - **Start without evicting** the remote session:
    `POST /api/apps/start-app/<app>/no-evict`. It is refused while a phone
    holds the session; use the relay's `apps.start`, which keeps it.

## 5. Reporting

Following [`contributing.md`](contributing.md), paste what you ran and what
the robot said into the PR as an annex:
- the scenario table;
- the relevant `journalctl` lines;
- the REST state before and after.

For races and timing, include the counts that prove the outcome. For
example: "1 stop, 1 uninstall, 1 completed job, 0 failed".
