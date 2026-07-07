# Secret handshake + WiFi QR provisioning: agent handoff

Branch: `1245-improve-the-first-interaction-after-building-the-robot`
(push small commits directly to it, short messages, NEVER main/develop).
Full design/history: `2026-07-01-secret-handshake-design.md` (same folder).
This file is "current state + what's next" only.

## NEXT UP (redesign requested by Remi, NOT yet built)

Build lab-first (new modules in `examples/secret_handshake_lab/` + tests +
replay), then mirror into the daemon, then iterate on the wireless robot.
Keep changes minimal; reuse the existing collision detector + state-machine
patterns.

CURRENT DEPLOYED FLOW (filmable as-is): 3+3 antenna collisions (torque OFF)
-> WiFi provisioning, no robot motion, a voice line per outcome. The
redesign REPLACES the collision trigger's action and adds a torque-ON
subsystem.

### A. Torque-OFF handshake -> a simple wake button
- Trigger: 3 antenna collisions in fast succession, <1 s between each
  (single round; drop the current second round). Torque-off + sleep-pose
  gate as today.
- Action: call the daemon's existing `wake_up()` (backend/abstract.py): it
  torques on, gotos the base/init pose, plays the flute "toudoum"
  (wake_up.wav). Do NOT hand-roll motion; reuse `wake_up`.
- Rationale: waking is safe/reversible/obvious, so one round of 3 is enough.
  WiFi is NO LONGER triggered by collisions (moves to a torque-on code).

### B. Torque-ON antenna-button handshakes (NEW subsystem)
Most handshakes now happen while awake (torque on, base pose). The antennas
are 4 buttons (soft PID => safe to move them under torque; Remi validated
this in the `fire_nation_attacked` game).
- BUTTON-PRESS DEFINITION (from `../fire_nation_attacked/.../main.py`): a
  press = `|antenna angle| > 0.18 rad` (~10 deg); max useful ~0.65 rad. The
  SIGN of the angle = direction => 4 buttons: left-external, left-internal,
  right-external, right-internal.
  CONVENTION TRAP: fire_nation uses `antennas[0]=right, [1]=left`; our
  collision code uses `[0]=left`. Confirm on the robot which sign is
  "external" per antenna before trusting the mapping (probe like
  `examples/secret_handshake_lab/live_contact_probe.py`).
- Use EXTERNAL presses only in codes (narrows accidental triggers).
- Timing: <1 s between presses/movements or the sequence resets. Reuse the
  collision refractory + a release latch (angle must fall back below
  threshold between counted presses).
- Three codes and their actions:
  1. WiFi provisioning: SIMULTANEOUS symmetric move -- BOTH antennas
     external together, then BOTH internal together, <1 s between the two
     movements, small simultaneity window (a few ticks; margin not too
     large). Action: `get_shared_provisioner(...).start()`. Robot is
     already awake, so provisioning stays no-motion (already is).
  2. Torque OFF everything: 3x left-external, then 2x right-external, <1 s
     between. Action: `backend.set_motor_control_mode(Disabled)`. Screenless
     way back to torque-off (then a collision-wake or WiFi can follow).
  3. Emotion: left-ext, right-ext, left-ext, right-ext (alternating, 4
     presses), <1 s between. Action: play ONE FIXED "excited" emotion move
     (NOT random -- some are very long). Short excited move from
     `reachy-mini-emotions-library`; play via the daemon move playback
     (design spec section 8 has the bundling plan).

### C. Wiring notes
- `wake_up` / `goto_sleep` / `set_motor_control_mode` already exist on the
  backend. Emotion playback: find the offline RecordedMoves/play_move path;
  bundle the one excited move under `assets/` so it ships (spec s8).
- The torque-ON detector must be armed only while torque is ON (mirror of
  the collision detector's torque-off gate). Keep it sub-microsecond per
  tick (50 Hz control loop, CM4).
- Accidental-trigger safety = external-only + exact sequences + <1 s timing
  + a release between presses.

## Current state (deployed + validated on the wireless robot)

WHAT WORKS NOW: collision handshake (torque-off) -> WiFi QR provisioning,
no motion, narrated voice line per outcome. Handshake detection and the
full provisioning flow are live-validated on the wireless robot.

Code map:
- `src/reachy_mini/daemon/backend/secret_handshake.py` -- pure collision
  detector + state machine. One guarded call in `RobotBackend._update()`,
  armed ONLY when `motor_control_mode == Disabled`. Kill switch
  `REACHY_HANDSHAKE_ENABLED=0`.
- `src/reachy_mini/daemon/app/services/wifi_provisioning.py` -- provisioner
  (dependency-injected; offline tests in
  `tests/unit_tests/test_wifi_provisioning.py`). No robot motion: runs in
  place, plays cues + voice lines.
- `src/reachy_mini/daemon/app/routers/wifi_provision.py` -- endpoints
  `POST/GET /wifi/provision_qr/{start,status}`.
- Assets: `src/reachy_mini/assets/wifi_*.wav` (intro + one voice per
  outcome; ElevenLabs, Jean Atlas voice), `handshake_*.wav`,
  `assets/wechat_qrcode/` (QR CNN models, ~1 MB, Apache-2.0).
- Lab (tuning ground): `examples/secret_handshake_lab/` -- detector, state
  machine, 50 tests, `replay_validate.py` (HF dataset regression),
  `bench.py`, live probes.

## Deploy / operate on the wireless robot

- SSH: `pollen@reachy-mini.local` (password `root`). It runs a branch build
  in `/venvs/mini_daemon`.
- UPDATE CODE BY FILE SYNC, NOT pip reinstall: a pip `--force-reinstall`
  re-breaks opencv (see below) and churns deps. Use rsync:
  `rsync -az --exclude=__pycache__ --exclude='*.pyc' src/reachy_mini/ \
   pollen@reachy-mini.local:/venvs/mini_daemon/lib/python3.12/site-packages/reachy_mini/`
  then `sudo systemctl restart reachy-mini-daemon`.
- Enable WiFi-on-handshake (opt-in, default OFF): drop-in
  `/etc/systemd/system/reachy-mini-daemon.service.d/override.conf` with
  `[Service]\nEnvironment=REACHY_HANDSHAKE_WIFI_PROVISION=1` (the unit file
  itself is overwritten by the launcher on restart, so use a drop-in).
- Verify: `journalctl -u reachy-mini-daemon -f` -> `Secret handshake:
  armed`; `curl -s localhost:8000/wifi/provision_qr/status`.

## Durable decisions (do NOT re-litigate)

- Collision law v3 (GEOMETRIC, measured, degrees): collision at center =
  `l+r in [-9,0]` AND `l in [20,150]`, with all antenna angles WRAPPED to
  [-180,180) first (the encoders are MULTI-TURN; a floppy antenna handled
  past +-180 reads a whole turn off). v1 (angle diff) and v2 (velocities)
  were REJECTED; do not re-propose dynamic/velocity laws.
- Collision debounce = 0.25 s refractory ONLY. A firm press held >0.25 s
  double-counts (a prime can happen in 2 knocks): this is an ACCEPTED QUIRK,
  pinned by a test. A "release latch / must-separate" fix was tried and
  REVERTED -- it drops knocks during fast tapping (apart windows are only
  2-3 ticks). No static threshold separates the two cases. Missed knocks
  are worse than an occasional extra count.
- Success/abort re-arm STRAIGHT to armed (no cooldown, no idle round-trip):
  the pose gate flickers under the user's hands, so a round-trip made
  immediate retries silently fail. (No `armed` journal line after
  success/abort -- it never left armed.)
- QR decoding: stock `cv2.QRCodeDetector` read 0/108 real scan frames (phone
  screen, blur) at ~270 ms/frame on the CM4. Use
  `cv2.wechat_qrcode.WeChatQRCode` + the bundled CNN models (44/108,
  ~110 ms/frame). Graceful fallback chain: models -> no-models -> plain.
- opencv on the robot: must be a SINGLE `opencv-contrib-python` install.
  Having both `opencv-python` and `opencv-contrib-python` clobbers the
  contrib modules (incl. WeChatQRCode). PACKAGING: the wireless extra should
  declare `opencv-contrib-python` explicitly.
- Provisioning connect/confirm reuses the desktop-app onboarding path
  (`/wifi/connect` then poll `/wifi/status`: busy -> wlan+ssid = success,
  hotspot after busy = reverted/failed), including revert-to-hotspot on bad
  credentials. Nothing new was reinvented there.

## Gotchas

- Remi's Lite daemon often runs on his Mac holding port 8000:
  `test_wireless.py` / `test_daemon.py` fail and
  `test_app.py::test_faulty_app` HANGS. Environmental; CI is the arbiter.
- All tunables live in the config dataclasses (daemon `secret_handshake.py`,
  mirrored in the lab); the lab README has the values table + measured
  evidence (`data/touch_sweep_*.csv`).
- The lab is the tuning ground: change law/timing there first, re-run its
  tests + `replay_validate.py`, THEN mirror into the daemon module.
- Antenna gotos after floppy handling: encoders are multi-turn, so a goto
  whose start is a multi-turn read can behave oddly on hardware. The
  collision law wraps angles for detection; motion (wake_up goto) does not.
  A `GotoMove` start-wrap was tried and REVERTED (Remi could not reproduce a
  bad goto on the Lite); left as-is. Watch this when wiring wake_up.
