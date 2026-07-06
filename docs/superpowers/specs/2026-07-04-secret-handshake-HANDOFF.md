# Secret handshake + WiFi QR provisioning: agent handoff

Date: 2026-07-04. Branch: `1245-improve-the-first-interaction-after-building-the-robot`
(push small commits directly to it, short messages, NEVER main).
Full design/history: `2026-07-01-secret-handshake-design.md` (same folder).
Read that spec top to bottom before changing anything; this file is only
"where we are and what is next".

## UPDATE 2026-07-06 (evening): WiFi QR provisioning LIVE-VALIDATED end to end

Full flow observed on the wireless robot: handshake -> narrated intro
(ElevenLabs voices in assets/wifi_*.wav, one per outcome) -> torque on +
5 s goto base pose -> camera scan -> QR decoded on the FIRST frame ->
/wifi/connect path -> success voice -> goto sleep + torque off. Journey
to get there, all committed:

1. Stock cv2.QRCodeDetector decoded 0/108 real scan frames (phone screen,
   blur + auto-exposure blowout) at ~270 ms/frame on the CM4. Replaced by
   cv2.wechat_qrcode.WeChatQRCode + its CNN models (bundled in
   assets/wechat_qrcode/, ~1 MB, Apache-2.0): 44/108 decoded at
   ~110 ms/frame on the CM4. Frames of every scan are dumped to
   /tmp/reachy_wifi_qr_frames (wiped per run) for diagnosis.
2. The robot had BOTH opencv-python 4.13 and opencv-contrib-python 4.11
   installed (mediapipe pulls contrib; something else pulled plain); they
   share the cv2 dir, so contrib modules were clobbered. Fixed on-robot:
   single opencv-contrib-python 4.13. PACKAGING DECISION: the wireless
   extra should declare opencv-contrib-python explicitly.
3. The handshake->provisioning trigger no longer skips when WiFi is
   already connected (re-provisioning allowed; an app will gate this
   later). Enabled on the robot via
   /etc/systemd/system/reachy-mini-daemon.service.d/override.conf
   (REACHY_HANDSHAKE_WIFI_PROVISION=1).
4. prepare/finish hooks use set_motor_control_mode (NOT
   enable/disable_motors directly), so motor_control_mode stays truthful
   during the self-moves and the handshake disarms while the robot moves.

Still open: erase-credentials + hotspot re-provision test (stage 2), the
section 10 soak, and the emotion action layer (section 8, not built).

## UPDATE 2026-07-06: retry friction + double-count fixed (commit cbb9dc79)

Remi's weekend testing surfaced two real usability bugs, both fixed
lab-first and deployed to the robot:

1. Immediate retry after success/abort silently failed. The machine
   dropped to idle and re-required pose gate + 0.5 s settle (+ 2 s
   cooldown after success); live, the user's hands jostle the floppy head,
   the gate flickers, and taps thrown before re-arm are discarded.
   NOW: success and abort return STRAIGHT to armed (pose gate only guards
   boot / torque-off transitions). Note: no `armed` journal line after a
   success/abort anymore, the machine is already armed.
2. A firm knock pressed longer than the 0.25 s refractory counted twice
   (contact + release re-crossing), so 2 knocks could prime (Remi saw
   this; there is NO boot-state bug). A release latch (80 ms not-pressed
   dwell between counts) fixed it in the lab and passed replay, but was
   REVERTED within the hour: live fast tapping separates the antennas for
   only 2-3 control ticks, so the latch dropped knocks and the gesture
   only worked slowly (Remi's live report). The double count is now a
   documented ACCEPTED QUIRK, pinned by a test. Lessons: (a) the recorded
   default.json knocker never fully separates the antennas, so "must
   separate between counts" fails replay; (b) fast-tap apart windows and
   release-crossing durations overlap, so no static dwell threshold works;
   (c) velocity discrimination stays rejected. Missed knocks are far worse
   than an occasional 2-knock prime.

`cooldown_s` is gone from both configs. Replay regression unchanged
(3 collisions + primed on all defaults), 50 tests, worst-case detector
cost unchanged (sub-microsecond).

## UPDATE 2026-07-04 (later): deployed and LIVE-VALIDATED on the wireless robot

Option B install done (`/venvs/mini_daemon` at 1.8.4 from this branch),
daemon restarted, and the full flow observed in the journal with Remi at
the robot: `armed -> primed -> success -> re-armed`, both sounds played
through ALSA. One real bug was found and fixed live (commit d8715149):
the antenna encoders are MULTI-TURN, and on Remi's robot they were parked
a full turn away from zero (rest read l=-340 deg, physically +20). The
raw `l in [20, 150]` gate could never pass. The collision law now wraps
angles to [-180, 180) in both the lab detector and the daemon module
(failing test written in the lab first, replay regression unchanged).

Still open: spec section 10 soak/multi-person validation, QR provisioning
live test (opencv is not installed on the robot), emotion action layer
(spec section 8, not built).

## State: everything is implemented and pushed, awaiting on-robot validation

1. Collision law v3 (GEOMETRIC, measured by Remi, degrees): collision at
   center = `l+r in [-9, 0]` AND `l in [20, 150]` (l = antenna index 0).
   v1 (angle diff) and v2 (velocities) were REJECTED; do not re-propose.
2. Lab (`examples/secret_handshake_lab/`): validated detector + two-round
   state machine, 28 tests, replay regression against the HF dataset
   (exactly 3 collisions per recorded gesture), bench.py, live probes with
   beeps. Remi live-validated the detector on 3 robots (100% detection).
3. Daemon (sounds only, default handshake 3+3, no hold gesture):
   - `src/reachy_mini/daemon/backend/secret_handshake.py` (pure module)
   - one guarded call in `RobotBackend._update()`; armed ONLY when
     `motor_control_mode == Disabled` (all torque off)
   - kill switch `REACHY_HANDSHAKE_ENABLED=0`
   - sounds `assets/handshake_*.wav` = tap-lab tones, regenerate with
     `examples/secret_handshake_lab/render_daemon_sounds.py`
   - tests `tests/unit_tests/test_secret_handshake.py`
4. WiFi QR provisioning (wireless only, additive):
   - `daemon/app/services/wifi_provisioning.py` (dependency-injected,
     offline tests in `tests/unit_tests/test_wifi_provisioning.py`)
   - endpoints `POST/GET /wifi/provision_qr/{start,status}`
   - camera = media server's existing IPC branch (GStreamerCamera);
     QR = lazy cv2 (state `unavailable` if opencv missing; opencv is NOT
     in the wireless-version extra, ISO decision pending)
   - connect/confirm mirrors the desktop-app onboarding
     (reachy-mini-desktop-app WiFiConfiguration.jsx): /wifi/connect route
     + /wifi/status polling (busy -> wlan+ssid ok, hotspot = reverted)
   - handshake SUCCESS -> provisioning is OPT-IN:
     `REACHY_HANDSHAKE_WIFI_PROVISION=1` (default OFF)

## Next step (was in progress at handoff): deploy to the wireless robot

Follow `docs/source/platforms/reachy_mini/install_daemon_from_branch.md`
Option B: `ssh pollen@reachy-mini.local` (password `root`), pip install the
branch into `/venvs/mini_daemon`, `sudo systemctl restart
reachy-mini-daemon`, then `journalctl -u reachy-mini-daemon -f` and have
Remi do the gesture: expect log lines `Secret handshake: primed/success`
and the sounds from the robot speaker.

BLOCKER at handoff: the robot did not answer on the LAN (mDNS cached
192.168.1.14 but dead; no `_reachy-mini._tcp` zeroconf advert). Likely
hotspot fallback or mid-boot; ask Remi about the robot's network state.

Then: spec section 10 validation (accidental-trigger soak, multi-person
reliability), QR provisioning live test (needs opencv on the robot),
and the emotion action layer (spec section 8, NOT built yet).

## Gotchas

- Remi's Lite daemon often runs on his Mac holding port 8000:
  `test_wireless.py`, `test_daemon.py` fail and
  `test_app.py::test_faulty_app` HANGS. Environmental; deselect locally,
  CI is the arbiter. `test_daemon_wireless_client_disconnection` needs a
  live wireless robot on the LAN.
- All tunables live in the two config dataclasses (secret_handshake.py in
  the daemon, mirrored in the lab); the lab README has the values table
  and the measured evidence (`data/touch_sweep_*.csv`).
- The lab is the tuning ground: change law/timing there first, re-run its
  tests + replay_validate.py, then mirror into the daemon module.
