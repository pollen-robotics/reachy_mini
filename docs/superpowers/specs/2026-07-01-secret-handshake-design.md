# Secret Handshake mechanism for Reachy Mini

Status: design converged, ready to implement. Prototype not started in the
daemon yet. Offline collision lab exists and is validated (see below).
Created 2026-07-01, revised 2026-07-02 after analyzing recorded data.
Author: Remi Fabre. Analysis + spec: Claude Code.

This document is written to be picked up cold by a fresh agent. Read it top to
bottom, then start at "For the next agent: build order".

---

## 1. Purpose

Right after a user builds a Wireless Reachy Mini it has no WiFi and cannot run
any app until WiFi is provisioned from a phone/laptop. During that window the
robot feels dead. We want a gesture, baked into the flashed ISO, that the user
can perform on the robot itself (no network, no app) to make it do something
delightful, and later to kick off WiFi provisioning on purpose.

The gesture is a physical "secret handshake" performed while the motors are
torque OFF. It must be extremely safe: near-impossible to trigger by accident,
and impossible to trigger during normal use (torque on). It ships enabled by
default in the next ISO, so correctness matters a lot.

## 2. What is decided (do not re-litigate)

- Trigger only when torque is OFF. A freshly booted robot is already torque off
  (the wireless launcher passes `--no-wake-up-on-start`), so it is armed out of
  the box.
- The gesture, v1: put the head in the sleep pose, then do 3 antenna collisions
  (the "shared prefix"), hear a confirmation beep, then do 3 more collisions to
  confirm, which triggers the action. Two rounds with a beep between them.
- The action, v1: play one emotion move, chosen per robot by a deterministic
  distribution (the "shiny Pokemon" idea). Bundled offline in the ISO.
- Enabled by default in the first ISO that ships it, with a config kill switch.
- WiFi provisioning is a SEPARATE later action (a different second-round
  gesture), not built now. Feasibility notes in section 11.
- The collision definition is a single scalar on the antenna angles. Validated
  against real recordings. See section 4.

## 3. Codebase integration points (verified, with locations)

- Control loop: `RobotBackend._update()` in
  `src/reachy_mini/daemon/backend/robot/backend.py` (~line 184), runs at 50 Hz
  and already reads antenna present positions and head pose every tick. This is
  where the detector gets fed, in one line. Cost is a few float compares.
- Antenna present positions are readable while torque is OFF. In the daemon:
  `get_present_antenna_joint_positions()` returns `[ant0, ant1]` = `[left,
  right]` (radians). Backend: `robot/backend.py` ~line 482.
- Torque mode: `self.motor_control_mode == MotorControlMode.Disabled` means
  torque off. Enum in `src/reachy_mini/io/protocol.py`.
- Sleep pose constants: `backend/abstract.py` ~line 933-954.
  `SLEEP_HEAD_POSE` (4x4) and `SLEEP_ANTENNAS_JOINT_POSITIONS = (-3.05, 3.05)`.
  Also `distance_between_poses(...)` helper is used by wake_up/goto_sleep.
- Playback (torque must be ON): `play_move()` in `backend/abstract.py`
  ~line 570. `wake_up()`/`goto_sleep()` ~line 956+. Enabling torque safely
  pins to the current pose first (see `skills/safe-torque.md`).
- Sounds: `self.play_sound("wake_up.wav")` pattern in `backend/abstract.py`.
  Bundled wavs live with the package.
- Emotions library loader: `src/reachy_mini/motion/recorded_move.py`. Downloads
  from HuggingFace, tries local cache first, so a bundled local path works
  offline. Move format = JSON (`time`, `set_target_data[].head/antennas/
  body_yaw`) with optional sidecar audio.
- Per-robot unique id (already production ready):
  `src/reachy_mini/utils/hardware_id.py` `get_hardware_id()` returns a
  deterministic 16-hex SHA-256, burned at manufacturing, stable across
  reflashes, on Lite and Wireless. This is the seed for the distribution.
- The ISO is built in a SEPARATE repo, `pollen-robotics/reachy-mini-os`. This
  repo is the daemon/SDK that gets installed into it. Shipping = the updated
  package lands in the next image; systemd already autostarts the daemon.

## 4. The collision primitive (validated)

Antenna present positions arrive as `[ant0, ant1]` = `[left, right]` radians.
Define one scalar:

    diff = ant0 - ant1

Geometry, measured from the recordings (`RemiFabre/secret-handshake`):

    rest (splayed outward, torque off) : diff ~ -0.35
    antennas meet at center            : diff ~ 0     (the "angles equal" touch)
    firm press / flex                  : diff  =  +2 .. +5

So the two states separate cleanly on one number. Detect contact with
hysteresis so a held touch is one event, not a flicker:

    not-contact -> contact   when diff > T_ON      (default 0.5)
    contact -> not-contact   when diff < T_OFF     (default 0.0)

Rest (-0.35) is far below T_OFF, firm press (+2..5) far above T_ON, so this is
robust to noise. These exact defaults are validated in
`examples/secret_handshake_lab/` (1 clean onset on every recording).

Optional robustness (add only if live testing shows accidental single-antenna
triggers): also require centeredness `abs(ant0 + ant1) < S_MAX`. Note the
recordings show `ant0 + ant1` wanders up to ~1.0 during real contact, so
S_MAX must be generous (>= 1.2) or it will chop real collisions. Prefer to
leave it out unless needed.

Per-robot calibration option: instead of a fixed T_OFF, sample the resting
`diff` for the first N ticks after arming and set thresholds relative to it.
This absorbs antenna zero-offset differences between robots. Recommended.

## 5. Counting "3 collisions" (implement + confirm live)

The user performs 3 collisions in a short rhythm. IMPORTANT nuance for the
implementer: in the 4 recorded "default" examples the antennas are brought
together once and then held (with audible knocks) rather than fully separated
between knocks, so pure edge-counting sees 1 onset in that data. The 3 knocks
are real (audible, visible on replay) but do not always fully separate in the
joint angles. Therefore implement BOTH counters and pick the one that matches
the user's natural gesture using `live_contact_probe.py`:

- Counter A, release edges (preferred if the user separates between taps):
  count rising edges of contact (section 4). Require a release (diff < T_OFF)
  between counted taps. This is the "discrete tap password" primitive and
  extends to arbitrary future tap patterns.
- Counter B, knock peaks (fallback if the user keeps light contact between
  knocks): while in sustained contact, count local maxima of diff whose
  prominence (dip on at least one side) exceeds a margin (~0.25 rad seen in
  data), with a refractory (~150 ms) between counted knocks.

Rhythm gate (both counters): exactly 3 collisions within a rolling window
(~2.5-3.5 s), with a minimum spacing (~120 ms) so a single bounce is not
double counted. Keep the window lenient for v1; the future "advanced password"
can tighten timing per user.

First deliverable is the standalone tester (already scaffolded, section 9) so
the user can watch for false positives before any daemon change.

## 6. Base / sleep pose gate (data-derived, generous)

The gate confirms the head is roughly in the sleep pose before collisions
count. Two facts from the data:

- The natural torque-off head rest at the very start of a recording is a
  wake-ish pose (pitch ~7deg, z ~-12mm), NOT the sleep pose. That is the user
  moving the head down into sleep. Ignore the first ~1-2 s.
- After settling into sleep, the head sits close to `SLEEP_HEAD_POSE`
  (z=-44mm, pitch=24.4deg). Measured deviations across recordings (t >= 2 s):
  z within ~7 mm, pitch mostly 16-26 deg (one outlier at 8 deg), roll within
  ~4 deg, x within ~13 mm, yaw naturally loose.

Recommended gate (compare live head pose to `SLEEP_HEAD_POSE`), generous:
  x: +-15 mm, y: +-12 mm, z: +-12 mm, roll: +-8 deg, pitch: +-15 deg,
  yaw: ignore (or +-40 deg).

Check the gate at ARMING (before the first collision), not continuously, since
manipulating the antennas jostles the floppy head. Keep it lenient: it is a
sanity gate, the real security is torque-off + the two-round collision rhythm.
Use the existing `distance_between_poses` / pose helpers rather than
hand-rolling matrix math.

## 7. Handshake state machine

Runs off the samples fed from `_update()`. Pure, no I/O; it calls provided
callbacks to beep and to run the action.

    IDLE
      -> ARMED            when torque OFF and head ~= sleep pose (section 6)
    ARMED
      -> PRIMED           on 3 collisions within the rhythm window; play BEEP
                          (this "sleep + 3 collisions + beep" is the shared
                           prefix of every future handshake)
    PRIMED
      -> run ACTION #1    on 3 more collisions within the window (v1: emotion)
      -> (future)         other second-round gestures select other actions
      -> IDLE             on timeout (~8 s) with no valid second round
    after action -> return to sleep + torque OFF -> ARMED again (repeatable)
    torque turns ON at any point -> IDLE (and stays inert while torque on)

Notes:
- When the action starts it enables torque, which flips the arming gate off, so
  the detector cannot re-fire mid-action. No extra locking needed, but still
  guard the action behind the existing move lock.
- All timeouts and counts live in a config dataclass with the section 4/5/6
  margins, so tuning never touches control-loop code.

## 8. Action layer: emotion + shiny-Pokemon distribution

- Bundle a small subset of emotion moves (JSON + sidecar audio) into
  `src/reachy_mini/assets/handshake_moves/` so they ship in the wheel
  (pyproject globs package data) and thus the ISO. Load via a local-path
  `RecordedMoves` (loader prefers local, so it works with no network).
- Pick deterministically from `get_hardware_id()`:

      bucket = int(get_hardware_id(), 16) % 1000
      #   0-329  happy      (common)
      # 330-659  excited    (common)
      # 660-989  <third>    (common, e.g. dance or sad)
      # 990-999  anger      (~1% "shiny")

  Exact move set and splits are a config table. Unit-test the distribution over
  many synthetic ids to confirm proportions and determinism.
- Playback: safe-enable torque (pin to current pose), `play_move(..., 
  initial_goto_duration=...)` so it eases into the move start, existing synced
  audio. On finish: `goto_sleep()` then disable torque, back to ARMED.

## 9. Offline + on-robot lab (already built, start here)

`examples/secret_handshake_lab/` (kept off the daemon critical path):
- `collision.py` - the pure `CollisionDetector` (Counter A primitive) and
  `KnockDetector` (Counter B: pressure peaks during held contact).
- `handshake.py` - the pure `HandshakeStateMachine` + `HandshakeConfig`
  (section 7, built 2026-07-02). Both handshakes: 3 taps + 3 taps ->
  ACTION_TAPS, 3 taps + long hold -> ACTION_HOLD. ~0.1 us per update. 20
  synthetic unit tests in `test_handshake.py`.
- `pose_gate.py` - the section 6 sleep-pose gate, pure floats.
- `replay_validate.py` - runs the detector AND the full machine over the HF
  recordings. Results: edge machine never primes on any recording (zero false
  positives); knock machine primes exactly where the human did the 3-knock
  gesture (default 1-3) but also fully triggers on free antenna play
  (collision-definition files), so edge stays the default.
- `live_handshake_probe.py` - ON-ROBOT full-machine tester with beeps
  (primed beep, distinct success fanfares per handshake, abort buzz). Flags:
  `--counter knock`, `--no-pose-gate`, `--disable-motors`. This is the
  section 12 step-2 empirical tool now.
- `sim_keyboard.py` - no-robot rehearsal (Enter = tap, h+Enter = hold), same
  loop and beeps.
- `live_contact_probe.py` - on-robot raw contact printer (lower level).
- Dataset: https://huggingface.co/datasets/RemiFabre/secret-handshake
  (`default*.json` = intended 3-collision gesture; `collision-definition*.json`
  = antennas swept through the full contact range). Note: `default4.json` sits
  ~17-23 deg shallower in pitch than the sleep pose and correctly never arms.

## 10. Safety and shipping (ISO critical, enabled by default)

- Armed only when `motor_control_mode == Disabled`. Inert during any app/use.
  Action turns torque on, which disarms the detector. No re-entry.
- Two-round confirmation (3 + beep + 3) keeps accidental-trigger probability
  extremely low. Keep it.
- Kill switch: config/env flag (e.g. `REACHY_HANDSHAKE_ENABLED`, default on).
- Fail-safe: wrap the detector call in try/except; any error is logged and
  swallowed, never propagates into the 50 Hz control loop.
- Reuse existing tested motion paths (safe torque, goto, play_move). New code
  only decides WHEN.
- Definition of done before the ISO ships (human, on hardware, cannot be done
  by the agent): (a) accidental-trigger soak, e.g. run dances/moves and wiggle
  antennas with torque off, expect zero triggers; (b) intended-trigger
  reliability across several people; (c) confirm the emotion plays offline on a
  freshly flashed, never-connected robot.

## 11. WiFi QR provisioning (deferred, v2+)

Feasible but heavier and deliberately decoupled. Camera is up before WiFi but
there is no snapshot path yet and no QR decoder in the constrained CM4 deps
(needs pyzbar/zbar or opencv QR, which add image size/CPU). Parsing
`WIFI:S:...;T:...;P:...;;` is trivial; feeding credentials into the existing
NetworkManager path is the real work and the OS-level parts live in
`reachy-mini-os`. Do NOT use a boot default-app (implicit, fragile, the
"was this the first run?" problem). Instead trigger provisioning on purpose via
a second-round gesture branch: handshake -> gesture B -> QR-scan mode ->
decode -> connect -> success/fail sound. Reuses everything here. Scope
separately when ready.

## 12. For the next agent: build order

1. Read section 3, then run `examples/secret_handshake_lab/replay_validate.py`
   to see the primitive pass on real data. Read `collision.py`.
2. Get `live_handshake_probe.py` onto a real robot, torque off, and have the
   user perform the full two-round gesture with beep feedback. Decide Counter
   A vs B (section 5) and lock `t_on`, `t_off`, rhythm window, min spacing.
   This is the single most important empirical step; do it before writing
   daemon code. [IN PROGRESS 2026-07-02: tooling ready, awaiting human]
3. Build the pure `HandshakeStateMachine` (section 7) + config dataclass, with
   unit tests driven by the recorded traces and synthetic tap sequences. No
   hardware imports. [DONE 2026-07-02: both counters, both handshakes, 20
   tests + HF replay regression all passing]
4. Build the action layer (section 8): bundle the emotion subset under
   `assets/handshake_moves/`, write `pick_emotion(hardware_id)` + distribution
   unit test, wire safe playback.
5. Promote the detector + state machine into a new
   `daemon/backend/secret_handshake.py`, and add the single guarded call in
   `RobotBackend._update()` behind the kill-switch flag. Fail-safe wrap.
6. Hand back to the human for the section 10 on-robot validation before the ISO
   default is trusted.

Keep everything pure and small (sections isolated as in this doc). The
control-loop footprint must stay: read cached antenna positions + head pose
(already read) -> one function call -> optional callback. Nothing heavier.
