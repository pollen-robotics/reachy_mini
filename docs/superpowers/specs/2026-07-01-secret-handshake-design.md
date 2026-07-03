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
- The collision definition is GEOMETRIC, measured by Remi by hand on 2
  robots: a narrow band on l+r plus a range on l (degrees). Validated
  against the real recordings. See section 4.

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

## 4. The collision primitive (v3, geometric; measured by Remi on 2 robots)

The definition, in DEGREES (`l_ant` = antenna index 0, `r_ant` = index 1,
as returned by `get_present_antenna_joint_positions()`, converted from rad).
The antennas are in collision "at the center" (y = 0 plane) when BOTH:

    1) l_ant + r_ant in [-7, -3]   (~= 0 with a slight consistent asymmetry,
                                    seen on both measured robots)
    2) l_ant        in [20, 150]

widened by a margin: the default 4 deg margin turns the sum band into
[-9, -1]. A collision EVENT = entering that region; a 0.25 s refractory
merges the double band-crossing of a firm press (pressing flexes the
antennas, sum shoots to +35..+73 deg in the recordings, then passes back
through the band on release).

Why the conditions work (validated on `RemiFabre/secret-handshake`, hard
regression in `replay_validate.py`):
- EXACTLY 3 collisions on each of the 4 default*.json gestures, at the
  audible knock times. The bring-together approach does not count because
  condition 2 fails while l is still below 20 deg.
- Rest is inert: sum (~-2) is inside the band but l (~-12) fails condition
  2. Single-antenna play and the crossed-parked state (sum ~ -11) are also
  outside the region.
- CAVEAT to judge live: slow slide play (collision-definition*.json) passes
  through and lingers in the band, so it can prime and even complete the
  hold handshake. Protection = two-round structure + sleep-pose gate +
  torque-off arming (+ section 10 soak testing).

HISTORY, do not retry these: v1 was `diff = ant0 - ant1` with absolute
thresholds; it failed live because the floppy friction-fit antennas rest
wherever they were last left (one robot rested at diff +56 deg, the dataset
robot at -20 deg), latching the detector "in contact" forever. v2 was
velocity-based coupled motion (m = min(|v0|,|v1|) spikes); it validated
offline but Remi rejected dynamic definitions as not robust enough and
measured the geometric law above by hand (see
`examples/secret_handshake_lab/position_gui.py`, built for that).

## 5. Counting "3 collisions" (implemented, confirm live)

A collision = one entry into the geometric region (section 4), refractory
0.25 s; natural knock spacing in the recordings is ~0.4 s.

Rhythm gate: 3 collisions within a rolling window (~3 s), minimum spacing
120 ms. Keep the window lenient for v1; the future "advanced password" can
tighten timing per user.

Second-round alternative gesture (handshake B): a gentle HOLD, i.e. staying
in the collision region for >= 1.0 s (membership dips bridged by a 0.3 s
grace). Quick taps leave the region immediately, so tapping cannot add up
to a hold; on the tick a 3rd tap lands, ACTION_TAPS wins over the hold.

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
      -> run ACTION #2    on a gentle hold in the region (>= 1 s, section 5)
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
- `collision.py` - the pure geometric `CollisionDetector` (section 4 v3).
- `handshake.py` - the pure `HandshakeStateMachine` + `HandshakeConfig`
  (section 7, rebuilt 2026-07-03 on the v3 law). Both handshakes: 3 taps +
  3 taps -> ACTION_TAPS, 3 taps + hold -> ACTION_HOLD. ~0.2 us per update.
  25 synthetic unit tests in `test_handshake.py`.
- `pose_gate.py` - the section 6 sleep-pose gate, pure floats.
- `position_gui.py` - torque toggles (head / antennas separately) + live
  present-position readout + snapshot logging; the tool Remi used to
  measure the v3 definition. Shows the collision flag live.
- `analyze_recordings.py` - v2-era investigation tool (velocity stats and
  plots), kept as a record of the analysis.
- `replay_validate.py` - hard regression over the HF recordings: EXACTLY 3
  collisions on every default gesture at the knock times, the full machine
  PRIMES on the gestures whose head pose arms and fires no action on them.
  Slide files are printed for judgment (they can prime and hold, see
  section 4 caveat).
- `live_handshake_probe.py` - ON-ROBOT full-machine tester with beeps
  (primed beep, distinct success fanfares per handshake, abort buzz). Flags:
  `--no-pose-gate`, `--disable-motors`. This is the section 12 step-2
  empirical tool. Status line: sum/l in degrees + IN-BAND flag.
- `sim_keyboard.py` - no-robot rehearsal (Enter = collision, h+Enter =
  hold), same loop and beeps.
- `live_contact_probe.py` - on-robot raw signal printer (l/r/sum in deg,
  lower level, no state machine).
- Dataset: https://huggingface.co/datasets/RemiFabre/secret-handshake
  (`default*.json` = intended 3-collision gesture; `collision-definition*.json`
  = one antenna slid over the other, touching ~90% of the time;
  `collisions-test.json` = later addition, 5 spaced-out collisions found).
  Notes: `default4.json` sits ~17-23 deg shallower in pitch than the sleep
  pose and correctly never arms; the collision-definition files END crossed
  and parked (sum ~ -11 deg, just below the band).

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
   user perform the full two-round gesture with beep feedback. Confirm the
   v3 band + margin, rhythm window, hold duration. [IN PROGRESS 2026-07-03:
   v1 angle-diff law failed live, v2 velocity law rejected by Remi, v3 is
   Remi's own measured geometric definition; awaiting his live pass,
   especially the section 4 slide caveat]
3. Build the pure `HandshakeStateMachine` (section 7) + config dataclass, with
   unit tests driven by the recorded traces and synthetic tap sequences. No
   hardware imports. [DONE 2026-07-03: v3 law, both handshakes, 25 tests +
   HF replay regression all passing; exactly 3 collisions found on each
   recorded gesture]
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
