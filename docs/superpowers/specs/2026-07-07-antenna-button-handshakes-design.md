# Antenna-button handshakes + base-relative press (design)

Date: 2026-07-07. Branch: `1245-improve-the-first-interaction-after-building-the-robot`.
Supersedes the "NEXT UP" section of `2026-07-04-secret-handshake-HANDOFF.md`.
Read that handoff and `2026-07-01-secret-handshake-design.md` for the full
history and the durable decisions (collision law v3, debounce, re-arm, QR
decoder, opencv packaging). This spec covers only the redesign.

## 1. Why

The deployed flow is: 3+3 antenna collisions (torque OFF) -> WiFi QR
provisioning. The redesign splits interaction into two subsystems:

- A torque-OFF **wake button** (collisions): the safe, obvious first action.
- A torque-ON **antenna-button** layer: the antennas become 4 soft buttons,
  and short press-codes trigger WiFi, torque-off, and an emotion. This is
  most of the interaction, done while the robot is awake.

## 2. The press primitive (the correction that motivated this spec)

The `fire_nation_attacked` button definition (`|angle| > 0.18 rad`) measures
from ZERO and only works because that game pins the antennas at `[0, 0]`. The
antennas now rest at `INIT_ANTENNAS_JOINT_POSITIONS = [-0.1745, +0.1745]` rad
(~10 deg out), so from-zero would read the resting antennas as pressed. A
first fix measured deviation from that fixed base, but that only works at the
base pose.

**A press is defined as a deviation of the PRESENT angle from the antenna's
live GOAL (commanded target)**, not from zero and not from a fixed base:

- Per antenna `i`, deviation `d_i = wrap_to_pi(present_i - goal_i)`. Wrap
  first (encoders are multi-turn; collision law v3 established this).
- External direction per antenna: `ext_i = sign(base_i)` (index 0 external is
  negative, index 1 positive). `base` NO LONGER references the deviation; it
  only fixes each antenna's external side.
- **External press**: `d_i * ext_i > PRESS_THRESHOLD` (0.18 rad, ~10 deg).
- **Internal press**: `d_i * ext_i < -PRESS_THRESHOLD`.
- No goal known (`goal_i is None`) -> the button subsystem is inert.

Why goal-relative is more robust: it works in ANY pose, and it is immune to
the robot's OWN commanded motion. When the robot moves the antennas on
purpose the present angle tracks the goal, so `d_i` stays ~0 and nothing
triggers; only an EXTERNAL push (present diverges from the commanded goal)
registers. The live goal is `RobotBackend.target_antenna_joint_positions`,
passed into the detector each tick.

The left/right index mapping (fire_nation uses `[0]=right, [1]=left`; the
collision code uses `[0]=left`) does NOT affect press DETECTION: external is
`sign(base_i)` per index. It only affects which physical antenna the coded
sequences call "left"; confirm on the robot with `live_contact_probe.py`
before trusting the code mapping.

**Release latch**: after a press is counted on an antenna, that antenna must
fall back below a release threshold (hysteresis: release at 0.12 rad
deviation) before the next press on it counts. This is a genuine button
debounce and is CORRECT here, unlike the collision path (where a release
latch was tried and reverted because fast knocks have only 2-3 ticks apart).
Button presses are deliberate and separated; a latch removes contact/release
double-counting cleanly.

## 3. Subsystem A: collision -> wake button (torque OFF)

- Trigger: 3 antenna collisions in fast succession (<1 s between each),
  SINGLE round. Drop the current second round.
- Gate: torque OFF + sleep-pose settle, exactly as today.
- Action: `await backend.wake_up()` (torques on, gotos base pose, plays the
  flute "toudoum"). Reuse it; do not hand-roll motion.
- WiFi is NO LONGER triggered by collisions (it moves to a torque-ON code).
- Rationale: waking is safe/reversible/obvious, so one round of 3 suffices.

Implementation: the existing `SecretHandshake` (daemon
`backend/secret_handshake.py`) collapses from 3+3 to a single round of 3 and
its SUCCESS event maps to wake instead of WiFi.

## 4. Subsystem B: antenna-button codes (torque ON)

Armed ONLY while torque is ON (mirror of the collision gate, which arms only
while Disabled). Consumes press events from the primitive. Timing: <1 s
between presses/moves or the sequence resets. External presses only in codes
(narrows accidental triggers). A release must occur between counted presses
(the latch).

Three codes:

1. **WiFi provisioning** -- SIMULTANEOUS symmetric move: BOTH antennas
   external together, then BOTH internal together, <1 s between the two
   movements, small simultaneity window (a few ticks). Action:
   `get_shared_provisioner(...).start()`. Robot already awake -> provisioning
   stays no-motion.
2. **Torque OFF everything** -- 3x left-external, then 2x right-external, <1 s
   between. Action: `set_motor_control_mode(Disabled)`. Screenless way back
   to torque-off (then a collision-wake or WiFi can follow).
3. **Emotion** -- left-ext, right-ext, left-ext, right-ext (alternating, 4
   presses), <1 s between. Action: play ONE FIXED short "excited" move (NOT
   random -- some are very long). Robot is already awake/torque-on, so no
   safe-enable/goto_sleep dance (unlike the old design s8): just play the
   bundled move.

## 5. Architecture / components

Lab-first (tuning ground `examples/secret_handshake_lab/`), then mirror into
the daemon. New pure modules, each one clear purpose, no I/O, sub-microsecond
per tick (50 Hz, CM4):

- `antenna_buttons.py` -- `AntennaButtonConfig` + `AntennaButtonDetector`.
  `update(t, ant0, ant1, goal0, goal1) -> list[Press]` where `Press` names
  antenna index + direction (external/internal); the press is a deviation of
  present from goal. Holds the per-antenna release latch. `base` (config)
  only fixes each external direction; no goal -> no presses.
- `button_codes.py` -- `ButtonCodeConfig` + `ButtonCodeMachine`.
  `update(t, presses, torque_on) -> CodeEvent | None`. Encodes the three
  sequences, the <1 s gap reset, and the simultaneity window for the WiFi
  symmetric move. Pure; returns an event, caller runs the action.

Daemon mirror:
- `backend/secret_handshake.py`: collision path collapses to one round of 3;
  add the button detector + code machine as a second, torque-ON-gated path.
  Both behind the existing `REACHY_HANDSHAKE_ENABLED` kill switch.
- `backend/robot/backend.py` `_update()`: one guarded call per path. Map
  events to actions: wake -> `wake_up`; WiFi -> provisioner; torque-off ->
  `set_motor_control_mode(Disabled)`; emotion -> play bundled move.
- Actions that block (wake_up, play_move, provisioner) run OFF the control
  loop (thread / async task), exactly as `_spawn_qr_provisioning` does today.

## 6. Emotion asset

Bundle ONE short excited move under `src/reachy_mini/assets/handshake_moves/`
(JSON + optional sidecar audio) so it ships in the wheel. Load via a
local-path `RecordedMoves` (loader prefers local cache -> works offline).
Source the move from `reachy-mini-emotions-library` (installed on the robot);
pull one short excited move via SSH and vendor it. Play with `play_move(...,
initial_goto_duration=...)` so it eases in from the current pose. If the
asset is missing, log and skip (graceful fallback, no crash).

## 7. Testing

- Unit tests for `antenna_buttons.py`: rest = no press; external/internal per
  antenna with the correct sign; base-relative (a reading at base is not a
  press even though `|angle| ~ 0.18`); release latch (held press counts once;
  press-release-press counts twice); wrap handling (multi-turn reading).
- Unit tests for `button_codes.py`: each of the 3 codes fires; wrong
  order/direction does not; >1 s gap resets; simultaneity window for WiFi
  (too-far-apart the two antennas -> no fire); internal presses ignored where
  external required; torque-off gate.
- Regression: extend `replay_validate.py` if button data exists; otherwise
  synthetic press streams.
- `bench.py`: confirm sub-microsecond per tick on the button path.
- Live on the wireless robot (Remi's is ON): probe the sign mapping, then run
  each code and the wake button; confirm no false triggers at rest.

## 8. Safety / accidental-trigger defense

External-only + exact sequences + <1 s timing + a release between presses +
torque-ON gate. The torque-off code disables torque (safe, reversible). Wake
and WiFi are safe. Emotion is a short fixed move. Kill switch unchanged
(`REACHY_HANDSHAKE_ENABLED=0`).

## 9. Durable decisions inherited (do NOT re-litigate)

Collision law v3, 0.25 s collision refractory (accepted double-count quirk),
re-arm straight to armed, WeChatQRCode + bundled models, single
opencv-contrib install, provisioning reuse of the onboarding path. See the
two prior specs.
