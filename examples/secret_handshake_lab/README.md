# Secret handshake lab

Offline + on-robot harness for the secret handshake. This is the pre-daemon
playground: the collision definition AND the full two-round handshake state
machine are validated here, then the same pure logic gets promoted into the
daemon control loop.

See the design spec: `docs/superpowers/specs/2026-07-01-secret-handshake-design.md`

## The gesture

Torque off, head in the sleep pose, then:

1. 3 antenna collisions (knock them together) -> confirmation beep (PRIMED)
2. within 8 s, either:
   - 3 more collisions -> handshake A fanfare (v1 action: the emotion)
   - rub the antennas together ~1 s -> handshake B fanfare (future: WiFi)

Do nothing after the beep and it buzzes low and resets. Torque on kills it
instantly. Everything partial decays on its own.

## The collision law (v2, 2026-07-02) [UNDER REVISION]

STATUS: Remi rejected the velocity-based definition below as not robust
enough for the final feature and is deriving a purely GEOMETRIC definition
(angle regions where the antennas collide) by hand, using `position_gui.py`
(torque toggles + live present positions + snapshot logging). The sections
below describe the v2 law as implemented in this lab today.

`m = min(|v0|, |v1|)` where v0, v1 are the antenna angular velocities
(measured over ~40 ms). A collision is a spike `m > 2.0` rad/s; a rub is
sustained `m > 0.25`.

Why not angles? The first version used `diff = ant0 - ant1` with absolute
thresholds and FAILED LIVE: the antennas are floppy friction-fit parts that
rest wherever they were last left. The live robot rested at diff=+0.97 where
the dataset robot rested at -0.35, which latched the old detector
permanently "in contact" so no collision could ever fire. Worse, the same
angle pair can occur touching (paused mid-slide) and not touching (parked
crossed after sliding one antenna over the other), so NO function of the
instantaneous angles can define contact. Coupled motion can: when the
antennas touch, moving one moves the other; when they do not, the
non-driven antenna stays still.

Validation on the dataset (`RemiFabre/secret-handshake`, see
`analyze_recordings.py` and `replay_validate.py`):

- Every audible knock in the 4 `default*.json` gestures appears as an
  m-spike of 4-9 rad/s (bring-together clack + 3 taps = 4-6 collisions).
- The 30 s of `collision-definition*.json` (antennas touching and sliding
  nearly the whole time) contain ZERO spikes above 1.2, and show sustained
  coupling of 0.3-1.2, which is what the rub gesture uses.
- The full machine PRIMES on the recorded gestures (the old law never did)
  and never fires an action on any recording.
- Single-antenna motion of any speed gives m ~ 0: playing with one antenna
  cannot trigger anything (the old law false-positived on this).
- `default4.json` never arms: the head sat ~20 deg shallower than the sleep
  pose (the spec's pitch outlier). The gate is doing its job; the live ARMED
  tick sound tells you when the pose is accepted.

## Files

- `collision.py` - pure coupled-motion detector (knock spikes + sustained
  coupling). The exact primitive for the 50 Hz control loop.
- `handshake.py` - pure `HandshakeStateMachine` + `HandshakeConfig`
  (~0.4 us per update). Returns events; never does I/O itself.
- `pose_gate.py` - pure "is the head in the sleep pose?" check (generous
  tolerances, sampled only while idle).
- `beeps.py` - stdlib beep rendering/playback for the lab scripts.
  `python beeps.py` auditions all 5 sounds.
- `test_handshake.py` - 23 synthetic 50 Hz unit tests (detector, both
  handshakes, timeouts, single-antenna immunity, base-position immunity,
  torque resets, repeatability, pose gate).
- `analyze_recordings.py` - the investigation tool: prints/plots the signals
  behind the law (`--plot` saves PNGs).
- `replay_validate.py` - hard regression over the HF recordings: knocks
  found on gestures, zero on slides, machine primes on gestures, no action
  ever fires.
- `live_handshake_probe.py` - ON THE ROBOT: full state machine with beeps.
  This is the thing to test.
- `sim_keyboard.py` - NO ROBOT: same loop and beeps, Enter = collision,
  r+Enter = rub. Rehearse the flow at a desk.
- `live_contact_probe.py` - on-robot raw signal printer (no state machine,
  no beeps): watch ant0/ant1/m live while tuning CollisionConfig.
- `position_gui.py` - mini_head_position_gui.py extended for the manual
  collision investigation: torque toggle buttons (head / antennas
  separately) with a live present-position readout that keeps streaming
  while torque is off, and a Snapshot button (Spacebar) that appends the
  current angles to `antenna_snapshots.csv`. Move the antennas by hand,
  snapshot the interesting configurations, derive the geometry.

## Run

```bash
# no robot: unit tests, recorded-data regression, keyboard rehearsal
python -m pytest examples/secret_handshake_lab/test_handshake.py -v
python examples/secret_handshake_lab/replay_validate.py
python examples/secret_handshake_lab/sim_keyboard.py

# on the robot (torque off so the antennas are floppy)
python examples/secret_handshake_lab/live_handshake_probe.py
python examples/secret_handshake_lab/live_handshake_probe.py --no-pose-gate   # desk mode
python examples/secret_handshake_lab/live_handshake_probe.py --disable-motors # torque off from here
```

The live status line shows `m` (coupled speed): ~0 at rest and when moving a
single antenna, spikes above 2 on each knock, 0.3-1.2 while rubbing.
