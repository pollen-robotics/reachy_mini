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
   - hold them gently together ~1 s -> handshake B fanfare (future: WiFi)

Do nothing after the beep and it buzzes low and resets. Torque on kills it
instantly. Everything partial decays on its own.

## The collision law (v3, geometric, 2026-07-03)

Measured by Remi by hand on 2 robots (using `position_gui.py`), in degrees.
The antennas are in collision "at the center" (y = 0 plane) when BOTH:

    1) l_ant + r_ant in [-7, -3]   (~= 0 with a slight consistent asymmetry)
    2) l_ant        in [20, 150]

widened by a margin into the working band, default [-9, 0]: live testing on
a 3rd robot (different batch, 100% of intended collisions detected) showed
many collisions at the old -1 top edge, so the top was relaxed to 0, the
theoretical symmetric contact point. The bottom stays at -9: the
crossed-parked state (antennas slid past each other) sits at sum ~ -11.
`l_ant` = antenna index 0, `r_ant` = index 1, as returned by
`get_present_antenna_joint_positions()`. A collision EVENT = entering that
region; a 0.25 s refractory merges the double band-crossing of a firm press
(pressing flexes the antennas: sum shoots up to +35..+73 deg, then passes
back through the band on release).

Validation on the dataset (`RemiFabre/secret-handshake`, see
`replay_validate.py`, hard regression):

- EXACTLY 3 collisions on each of the 4 `default*.json` gestures, at the
  audible knock times. The bring-together does not count: condition 2
  filters the approach (l still below 20 deg when the band is crossed).
- Rest is inert: sum (~-2) is in the band but l (~-12) fails condition 2.
  Same for single-antenna play and the crossed-parked state (sum ~ -11).
- The full machine primes on the recorded gestures and fires no action on
  them. `default4.json` never arms (head ~20 deg shallower than the sleep
  pose, the known outlier; the gate is doing its job).
- CAVEAT to judge live: slow slide play (`collision-definition*.json`)
  passes through and lingers in the band, so it can prime and even complete
  the hold handshake. The real protection is the two-round structure + the
  sleep-pose gate + torque-off arming.

History, kept so it is not retried: v1 was `diff = ant0 - ant1` with
absolute thresholds; it failed live because the floppy antennas rest
wherever they were left (one robot rested at diff +56 deg, another at -20).
v2 was velocity-based coupled motion; it validated offline but Remi rejected
dynamics as not robust enough for the final feature and measured the
geometric definition above instead.

## Files

- `collision.py` - the pure geometric collision detector (the definition
  above). The exact primitive for the 50 Hz control loop.
- `handshake.py` - pure `HandshakeStateMachine` + `HandshakeConfig`
  (~0.2 us per update). Returns events; never does I/O itself.
- `pose_gate.py` - pure "is the head in the sleep pose?" check (generous
  tolerances, sampled only while idle).
- `beeps.py` - stdlib beep rendering/playback for the lab scripts.
  `python beeps.py` auditions all 5 sounds.
- `test_handshake.py` - 25 synthetic 50 Hz unit tests (detector, both
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
  h+Enter = hold. Rehearse the flow at a desk.
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

# on the robot (both probes turn torque OFF at startup so the antennas are
# floppy; pass --keep-torque to leave the motors alone)
python examples/secret_handshake_lab/live_handshake_probe.py
python examples/secret_handshake_lab/live_handshake_probe.py --no-pose-gate   # desk mode
```

The live status lines show `sum` and `l` in degrees plus an IN-BAND flag, so
you can see exactly why a collision does or does not register.
