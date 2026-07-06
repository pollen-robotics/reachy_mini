# Secret handshake lab

Offline + on-robot harness for the secret handshake. This is the pre-daemon
playground: the collision definition AND the full two-round handshake state
machine are validated here, then the same pure logic gets promoted into the
daemon control loop.

STATUS: promoted. The daemon now ships the default handshake (3 taps ->
beep -> 3 taps -> success sound) in
`src/reachy_mini/daemon/backend/secret_handshake.py` (kill switch:
`REACHY_HANDSHAKE_ENABLED=0`), with sounds rendered from the tap-lab tones
by `render_daemon_sounds.py`. The lab remains the place to tune and
re-validate before changing the daemon module. The hold gesture (handshake
B) exists only in the lab for now.

See the design spec: `docs/superpowers/specs/2026-07-01-secret-handshake-design.md`

## The gesture

Torque off, head in the sleep pose, then:

1. 3 antenna collisions in quick succession -> confirmation beep (PRIMED).
   A 1 s pause between collisions resets the count.
2. within 3 s, either:
   - 3 more collisions -> handshake A fanfare (v1 action: the emotion)
   - hold them gently together ~1 s -> handshake B fanfare (future: WiFi)

Do nothing after the beep and it buzzes low and resets. Torque on kills it
instantly. Everything partial decays on its own.

## Tunables at a glance

One source of truth: `CollisionConfig` (collision.py) and `HandshakeConfig`
(handshake.py). Current values:

| what | value |
|---|---|
| collision region | `l + r` in [-9, 0] deg AND `l` in [20, 150] deg (angles wrapped to [-180, 180): the encoders are multi-turn) |
| collision debounce | 0.25 s refractory (a >0.25 s press counts twice: known accepted quirk, see `CollisionConfig`) |
| collisions per round | 3 |
| sequence reset | 1.0 s without a collision |
| arming settle | head in sleep pose for 0.5 s (idle only: boot and torque-off transitions) |
| round 2 window | 3.0 s after the prime beep |
| hold gesture | 1.0 s in the region (0.3 s flicker grace) |
| after an action or abort | straight back to armed: immediate retry works |

## Control-loop cost (bench.py)

The daemon integration is ONE call per 50 Hz tick:
`SecretHandshake.update(t, ant0, ant1, head_pose, torque_off)`. Measured
per-call cost (M-series Mac, numpy 4x4 pose, median of 5x50k calls):

| scenario | ns/call | share of the 20 ms tick |
|---|---|---|
| torque ON (all normal robot use) | 115 | 0.0006% |
| idle, head up (pose gate fails) | 460 | 0.0023% |
| armed, antennas at rest | 278 | 0.0014% |
| armed, continuous tap traffic (worst) | 874 | 0.0044% |
| primed, holding in the band | 295 | 0.0015% |

Worst case is ~23,000x smaller than the tick budget. On the CM4 (Wireless
robots) expect roughly 5-15x slower, still under 0.07% of the budget.

## Latency note

The state machine reacts on the exact tick. What feels slow in the lab is
the TEMPORARY sound path: each beep spawns an afplay/aplay process
(~150-400 ms to open the audio device). The daemon will play through the
robot speaker instead.

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

### Measured touch-sum envelope (static sweeps, 2026-07-03)

`data/touch_sweep_*.csv`: antennas held gently touching, swept l from ~26 to
~145 deg, one pass per stacking order (which antenna rests on top), on a
Wireless and a Lite robot. Sum of the two angles while touching, in degrees:

    Wireless: [-5.4, -1.5]  (mean -3.1)
    Lite    : [-6.8, -1.8]  (mean -4.5)

Robot-to-robot centers differ by ~1.3 deg. The STACKING ORDER is the
biggest term: at mid-travel (l 60-110 deg) the second stacking sits 2.5-3.4
deg lower than the first, shrinking to ~0 (or slightly reversed) at both
ends of travel; the same pattern on both robots. The union of everything
measured, [-6.8, -1.5], sits inside the working band [-9, 0] with ~2 deg to
spare on each side; the bottom edge -9 is also almost exactly the midpoint
between the deepest real touch (-6.8) and the crossed-parked state (~-11).

History, kept so it is not retried: v1 was `diff = ant0 - ant1` with
absolute thresholds; it failed live because the floppy antennas rest
wherever they were left (one robot rested at diff +56 deg, another at -20).
v2 was velocity-based coupled motion; it validated offline but Remi rejected
dynamics as not robust enough for the final feature and measured the
geometric definition above instead.

## Files

- `collision.py` - the pure geometric collision detector (the definition
  above). The exact primitive for the 50 Hz control loop.
- `handshake.py` - pure `HandshakeStateMachine` + `HandshakeConfig`, and
  `SecretHandshake`: the single-call facade the daemon loop will use.
  Returns events; never does I/O itself. Tunables banner at the top.
- `pose_gate.py` - pure "is the head in the sleep pose?" check (generous
  tolerances, sampled only while idle).
- `beeps.py` - stdlib beep rendering/playback for the lab scripts.
  `python beeps.py` auditions all 5 sounds.
- `test_handshake.py` - 28 synthetic 50 Hz unit tests (detector, both
  handshakes, quick-succession timing, single-antenna immunity, torque
  resets, repeatability, pose gate, the SecretHandshake facade).
- `bench.py` - control-loop cost benchmark (table above).
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
