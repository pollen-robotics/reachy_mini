# Secret handshake lab

Offline + on-robot harness for the secret handshake. This is the pre-daemon
playground: the collision definition AND the full two-round handshake state
machine are validated here, then the same pure logic gets promoted into the
daemon control loop.

See the design spec: `docs/superpowers/specs/2026-07-01-secret-handshake-design.md`

## The gesture

Torque off, head in the sleep pose, then:

1. 3 antenna collisions -> confirmation beep (PRIMED, the shared prefix)
2. within 8 s, either:
   - 3 more collisions -> handshake A fanfare (v1 action: the emotion)
   - press-and-hold ~1.2 s -> handshake B fanfare (future: WiFi provisioning)

Do nothing after the beep and it buzzes low and resets. Torque on kills it
instantly. Everything partial decays on its own.

## Files

- `collision.py` - pure contact primitives: `CollisionDetector` (edge with
  hysteresis) and `KnockDetector` (pressure peaks during a held contact).
- `handshake.py` - pure `HandshakeStateMachine` + `HandshakeConfig`. The exact
  logic destined for the 50 Hz control loop (~0.1 us per update). Returns
  events; never does I/O itself.
- `pose_gate.py` - pure "is the head in the sleep pose?" check (generous
  tolerances, sampled only while idle).
- `beeps.py` - stdlib beep rendering/playback for the lab scripts.
  `python beeps.py` auditions all 5 sounds.
- `test_handshake.py` - 20 synthetic 50 Hz unit tests (both handshakes, both
  counters, timeouts, debounce, torque resets, repeatability, pose gate).
- `replay_validate.py` - replays the recorded HF dataset through the detector
  AND the full machine. Regression: edge mode must never prime on it.
- `live_handshake_probe.py` - ON THE ROBOT: full state machine with beeps.
  This is the thing to test.
- `sim_keyboard.py` - NO ROBOT: same loop and beeps, Enter = tap,
  h+Enter = hold. Rehearse the flow at a desk.
- `live_contact_probe.py` - on-robot raw contact printer (the older,
  lower-level tester; still useful to stare at `diff` while tuning).

## Run

```bash
# no robot: unit tests, recorded-data regression, keyboard rehearsal
python -m pytest examples/secret_handshake_lab/test_handshake.py -v
python examples/secret_handshake_lab/replay_validate.py
python examples/secret_handshake_lab/sim_keyboard.py

# on the robot (torque off so the antennas are floppy)
python examples/secret_handshake_lab/live_handshake_probe.py
python examples/secret_handshake_lab/live_handshake_probe.py --counter knock
python examples/secret_handshake_lab/live_handshake_probe.py --no-pose-gate   # desk mode
python examples/secret_handshake_lab/live_handshake_probe.py --disable-motors # torque off from here
```

## Tap counters (pick live, spec section 5)

- `edge` (default, strict): a tap = a NEW contact; you must release between
  taps. On the recorded data this can NEVER false-positive (the machine never
  primes on any recording).
- `knock` (fallback, lenient): pressure peaks while the antennas stay together
  also count. On the recordings it primes exactly where the human did the
  3-knock gesture (default 1-3 at ~2.0-2.5 s), BUT free antenna play
  (collision-definition files) can fully trigger it. Prefer `edge` if the
  beep feedback teaches people to separate between taps.

## Findings from the recorded data

- Edge machine on all 6 recordings: arms, never primes. Zero false positives.
- Knock machine hears the real 3-knock gesture (primes on default 1-3) and is
  too permissive on deliberate antenna sweeping.
- `default4.json` never arms: the head sat ~17-23 deg shallower than the sleep
  pose (the spec's pitch outlier). The gate is doing its job; the live ARMED
  tick sound tells you when the pose is accepted.

## Contact definition (validated)

`diff = ant0 - ant1` (ant0 = left, ant1 = right, radians).
Rest ~ -0.35, contact > ~2. Hysteresis: on at `diff > 0.5`, off at `diff < 0`.
Knocks: prominence 0.25 rad, refractory 150 ms.
