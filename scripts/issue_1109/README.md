# Issue #1109 investigation tooling

Two on-robot scripts to support the conversation on PR #1110:

- `burst_characterization.py` — measure motor-EMI burst duration on GPIO23.
- `hold_time_sweep.py` — sweep `HOLD_TIME` values, find the floor that
  survives motor stress with zero spurious shutdowns.

Both scripts run **on the Reachy Mini's Raspberry Pi**, not on the Mac.
Copy with `scp -r scripts/issue_1109 pollen@reachy-mini.local:/home/pollen/`
or equivalent.

## Pre-conditions (both scripts)

```sh
# Take exclusive ownership of GPIO23 (otherwise the daemon's own monitor
# will also fire callbacks, and may shut the robot down mid-test):
sudo systemctl stop gpio-shutdown-daemon

# The reachy-mini-daemon must be running for set_target to work:
systemctl status reachy-mini-daemon

# Do not touch the latch during these runs. We're characterizing
# motor-EMI alone, so any deliberate latch pull would contaminate the
# data.
```

Restore the daemon after testing:

```sh
sudo systemctl start gpio-shutdown-daemon
```

## Workflow

### 1. Characterize EMI burst duration

```sh
# 5-minute run with 10 Hz set_target stress (matches #1109's reproducer)
python3 burst_characterization.py --duration 300 --freq 10

# Optional baseline: passive EMI floor (no motor activity)
python3 burst_characterization.py --duration 60 --no-motor
```

Output: `logs/issue_1109/burst-<UTC>.jsonl` plus a stdout summary with
burst duration p50 / p95 / p99 and inter-edge gap counts (<1 ms, <10 ms,
<100 ms).

**What the numbers tell us:** the production Timer cancels on any
`when_pressed` edge during the hold window, so what matters is **the
maximum gap between adjacent edges inside a single burst** —
HOLD_TIME just needs to outlast one such gap. Burst durations from
this script give us a measured upper bound to compare against the
"sub-millisecond" claim in the PR description.

### 2. Sweep HOLD_TIME to find the safe floor

```sh
# 6-value sweep at 2 min/step = ~12 min total
python3 hold_time_sweep.py \
    --hold-times 0.05 0.1 0.2 0.5 1.0 2.0 \
    --duration-per-step 120 --freq 10

# Quick smoke (narrower range, shorter duration)
python3 hold_time_sweep.py \
    --hold-times 0.1 0.2 0.5 \
    --duration-per-step 60
```

Output: `logs/issue_1109/sweep-<UTC>.jsonl` plus a stdout table.

**Pass criterion:** a HOLD_TIME passes if `n_timers_fired == 0` after
its full step duration. The **floor** is the smallest passing value.
The `pull_up=False` setting matches production
`shutdown_monitor.py`, so the edge semantics are identical.

The harness replaces `shutdown_now()` with a counter — your robot will
not actually shut down regardless of HOLD_TIME or motor activity.

### 3. Pick HOLD_TIME for the PR

Bound the safe range from both directions:

- **Floor** (from `hold_time_sweep.py`): smallest value with 0 spurious
  fires during sustained motor stress.
- **Ceiling** (from the review on PR #1110): ~2 s, after which the
  Wireless latch-OUT power cut beats `shutdown_now()` to completion.

### Empirical result on Reachy Mini Wireless (2026-05-12)

`burst_characterization.py --duration 300 --freq 10` produced **0
edges** across 2,887 motor commands. `hold_time_sweep.py
--hold-times 0.05 0.1 0.2 0.5 1.0 2.0 --duration-per-step 120 --freq
10` produced **0 spurious fires at every HOLD_TIME tested**. Total: 17
minutes of stress, ~10,000 motor commands, zero GPIO23 edges.

On this hardware the EMI condition described in #1109 is below the
measurement floor, so the floor for HOLD_TIME is set by gesture-
recognition UX, not EMI. **HOLD_TIME = 0.5 s** was chosen: 10× a typical
e-stop tap (~50 ms), 4× headroom below the ~2 s power-rail cut.

## Notes

- Both scripts use `gpiozero.Button(23, pull_up=False)`, which matches
  the production `shutdown_monitor.py` and defers pull configuration to
  the kernel device tree.
- The motor-stress loop drives `head` (sinusoidal yaw), `antennas`
  (sweeping), and `body_yaw` (slow oscillation) simultaneously — heavier
  than a single-joint loop, intended to maximize EMI per unit wall-clock.
- These scripts are not part of the PR's deliverable; they live in
  `scripts/issue_1109/` for the duration of the #1109 / #1110
  investigation and can be removed before merge if Pierre prefers a
  smaller diff.
