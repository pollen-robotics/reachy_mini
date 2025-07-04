#!/usr/bin/env python3
# reachy_rhythm_controller.py  – v2.6
"""
Live 5-s rolling plot, robust beat filter, dual beat markers at ±1,
PNG saved on Ctrl-C.  Compatible with Python 3.8+.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import os
import sys
import threading
import time
from dataclasses import dataclass, field

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from dance_moves import AVAILABLE_DANCE_MOVES, MOVE_SPECIFIC_PARAMS

# ───────────────────────────── CLI ──────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dance', default='head_bob_z')
# MODIFIED: Removed --expected-bpm argument
parser.add_argument('--log-mode', choices=['rewrite', 'stream'], default='rewrite')
parser.add_argument('--live-plot', action='store_true', help='Show rolling 5 s plot')
parser.add_argument('--save-dir', default='.', help='Folder for PNG on Ctrl-C')
args = parser.parse_args()

# ───────────────────── Runtime constants ───────────────────────
CONTROL_TS   = 0.02
AUDIO_WIN    = 2.0
BUF_LEN      = int(44100 * AUDIO_WIN)

BPM_BUF      = 4
BPM_STD_TH   = 3.0
UNSTABLE_TMO = 5.0
SILENCE_TMO  = 2.0

PHASE_KP     = 0.8
MAX_RATE     = 0.15
HOLD         = 0.45

LIVE_WIN     = 5.0
PRINT_DT     = 0.25
TOLERANCE    = 0.15            # MODIFIED: Slightly increased tolerance to 15%

NEUTRAL_POS = np.zeros(3)
NEUTRAL_EUL = np.zeros(3)

# ───────────────────── Helper math ─────────────────────────────
def mod_diff(a: float, b: float) -> float:
    """Shortest signed distance on a unit circle (beats)."""
    return ((a - b + 0.5) % 1.0) - 0.5

def head_pose(pos: np.ndarray, eul: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3]   = pos
    m[:3, :3]  = R.from_euler('xyz', eul).as_matrix()
    return m

# ───────────────────── Shared state objects ───────────────────
class MusicState:
    """Data filled by the audio thread, read by the main loop."""
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.goal_bpm = 0.0
        self.last_beat_time = 0.0
        self.raw_bpm = np.nan
        self.state = 'Init'
        self.beats: collections.deque[float] = collections.deque(maxlen=512)

class PLL:
    def __init__(self) -> None:
        self.phase = 0.0
        self.rate  = 1.0

    def step(self, dt: float, bpm: float, last_beat: float,
             now: float) -> tuple[float, float, float]:
        if bpm == 0.0 or last_beat == 0.0:
            return self.phase, 0.0, self.rate

        period = 60.0 / bpm
        expected_phase = ((now - last_beat) / period) % 1.0
        err = mod_diff(expected_phase, self.phase)

        self.rate = 1.0 + np.clip(PHASE_KP * err, -MAX_RATE, MAX_RATE)
        if abs(err) > HOLD:                      # freeze if way off
            return self.phase, err, 0.0

        self.phase = (self.phase + dt * (bpm / 60.0) * self.rate) % 1.0
        return self.phase, err, self.rate

# ───────────────────── Data logging ───────────────────────────
@dataclass
class LogEntry:
    t: float
    raw: float
    stable: float
    err: float
    rate: float
    phase: float

@dataclass
class Logger:
    entries: list[LogEntry] = field(default_factory=list)

    def add(self, **kw) -> None:
        self.entries.append(LogEntry(**kw))

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.array([getattr(e, k) for e in self.entries])
                for k in LogEntry.__annotations__}

logger = Logger()
# MODIFIED: No longer need these as globals for --expected-bpm logic
# first_lock_bpm : float | None = None
# first_lock_time: float | None = None

# ───────────────────── Audio thread ───────────────────────────
def audio_thread(state: MusicState) -> None:
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=44100,
                     input=True, frames_per_buffer=2048)

    buf = np.empty(0, dtype=np.float32)
    bpm_hist = collections.deque(maxlen=BPM_BUF)

    while True:
        buf = np.append(
            buf,
            np.frombuffer(stream.read(2048, exception_on_overflow=False),
                          dtype=np.float32)
        )
        if len(buf) < BUF_LEN:
            continue

        tempo, beat_frames = librosa.beat.beat_track(
            y=buf, sr=44100, units='frames', tightness=100)
        now = time.time()

        with state.lock:
            state.raw_bpm = np.nan if tempo < 40 else float(tempo)
        if tempo > 40:
            bpm_hist.append(float(tempo))

        # Convert beat frames → absolute times
        sr = 44100
        win_dur = len(buf) / sr
        abs_times = [now - (win_dur - librosa.frames_to_time(f, sr=sr))
                     for f in beat_frames]

        with state.lock:
            for t in abs_times:
                if not state.beats or t - state.beats[-1] > 0.05:  # ≥50 ms apart
                    state.beats.append(t)

            if len(bpm_hist) < BPM_BUF:
                state.state = f'Gathering {len(bpm_hist)}/{BPM_BUF}'
            elif np.std(bpm_hist) < BPM_STD_TH:
                state.goal_bpm = float(np.mean(bpm_hist))
                state.state = 'Locked'
                state.last_beat_time = now
            else:
                state.state = 'Unstable'

        buf = buf[-int(44100 * 1.5):]           # keep last 1.5 s

# ───────────────────── Plot helpers ───────────────────────────
def init_live_plot(): # MODIFIED: Doesn't need ref_bpm at init
    plt.ion()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    raw_line, = ax[0].plot([], [], 'o', ms=3, label='Raw', alpha=0.6)
    act_line, = ax[0].plot([], [], '-', label='Active')
    # NEW: Reference line will be added later, so we create a placeholder
    ref_line, = ax[0].plot([], [], linestyle='dashed', label='Reference')
    ax[0].set_ylabel('BPM')
    ax[0].legend()

    sin_line,  = ax[1].plot([], [], '-',  label='sin phase')
    theo_line, = ax[1].plot([], [], '--', label='theoretical')
    acc_up,    = ax[1].plot([], [], '^', color='green', ms=5, label='beat OK')
    acc_dn,    = ax[1].plot([], [], 'v', color='green', ms=5)
    rej_up,    = ax[1].plot([], [], '^', color='red',   ms=5, label='beat NOK')
    rej_dn,    = ax[1].plot([], [], 'v', color='red',   ms=5)
    ax[1].set_ylabel('amp')
    ax[1].set_xlabel('s')
    ax[1].legend()
    fig.tight_layout()

    return fig, ax, (raw_line, act_line, ref_line, sin_line, theo_line,
                     acc_up, acc_dn, rej_up, rej_dn)

def theoretical_curve(bpm: float | None,
                      ts: np.ndarray,
                      t0: float | None) -> np.ndarray:
    if bpm is None or t0 is None: # MODIFIED: Check for t0 as well
        return np.full_like(ts, np.nan)
    return np.sin(2 * np.pi * bpm / 60.0 * (ts - t0))

def refresh_live(lines, ax, t, raw, stab, ref_bpm, amp,
                 theo, acc_x, rej_x) -> None: # MODIFIED: Pass ref_bpm
    rl, al, re, sl, tl, au, ad, ru, rd = lines
    rl.set_data(t, raw)
    al.set_data(t, stab)
    if ref_bpm is not None: # NEW: Only plot reference line if it exists
        re.set_data([t[0], t[-1]], [ref_bpm, ref_bpm])
    sl.set_data(t, amp)
    tl.set_data(t, theo)
    au.set_data(acc_x, np.ones_like(acc_x))
    ad.set_data(acc_x, -np.ones_like(acc_x))
    ru.set_data(rej_x, np.ones_like(rej_x))
    rd.set_data(rej_x, -np.ones_like(rej_x))

    for a in ax:
        a.relim()
        a.autoscale_view()
    plt.pause(0.001)

# ───────────────────── Console printer ───────────────────────
TEMPLATE = (
    '--- Reachy Live Groove ---\n'
    'State: {st}\nRaw BPM: {raw:>6}\nStable BPM: {stab:6.1f}\n'
    'Phase err: {err:+.3f} beat\nRate: {rate:6.1f} %\nΔbeat: {dbeat:>6}\n'
)
def console_print(data: dict, first: bool) -> None:
    if args.log_mode == 'rewrite' and not first:
        sys.stdout.write('\033[7F\033[J')
    print(TEMPLATE.format(**data),
          end='' if args.log_mode == 'rewrite' else '\n')
    sys.stdout.flush()

# ───────────────────── Main loop ─────────────────────────────
def main() -> None:
    # MODIFIED: move these into main's scope
    ref_bpm: float | None = None
    first_beat_time: float | None = None

    music = MusicState()
    threading.Thread(target=audio_thread, args=(music,), daemon=True).start()

    pll = PLL()
    neutral = head_pose(NEUTRAL_POS, NEUTRAL_EUL)

    fig, ax, lines = (init_live_plot()
                      if args.live_plot else (None, None, None))

    move_fn = AVAILABLE_DANCE_MOVES[args.dance]
    params  = MOVE_SPECIFIC_PARAMS.get(args.dance, {})
    phase_offset = 1.0 / (4 * params.get('frequency_factor', 1.0))

    last_loop = time.time()
    last_ui   = 0.0
    last_plot = 0.0
    loop_idx  = 0

    unstable_since: float | None = None
    active_bpm = 0.0
    processed_beats = 0
    accepted: list[float] = []
    rejected: list[float] = []
    dbeat_ms: float | None = None

    # NEW: State for the robust beat filter
    last_accepted_beat_index = -1

    print('Connecting to Reachy Mini…')
    with ReachyMini() as bot:
        bot.set_target(head=neutral, antennas=np.zeros(2))
        time.sleep(1)
        print('Robot ready — play music!\n')

        while True:
            now = time.time()
            dt  = now - last_loop
            last_loop = now

            # Read shared music state
            with music.lock:
                goal_bpm   = music.goal_bpm
                raw_bpm    = music.raw_bpm
                state      = music.state
                last_beat  = music.last_beat_time
                new_beats  = list(music.beats)[processed_beats:]

            processed_beats += len(new_beats)

            # Determine reference BPM (based on first stable lock)
            if state == 'Locked' and ref_bpm is None and goal_bpm > 0:
                ref_bpm = goal_bpm
                print(f"\n--- LOCKED reference BPM: {ref_bpm:.1f} ---\n")

            expected_interval = 60.0 / ref_bpm if ref_bpm else None

            # NEW: Robust beat filtering logic
            if new_beats and ref_bpm is not None and expected_interval is not None:
                for bt in new_beats:
                    # The very first beat anchors our theoretical grid
                    if not accepted:
                        accepted.append(bt)
                        first_beat_time = bt
                        last_accepted_beat_index = 0
                        continue

                    # Calculate where this beat falls on the theoretical grid
                    time_since_anchor = bt - first_beat_time
                    # Round to the nearest beat number
                    beat_index = round(time_since_anchor / expected_interval)

                    # If it's a past or same beat, it's a false positive
                    if beat_index <= last_accepted_beat_index:
                        rejected.append(bt)
                        continue

                    ideal_time = first_beat_time + beat_index * expected_interval
                    error = abs(bt - ideal_time)

                    if error <= expected_interval * TOLERANCE:
                        # Good beat, accept it
                        dbeat_ms = (bt - accepted[-1]) * 1000
                        accepted.append(bt)
                        last_accepted_beat_index = beat_index
                    else:
                        # Bad beat, reject it
                        rejected.append(bt)

            # Handle unstable / silence
            if state == 'Locked':
                active_bpm = goal_bpm
                unstable_since = None
            else:
                unstable_since = unstable_since or now
                if now - unstable_since > UNSTABLE_TMO:
                    active_bpm = 0.0
            if now - last_beat > SILENCE_TMO:
                active_bpm = 0.0

            # PLL & robot control
            if active_bpm == 0.0:
                bot.set_target(head=neutral, antennas=np.zeros(2))
                phase, err, rate = pll.phase, 0.0, 0.0
            else:
                phase, err, rate = pll.step(dt, active_bpm, last_beat, now)
                if rate == 0.0:
                    bot.set_target(head=neutral, antennas=np.zeros(2))
                else:
                    beat_phase = (phase + phase_offset) % 1.0
                    offs = move_fn(beat_phase, **params)
                    bot.set_target(
                        head=head_pose(NEUTRAL_POS +
                                       offs.get('position_offset', np.zeros(3)),
                                       NEUTRAL_EUL +
                                       offs.get('orientation_offset', np.zeros(3))),
                        antennas=offs.get('antennas_offset', np.zeros(2))
                    )

            logger.add(t=now, raw=raw_bpm, stable=active_bpm,
                       err=err, rate=rate * 100.0, phase=phase)

            # Console output
            if now - last_ui > PRINT_DT:
                console_print({
                    'st'   : state,
                    'raw'  : '--' if np.isnan(raw_bpm) else f'{raw_bpm:6.1f}',
                    'stab' : active_bpm,
                    'err'  : err,
                    'rate' : rate * 100.0,
                    'dbeat': '--' if dbeat_ms is None else f'{dbeat_ms:6.0f}'
                }, first=(loop_idx == 0))
                last_ui = now
                loop_idx += 1

            # Live plot refresh
            if args.live_plot and now - last_plot > PRINT_DT:
                arr = logger.arrays()
                tvals = arr['t'] - arr['t'][0]
                mask = tvals >= tvals[-1] - LIVE_WIN

                amp_vals  = np.sin(2 * np.pi * arr['phase'])
                theo_vals = theoretical_curve(ref_bpm, arr['t'], first_beat_time)

                acc_x = np.array([bt - arr['t'][0]
                                  for bt in accepted
                                  if bt >= arr['t'][0] + tvals[-1] - LIVE_WIN])
                rej_x = np.array([bt - arr['t'][0]
                                  for bt in rejected
                                  if bt >= arr['t'][0] + tvals[-1] - LIVE_WIN])

                refresh_live(lines, ax,
                             tvals[mask],
                             arr['raw'][mask],
                             arr['stable'][mask],
                             ref_bpm, # MODIFIED: pass ref_bpm to plotting
                             amp_vals[mask],
                             theo_vals[mask],
                             acc_x,
                             rej_x)
                last_plot = now

            time.sleep(max(0, CONTROL_TS - (time.time() - now)))

# ───────────────────── Save live plot ─────────────────────────
def save_live_png() -> None:
    """Save current live figure to PNG."""
    if not args.live_plot:
        return
    os.makedirs(args.save_dir, exist_ok=True)
    fname = f'reachy_live_{datetime.datetime.now():%Y%m%d_%H%M%S}.png'
    path  = os.path.join(args.save_dir, fname)
    plt.gcf().savefig(path, dpi=150)
    print(f'\nPNG saved to {path}')

# ───────────────────── Entrypoint ─────────────────────────────
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        save_live_png()
        if args.live_plot:
            plt.show() # keep plot open after Ctrl-C