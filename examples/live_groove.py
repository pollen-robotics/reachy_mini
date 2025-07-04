#!/usr/bin/env python3
# reachy_rhythm_controller.py  – v6.2 (Final, Readable)
"""
Real-time robot rhythm synchronization using a decoupled architecture.

This script synchronizes a Reachy Mini robot's movements to music detected
from a microphone. It features:
- A high-frequency (100 Hz), stable control loop.
- A separate thread for audio processing using Librosa to detect beat candidates.
- A live beat filtering algorithm to reject spurious detections.
- A robust BPM calculation based on the median of recent filtered beat intervals.
- A dynamically-toggleable Phase-Locked Loop (PLL) to correct the robot's phase
  against the filtered beats, ensuring tight, long-term synchronization.
- A non-blocking UI thread for live plotting and statistics.

To run this script, you will need to install the 'pynput' library:
  pip install pynput
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
from queue import Queue, Empty

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from dance_moves import AVAILABLE_DANCE_MOVES, MOVE_SPECIFIC_PARAMS

# ───────────────────────────── Configuration ──────────────────────────────
@dataclass
class Config:
    """A single object to hold all tunable parameters and constants."""
    # --- CLI Arguments ---
    dance_move: str = 'head_bob_z'
    live_plot: bool = False
    save_dir: str = '.'

    # --- Core ---
    control_ts: float = 0.01  # Target control loop period (0.01s = 100 Hz).

    # --- Audio Processing ---
    audio_win: float = 2.0
    audio_rate: int = 44100
    audio_chunk_size: int = 2048
    bpm_stability_buffer: int = 4
    bpm_stability_threshold: float = 3.0
    silence_tmo: float = 2.0

    # --- Beat Filtering & BPM Calculation ---
    beat_buffer_size: int = 20
    min_interval_factor: float = 0.5

    # --- Synchronization & PLL ---
    pll_kp: float = 0.8
    pll_max_rate: float = 0.15
    pll_hold_threshold: float = 0.45
    beat_time_offset: float = 0.0

    # --- UI & Logging ---
    ui_update_rate: float = 2.0
    plot_window_duration: float = 5.0
    
    # --- Static ---
    neutral_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    neutral_eul: np.ndarray = field(default_factory=lambda: np.zeros(3))
    audio_buffer_len: int = field(init=False)

    def __post_init__(self):
        self.audio_buffer_len = int(self.audio_rate * self.audio_win)


# ───────────────────── Helper Classes and Functions ──────────────────────────
class SharedState:
    """A thread-safe object for managing state shared between threads."""
    def __init__(self):
        self.lock = threading.Lock()
        self.pll_enabled = True

    def toggle_pll(self) -> bool:
        with self.lock:
            self.pll_enabled = not self.pll_enabled
            return self.pll_enabled

def head_pose(pos: np.ndarray, eul: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = pos
    m[:3, :3] = R.from_euler('xyz', eul).as_matrix()
    return m

class MusicState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.librosa_bpm = 0.0
        self.last_event_time = 0.0
        self.state = 'Init'
        self.beats: collections.deque[float] = collections.deque(maxlen=512)

class PLL:
    def __init__(self) -> None:
        self.phase = 0.0
        self.rate = 1.0

    def step(self, dt: float, bpm: float, last_beat_time: float, now: float, config: Config) -> tuple[float, float, float]:
        if bpm == 0.0 or last_beat_time == 0.0:
            return self.phase, 0.0, 1.0
        period = 60.0 / bpm
        expected_phase = ((now - last_beat_time) / period) % 1.0
        err = ((expected_phase - self.phase + 0.5) % 1.0) - 0.5
        self.rate = 1.0 + np.clip(config.pll_kp * err, -config.pll_max_rate, config.pll_max_rate)
        if abs(err) > config.pll_hold_threshold:
            return self.phase, err, 0.0
        self.phase = (self.phase + dt * (bpm / 60.0) * self.rate) % 1.0
        return self.phase, err, self.rate

# ───────────────────────────── Worker Threads ──────────────────────────────
def keyboard_listener_thread(shared_state: SharedState, stop_event: threading.Event):
    def on_press(key):
        if stop_event.is_set():
            return False
        try:
            if key.char == 'p':
                new_state = shared_state.toggle_pll()
                print(f"\n--- PLL Correction Toggled: {'ON' if new_state else 'OFF'} ---\n")
        except AttributeError:
            pass
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def audio_thread(state: MusicState, config: Config, stop_event: threading.Event) -> None:
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=config.audio_rate,
                     input=True, frames_per_buffer=config.audio_chunk_size)
    buf = np.empty(0, dtype=np.float32)
    bpm_hist = collections.deque(maxlen=config.bpm_stability_buffer)

    while not stop_event.is_set():
        try:
            audio_chunk = np.frombuffer(stream.read(config.audio_chunk_size, exception_on_overflow=False), dtype=np.float32)
            buf = np.append(buf, audio_chunk)
        except (IOError, ValueError):
            continue

        if len(buf) < config.audio_buffer_len:
            continue

        tempo, beat_frames = librosa.beat.beat_track(y=buf, sr=config.audio_rate, units='frames', tightness=100)
        now = time.time()
        tempo_val = tempo[0] if isinstance(tempo, np.ndarray) and tempo.size > 0 else tempo

        with state.lock:
            state.last_event_time = now
            if tempo_val > 40:
                bpm_hist.append(float(tempo_val))
                state.librosa_bpm = float(np.mean(bpm_hist))

            win_dur = len(buf) / config.audio_rate
            abs_times = [now - (win_dur - librosa.frames_to_time(f, sr=config.audio_rate)) for f in beat_frames]

            for t in abs_times:
                if not state.beats or t - state.beats[-1] > 0.05:
                    state.beats.append(t + config.beat_time_offset)
            
            if len(bpm_hist) < config.bpm_stability_buffer:
                state.state = 'Gathering'
            elif np.std(bpm_hist) < config.bpm_stability_threshold:
                state.state = 'Locked'
            else:
                state.state = 'Unstable'

        buf = buf[-int(config.audio_rate * 1.5):]
    
    stream.stop_stream()
    stream.close()
    pa.terminate()

def ui_thread(data_queue: Queue, config: Config, shared_state: SharedState, stop_event: threading.Event):
    fig = None
    if config.live_plot:
        plt.ion()
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6), constrained_layout=True)
        librosa_line, = ax[0].plot([], [], 'o', ms=3, label='Librosa BPM', alpha=0.6)
        calc_line, = ax[0].plot([], [], '-', label='Calculated BPM')
        ax[0].set_ylabel('BPM')
        ax[0].set_ylim(0, 200)
        ax[0].legend()
        
        phase_line, = ax[1].plot([], [], '-', label='PLL Phase')
        ax[1].plot([], [], color='g', linestyle='-', label='Accepted Beat')
        ax[1].plot([], [], color='r', linestyle='--', label='Rejected Beat')
        ax[1].set_ylabel('Phase/Beat')
        ax[1].set_xlabel('Time (s)')
        ax[1].legend()

    log_buffer_size = int((1.0 / config.control_ts) * config.plot_window_duration * 1.2)
    log_entries = collections.deque(maxlen=log_buffer_size)
    loop_dts, last_ui_print_time, last_data = [], time.time(), None

    while not stop_event.is_set():
        try:
            while True:
                data = data_queue.get_nowait()
                loop_dts.append(data['loop_dt'])
                log_entries.append(data)
                last_data = data
        except Empty:
            pass

        now = time.time()
        if not last_data or now - last_ui_print_time < (1.0 / config.ui_update_rate):
            time.sleep(0.1)
            continue
        
        last_ui_print_time = now
        if loop_dts:
            dts_ms = np.array(loop_dts) * 1000
            stats = {'mean': np.mean(dts_ms), 'var': np.var(dts_ms), 'max': np.max(dts_ms), 'num': len(dts_ms)}
        else:
            stats = {'mean': 0, 'var': 0, 'max': 0, 'num': 0}

        with shared_state.lock:
            pll_status = 'ON' if shared_state.pll_enabled else 'OFF'

        print(f"\n--- Reachy Rhythm Controller ---\n"
              f"Control Loop (last {stats['num']} loops): Mean: {stats['mean']:.1f}ms | Var: {stats['var']:.2f} | Max: {stats['max']:.1f}ms\n"
              f"State: {last_data['state']:<10} [Smart Correction (P key): {pll_status}]\n"
              f"Librosa BPM: {last_data['librosa_bpm']:6.1f}\n"
              f"Calc. BPM:   {last_data['calculated_bpm']:6.1f}")
        sys.stdout.flush()
        loop_dts.clear()
        
        if config.live_plot and fig:
            if len(log_entries) < 2:
                continue

            t = np.array([e['t'] for e in log_entries])
            start_time = log_entries[0]['t']
            t -= start_time
            mask = t >= t[-1] - config.plot_window_duration
            if not np.any(mask):
                continue
            
            librosa_bpm = np.array([e['librosa_bpm'] for e in log_entries])
            calc_bpm = np.array([e['calculated_bpm'] for e in log_entries])
            phase = np.sin(2 * np.pi * np.array([e['phase'] for e in log_entries]))
            
            librosa_line.set_data(t[mask], librosa_bpm[mask])
            calc_line.set_data(t[mask], calc_bpm[mask])
            phase_line.set_data(t[mask], phase[mask])
            
            for collection in ax[1].collections[:]:
                collection.remove()

            acc_beats = np.array([b for e in log_entries for b in e['accepted_beats']]) - start_time
            rej_beats = np.array([b for e in log_entries for b in e['rejected_beats']]) - start_time
            
            ax[1].vlines(acc_beats[acc_beats > t[mask][0]], -1, 1, colors='g', linestyles='solid')
            ax[1].vlines(rej_beats[rej_beats > t[mask][0]], -1, 1, colors='r', linestyles='dashed')
            
            for a in ax:
                a.relim()
                a.autoscale_view(scalex=False)
            
            ax[0].set_ylim(0, 200)
            ax[1].set_ylim(-1.1, 1.1)
            plt.pause(0.001)

    print("\nUI thread stopped.")
    if config.live_plot and fig:
        os.makedirs(config.save_dir, exist_ok=True)
        fname = f'reachy_live_{datetime.datetime.now():%Y%m%d_%H%M%S}.png'
        path = os.path.join(config.save_dir, fname)
        fig.savefig(path, dpi=150)
        print(f"Plot saved to {path}")
        plt.ioff()
        plt.show()

# ───────────────────────────── Main Control Loop ─────────────────────────────
def main(config: Config) -> None:
    data_queue, stop_event, music, shared_state = Queue(), threading.Event(), MusicState(), SharedState()
    
    threading.Thread(target=audio_thread, args=(music, config, stop_event), daemon=True).start()
    threading.Thread(target=keyboard_listener_thread, args=(shared_state, stop_event), daemon=True).start()
    ui_thread_obj = threading.Thread(target=ui_thread, args=(data_queue, config, shared_state, stop_event), daemon=True)
    ui_thread_obj.start()
    
    pll = PLL()
    neutral = head_pose(config.neutral_pos, config.neutral_eul)
    move_fn = AVAILABLE_DANCE_MOVES[config.dance_move]
    params = MOVE_SPECIFIC_PARAMS.get(config.dance_move, {})
    phase_offset = 1.0 / (4 * params.get('frequency_factor', 1.0))
    
    last_loop = time.time()
    processed_beats, active_bpm = 0, 0.0
    filtered_beat_times = collections.deque(maxlen=config.beat_buffer_size)
    
    print('Connecting to Reachy Mini…')
    with ReachyMini() as bot:
        bot.set_target(head=neutral, antennas=np.zeros(2))
        time.sleep(1)
        print('Robot ready — play music! (Press P to toggle smart correction)\n')
        
        try:
            while True:
                now = time.time()
                dt = now - last_loop
                last_loop = now
                
                with music.lock:
                    librosa_bpm = music.librosa_bpm
                    state = music.state
                    last_event_time = music.last_event_time
                    new_beats = list(music.beats)[processed_beats:]
                processed_beats += len(new_beats)
                
                accepted_this_frame, rejected_this_frame = [], []
                if new_beats:
                    ref_bpm = active_bpm if active_bpm > 0 else librosa_bpm
                    if ref_bpm > 0:
                        expected_interval = 60.0 / ref_bpm
                        min_interval = expected_interval * config.min_interval_factor
                        i = 0
                        while i < len(new_beats):
                            last_beat = filtered_beat_times[-1] if filtered_beat_times else new_beats[i] - expected_interval
                            current_beat = new_beats[i]
                            if i + 1 < len(new_beats) and (new_beats[i+1] - current_beat) < min_interval:
                                competitor = new_beats[i+1]
                                err_current = abs((current_beat - last_beat) - expected_interval)
                                err_competitor = abs((competitor - last_beat) - expected_interval)
                                if err_current <= err_competitor:
                                    accepted_this_frame.append(current_beat)
                                    rejected_this_frame.append(competitor)
                                else:
                                    rejected_this_frame.append(current_beat)
                                    accepted_this_frame.append(competitor)
                                i += 2
                            else:
                                if (current_beat - last_beat) > min_interval:
                                    accepted_this_frame.append(current_beat)
                                else:
                                    rejected_this_frame.append(current_beat)
                                i += 1
                    else:
                        accepted_this_frame.extend(new_beats)
                
                filtered_beat_times.extend(accepted_this_frame)
                
                calculated_bpm = 0.0
                if len(filtered_beat_times) > 5:
                    intervals = np.diff(filtered_beat_times)
                    median_interval = np.median(intervals)
                    if median_interval > 0:
                        calculated_bpm = 60.0 / median_interval
                
                if state == 'Locked' or calculated_bpm > 0:
                    active_bpm = calculated_bpm if calculated_bpm > 0 else librosa_bpm
                
                if now - last_event_time > config.silence_tmo:
                    active_bpm = 0.0
                
                with shared_state.lock:
                    pll_on = shared_state.pll_enabled
                
                if active_bpm == 0.0:
                    bot.set_target(head=neutral, antennas=np.zeros(2))
                    phase, err, rate = pll.phase, 0.0, 1.0
                else:
                    last_good_beat = filtered_beat_times[-1] if filtered_beat_times else 0
                    if pll_on:
                        phase, err, rate = pll.step(dt, active_bpm, last_good_beat, now, config)
                    else:
                        pll.phase = (pll.phase + dt * (active_bpm / 60.0)) % 1.0
                        phase, err, rate = pll.phase, 0.0, 1.0
                    
                    beat_phase = (phase + phase_offset) % 1.0
                    offs = move_fn(beat_phase, **params)
                    bot.set_target(
                        head=head_pose(config.neutral_pos + offs.get('position_offset', np.zeros(3)),
                                       config.neutral_eul + offs.get('orientation_offset', np.zeros(3))),
                        antennas=offs.get('antennas_offset', np.zeros(2))
                    )
                
                data_queue.put({
                    't': now, 'loop_dt': dt, 'state': state,
                    'librosa_bpm': librosa_bpm, 'calculated_bpm': active_bpm,
                    'phase': phase, 'accepted_beats': accepted_this_frame,
                    'rejected_beats': rejected_this_frame
                })
                
                time.sleep(max(0, config.control_ts - (time.time() - now)))
        
        except KeyboardInterrupt:
            print("\nCtrl-C received, shutting down...")
        finally:
            stop_event.set()
            ui_thread_obj.join(timeout=3)
            print("Shutdown complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reachy Rhythm Controller", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dance', default='head_bob_z', choices=AVAILABLE_DANCE_MOVES.keys())
    parser.add_argument('--live-plot', action='store_true', help='Show rolling 5s plot')
    parser.add_argument('--save-dir', default='.', help='Folder for PNG on Ctrl-C')
    cli_args = parser.parse_args()

    config = Config(
        dance_move=cli_args.dance,
        live_plot=cli_args.live_plot,
        save_dir=cli_args.save_dir
    )
    main(config)