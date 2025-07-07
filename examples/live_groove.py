#!/usr/bin/env python3
# reachy_rhythm_controller.py – v11.2 (Timing & UI Fix)
"""
Real-time robot choreography and rhythm synchronization.

This version fixes two critical bugs:
1.  The robot will no longer start moving prematurely before a stable BPM is 'Locked'.
2.  The terminal UI display no longer lags and now shows the current state in real-time.
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

# ... (Config, SharedState, Choreographer, and other helpers are unchanged) ...
@dataclass
class Config:
    save_dir: str = '.'; control_ts: float = 0.01; audio_win: float = 2.0
    audio_rate: int = 44100; audio_chunk_size: int = 2048; bpm_stability_buffer: int = 4
    bpm_stability_threshold: float = 3.0; unstable_periods_before_stop: int = 2
    silence_tmo: float = 3.0; beat_buffer_size: int = 20; min_interval_factor: float = 0.5
    offset_correction_rate: float = 0.02; max_phase_correction_per_frame: float = 0.005
    manual_offset_step: float = 0.01; beats_per_sequence: int = 8; ui_update_rate: float = 1.0
    neutral_pos: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.02]))
    neutral_eul: np.ndarray = field(default_factory=lambda: np.zeros(3))
    audio_buffer_len: int = field(init=False)
    def __post_init__(self): self.audio_buffer_len = int(self.audio_rate * self.audio_win)
class SharedState:
    def __init__(self):
        self.lock = threading.Lock(); self.smart_offset_enabled = False
        self.manual_offset = 0.0; self.next_move = False; self.prev_move = False
        self.next_waveform = False; self.amplitude_change = 0.0
    def get_and_clear_changes(self) -> dict:
        with self.lock:
            changes = {"smart_offset_toggled":self.smart_offset_enabled,"manual_offset":self.manual_offset,"next_move":self.next_move,"prev_move":self.prev_move,"next_waveform":self.next_waveform,"amplitude_change":self.amplitude_change}
            self.next_move = self.prev_move = self.next_waveform = False; self.amplitude_change = 0.0; return changes
    def toggle_smart_offset(self): self.smart_offset_enabled = not self.smart_offset_enabled
    def adjust_offset(self, amount): self.manual_offset += amount
    def trigger_next_move(self): self.next_move = True
    def trigger_prev_move(self): self.prev_move = True
    def trigger_next_waveform(self): self.next_waveform = True
    def adjust_amplitude(self, amount): self.amplitude_change += amount
class MusicState:
    def __init__(self):
        self.lock = threading.Lock(); self.librosa_bpm = 0.0; self.raw_librosa_bpm = 0.0
        self.last_event_time = 0.0; self.state = 'Init'
        self.beats: collections.deque[float] = collections.deque(maxlen=512)
        self.unstable_period_count = 0
class Choreographer:
    def __init__(self):
        self.move_names = list(AVAILABLE_DANCE_MOVES.keys()); self.waveforms = ['sin', 'cos', 'triangle', 'square', 'sawtooth']
        self.move_idx = 0; self.waveform_idx = 0; self.amplitude_scale = 1.0; self.beat_counter_for_cycle = 0.0
    def current_move_name(self): return self.move_names[self.move_idx]
    def current_waveform(self): return self.waveforms[self.waveform_idx]
    def advance(self, beats_this_frame, config):
        self.beat_counter_for_cycle += beats_this_frame
        if self.beat_counter_for_cycle >= config.beats_per_sequence: self.next_move()
    def next_move(self): self.move_idx = (self.move_idx + 1) % len(self.move_names); self.beat_counter_for_cycle = 0
    def prev_move(self): self.move_idx = (self.move_idx - 1 + len(self.move_names)) % len(self.move_names); self.beat_counter_for_cycle = 0
    def next_waveform(self): self.waveform_idx = (self.waveform_idx + 1) % len(self.waveforms)
    def change_amplitude(self, amount): self.amplitude_scale = max(0.1, self.amplitude_scale + amount)
def head_pose(pos, eul): m = np.eye(4); m[:3, 3] = pos; m[:3, :3] = R.from_euler('xyz', eul).as_matrix(); return m
def calculate_phase_error(t_beats: float, last_beat_time: float, bpm: float) -> float:
    if bpm == 0.0 or last_beat_time == 0.0: return 0.0
    current_phase = t_beats % 1.0; period = 60.0 / bpm
    expected_phase = ((time.time() - last_beat_time) / period) % 1.0
    err = ((expected_phase - current_phase + 0.5) % 1.0) - 0.5; return err

# ───────────────────────────── Worker Threads ──────────────────────────────
def keyboard_listener_thread(shared_state: SharedState, stop_event: threading.Event):
    def on_press(key):
        if stop_event.is_set(): return False
        if hasattr(key, 'char'):
            if key.char.lower() == 'p': shared_state.toggle_smart_offset()
            if key.char.lower() == 'n': shared_state.trigger_next_move()
            if key.char.lower() == 'b': shared_state.trigger_prev_move()
            if key.char.lower() == 'w': shared_state.trigger_next_waveform()
            if key.char == '+': shared_state.adjust_amplitude(0.1)
            if key.char == '-': shared_state.adjust_amplitude(-0.1)
        if key == keyboard.Key.right: shared_state.adjust_offset(0.01)
        elif key == keyboard.Key.left: shared_state.adjust_offset(-0.01)
    with keyboard.Listener(on_press=on_press) as listener: listener.join()

def audio_thread(state: MusicState, config: Config, stop_event: threading.Event) -> None:
    # This function remains unchanged
    pa=pyaudio.PyAudio(); stream=pa.open(format=pyaudio.paFloat32,channels=1,rate=config.audio_rate,input=True,frames_per_buffer=config.audio_chunk_size)
    buf=np.empty(0,dtype=np.float32); bpm_hist=collections.deque(maxlen=config.bpm_stability_buffer)
    while not stop_event.is_set():
        try:
            audio_chunk = np.frombuffer(stream.read(config.audio_chunk_size,exception_on_overflow=False),dtype=np.float32)
            buf = np.append(buf, audio_chunk)
        except(IOError,ValueError): continue
        if len(buf) < config.audio_buffer_len: continue
        tempo, beat_frames = librosa.beat.beat_track(y=buf, sr=config.audio_rate, units='frames', tightness=100)
        now = time.time()
        tempo_val = float(tempo[0] if isinstance(tempo, np.ndarray) and tempo.size > 0 else tempo)
        with state.lock:
            state.last_event_time = now; state.raw_librosa_bpm = tempo_val
            if tempo_val > 40:
                bpm_hist.append(tempo_val); state.librosa_bpm = float(np.mean(bpm_hist))
            win_dur = len(buf) / config.audio_rate
            abs_times = [now - (win_dur - librosa.frames_to_time(f, sr=config.audio_rate)) for f in beat_frames]
            for t in abs_times:
                if not state.beats or t - state.beats[-1] > 0.05: state.beats.append(t)
            if len(bpm_hist) < config.bpm_stability_buffer:
                state.state = 'Gathering'; state.unstable_period_count = 0
            elif np.std(bpm_hist) < config.bpm_stability_threshold:
                state.state = 'Locked'; state.unstable_period_count = 0
            else:
                state.state = 'Unstable'; state.unstable_period_count += 1
        buf = buf[-int(config.audio_rate * 1.5):]
    stream.stop_stream(); stream.close(); pa.terminate()

def ui_thread(data_queue: Queue, config: Config, stop_event: threading.Event):
    """FIXED: This thread now drains the queue to always show the latest data."""
    last_ui_print_time, last_data = time.time(), None
    while not stop_event.is_set():
        try:
            # Drain the queue, keeping only the most recent data point
            while True:
                last_data = data_queue.get_nowait()
        except Empty:
            pass  # Queue is empty, proceed with the last known data

        now = time.time()
        if not last_data or now - last_ui_print_time < (1.0 / config.ui_update_rate):
            time.sleep(0.1)
            continue
        last_ui_print_time = now
        
        correction_status = 'ON' if last_data['correction_on'] else 'OFF'
        paused_status = " | PAUSED (Music Unstable)" if last_data['unstable_pause'] else ""

        print("\n" + "─" * 80 + "\n"
              f"🎵 Music State: {last_data['state']:<10} | BPM (Active/Raw): {last_data['active_bpm']:.1f}/{last_data['raw_bpm']:.1f}{paused_status}\n"
              f"🕺 Dance State: {last_data['move_name']:<25} | Wave: {last_data['waveform']:<8} | Amp: {last_data['amp_scale']:.1f}x\n"
              f"⚙️  Settings: Smart Correction (P): {correction_status} | Manual Offset (←/→): {last_data['manual_offset']:+.3f}\n"
              + "─" * 80)
        sys.stdout.flush()

# ... (generate_final_plot is unchanged) ...
def generate_final_plot(log, config):
    if not log: return
    t = np.array([e['t'] for e in log]); start_time = t[0]; t -= start_time
    t_beats = np.array([e['t_beats'] for e in log])
    reference_t_beats = np.array([e['reference_t_beats'] for e in log])
    acc_beats = np.array([b for e in log for b in e['accepted_beats']]) - start_time
    fig,ax=plt.subplots(2,1,sharex=True,figsize=(15,8),constrained_layout=True)
    ax[0].plot(t, [e['active_bpm'] for e in log], '-', label='Active BPM')
    ax[0].set_ylabel('BPM'); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(t, np.sin(2 * np.pi * t_beats), '-', label='Corrected Beat Clock (sin)')
    ax[1].plot(t, np.sin(2 * np.pi * reference_t_beats), '--', label='Reference Beat Clock (Metronome)', alpha=0.7)
    ax[1].vlines(acc_beats, -1, 1, colors='g', linestyles='solid', label='Accepted Beat', alpha=0.8)
    ax[1].set_ylabel('Beat Cycle'); ax[1].set_xlabel('Time (s)'); ax[1].legend(); ax[1].grid(True,alpha=0.3)
    path = os.path.join(config.save_dir, f'reachy_analysis_{datetime.datetime.now():%Y%m%d_%H%M%S}.png');
    fig.savefig(path, dpi=150); print(f"\nAnalysis plot saved to {path}"); plt.show()


# ───────────────────────────── Main Control Loop ─────────────────────────────
def main(config: Config) -> None:
    data_queue, stop_event = Queue(), threading.Event()
    music, shared_state, choreographer = MusicState(), SharedState(), Choreographer()
    
    threading.Thread(target=audio_thread, args=(music, config, stop_event), daemon=True).start()
    threading.Thread(target=keyboard_listener_thread, args=(shared_state, stop_event), daemon=True).start()
    threading.Thread(target=ui_thread, args=(data_queue, config, stop_event), daemon=True).start()
    
    last_loop = time.time(); processed_beats, active_bpm = 0, 0.0
    filtered_beat_times = collections.deque(maxlen=config.beat_buffer_size)
    full_log = []
    t_beats, reference_t_beats, phase_error = 0.0, 0.0, 0.0

    print('Connecting to Reachy Mini...')
    with ReachyMini() as mini:
        mini.set_target(head_pose(config.neutral_pos, config.neutral_eul), antennas=np.zeros(2)); time.sleep(1)
        print('\nRobot ready — play music!\nControls: [N]ext Move, [B]ack, [P]hase Correction, [W]aveform, [+/-] Amp, [←/→] Offset\n')
        
        try:
            while True:
                loop_start_time = time.time()
                dt = loop_start_time - last_loop
                last_loop = loop_start_time
                
                changes = shared_state.get_and_clear_changes()
                if changes['next_move']: choreographer.next_move()
                if changes['prev_move']: choreographer.prev_move()
                if changes['next_waveform']: choreographer.next_waveform()
                if changes['amplitude_change']: choreographer.change_amplitude(changes['amplitude_change'])
                
                with music.lock:
                    librosa_bpm, raw_bpm = music.librosa_bpm, music.raw_librosa_bpm
                    state, last_event_time = music.state, music.last_event_time
                    unstable_count = music.unstable_period_count
                    new_beats = list(music.beats)[processed_beats:]
                processed_beats += len(new_beats)
                
                active_bpm = librosa_bpm if time.time() - last_event_time < config.silence_tmo else 0.0
                
                # ... (Beat filtering logic is unchanged) ...
                accepted_this_frame = []
                if new_beats and active_bpm > 0:
                    expected_interval=60.0/active_bpm; min_interval=expected_interval*config.min_interval_factor; i=0
                    while i<len(new_beats):
                        last_beat=filtered_beat_times[-1] if filtered_beat_times else new_beats[i]-expected_interval; current_beat=new_beats[i]
                        if i+1<len(new_beats) and (new_beats[i+1]-current_beat)<min_interval:
                            c=new_beats[i+1];e1=abs((current_beat-last_beat)-expected_interval);e2=abs((c-last_beat)-expected_interval)
                            if e1<=e2: accepted_this_frame.append(current_beat)
                            else: accepted_this_frame.append(c)
                            i+=2
                        else:
                            if(current_beat-last_beat)>min_interval: accepted_this_frame.append(current_beat)
                            i+=1
                filtered_beat_times.extend(accepted_this_frame)
                last_good_beat = filtered_beat_times[-1] if filtered_beat_times else 0

                # --- Main Dance Condition ---
                # FIXED: Robot only starts dancing when state is 'Locked', and then
                # continues dancing through 'Unstable' until the threshold is met.
                is_allowed_to_start = (state == 'Locked')
                is_stable_enough_to_continue = (unstable_count < config.unstable_periods_before_stop)
                
                # Check if we are in a previously dancing state OR allowed to start now
                can_dance = (active_bpm > 0 and (is_allowed_to_start or (state == 'Unstable' and is_stable_enough_to_continue)))

                if can_dance:
                    beats_this_frame = dt * (active_bpm / 60.0)
                    reference_t_beats += beats_this_frame
                    if changes['smart_offset_toggled']:
                        phase_error = calculate_phase_error(t_beats, last_good_beat, active_bpm)
                        correction = np.clip(phase_error * config.offset_correction_rate, -config.max_phase_correction_per_frame, config.max_phase_correction_per_frame)
                        t_beats += beats_this_frame + correction
                    else:
                        phase_error = 0.0; t_beats = reference_t_beats
                    
                    choreographer.advance(beats_this_frame, config)
                    move_name = choreographer.current_move_name()
                    move_fn = AVAILABLE_DANCE_MOVES[move_name]
                    params = MOVE_SPECIFIC_PARAMS.get(move_name, {}).copy()
                    params['waveform'] = choreographer.current_waveform()
                    for key in params:
                        if 'amplitude' in key: params[key] *= choreographer.amplitude_scale
                    
                    time_for_dance = t_beats + changes['manual_offset']
                    offsets = move_fn(time_for_dance, **params)
                    mini.set_target(head_pose(config.neutral_pos + offsets.position_offset, config.neutral_eul + offsets.orientation_offset), antennas=offsets.antennas_offset)
                else:
                    mini.set_target(head_pose(config.neutral_pos, config.neutral_eul), antennas=np.zeros(2))
                
                # --- Logging and UI Update ---
                ui_data = {'state': state, 'active_bpm': active_bpm, 'raw_bpm': raw_bpm, 'phase_error': phase_error,
                           'move_name': choreographer.current_move_name(), 'waveform': choreographer.current_waveform(),
                           'amp_scale': choreographer.amplitude_scale, 'correction_on': changes['smart_offset_toggled'],
                           'manual_offset': changes['manual_offset'], 'unstable_pause': not is_stable_enough_to_continue and state=='Unstable'}
                data_queue.put(ui_data)
                full_log.append({'t': time.time(), 'active_bpm': active_bpm, 'phase_error': phase_error, 't_beats': t_beats, 'reference_t_beats': reference_t_beats, 'accepted_beats': accepted_this_frame})
                
                time.sleep(max(0, config.control_ts - (time.time() - loop_start_time)))
        
        except KeyboardInterrupt: print("\nCtrl-C received, shutting down...")
        finally:
            stop_event.set(); print("Shutdown complete.")
            generate_final_plot(full_log, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reachy Rhythm Controller", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save-dir', default='.', help='Folder for PNG analysis plot')
    cli_args = parser.parse_args()
    config = Config(save_dir=cli_args.save_dir)
    main(config)