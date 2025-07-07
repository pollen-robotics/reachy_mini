#!/usr/bin/env python3
# reachy_rhythm_controller.py  – v10.3 (Final, Dual Phase)
"""
Real-time robot rhythm synchronization using a continuous, incremental clock.

This script synchronizes a Reachy Mini robot to music by advancing a
continuous "musical time" on each frame. This ensures the robot's phase
is always smooth, even when the detected BPM changes.

- A phase error is calculated to keep the robot's timing aligned with
  detected real-world beats by applying small, smooth, limited corrections.
- The final plot always shows two sine waves: the corrected phase and a
  perfect metronome reference for comparison.
- This "smart correction" can be toggled with the 'P' key.
- Manual phase offset can be adjusted with Left/Right arrow keys.
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
    dance_move: str = 'head_bob_z'
    save_dir: str = '.'
    control_ts: float = 0.01
    audio_win: float = 2.0
    audio_rate: int = 44100
    audio_chunk_size: int = 2048
    bpm_stability_buffer: int = 4
    bpm_stability_threshold: float = 3.0
    silence_tmo: float = 3.0
    beat_buffer_size: int = 20
    min_interval_factor: float = 0.5
    offset_correction_rate: float = 0.02
    max_phase_correction_per_frame: float = 0.005 # Limit correction to 0.5% of a beat per frame
    manual_offset_step: float = 0.01
    ui_update_rate: float = 2.0
    neutral_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    neutral_eul: np.ndarray = field(default_factory=lambda: np.zeros(3))
    audio_buffer_len: int = field(init=False)

    def __post_init__(self):
        self.audio_buffer_len = int(self.audio_rate * self.audio_win)


# ───────────────────── Helper Classes and Functions ──────────────────────────
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.smart_offset_enabled = True
        self.manual_offset = 0.0

    def toggle_smart_offset(self) -> bool:
        with self.lock: self.smart_offset_enabled = not self.smart_offset_enabled; return self.smart_offset_enabled

    def adjust_offset(self, amount: float) -> float:
        with self.lock: self.manual_offset += amount; return self.manual_offset

def head_pose(pos, eul):
    m = np.eye(4); m[:3, 3] = pos; m[:3, :3] = R.from_euler('xyz', eul).as_matrix(); return m

class MusicState:
    def __init__(self):
        self.lock = threading.Lock(); self.librosa_bpm = 0.0; self.last_event_time = 0.0
        self.state = 'Init'; self.beats: collections.deque[float] = collections.deque(maxlen=512)

def calculate_phase_error(musical_time: float, last_beat_time: float, bpm: float) -> float:
    if bpm == 0.0 or last_beat_time == 0.0: return 0.0
    current_phase = musical_time % 1.0
    period = 60.0 / bpm
    expected_phase = ((time.time() - last_beat_time) / period) % 1.0
    err = ((expected_phase - current_phase + 0.5) % 1.0) - 0.5
    return err

# ───────────────────────────── Worker Threads ──────────────────────────────
def keyboard_listener_thread(shared_state: SharedState, config: Config, stop_event: threading.Event):
    def on_press(key):
        if stop_event.is_set(): return False
        if hasattr(key, 'char') and key.char == 'p':
            new_state = shared_state.toggle_smart_offset()
            print(f"\n--- Smart Correction Toggled: {'ON' if new_state else 'OFF'} ---\n")
        if key == keyboard.Key.right:
            new_offset = shared_state.adjust_offset(config.manual_offset_step)
            print(f"\n--- Manual Offset -> {new_offset:+.3f} ---\n")
        elif key == keyboard.Key.left:
            new_offset = shared_state.adjust_offset(-config.manual_offset_step)
            print(f"\n--- Manual Offset -> {new_offset:+.3f} ---\n")
    with keyboard.Listener(on_press=on_press) as listener: listener.join()

def audio_thread(state: MusicState, config: Config, stop_event: threading.Event) -> None:
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
        tempo_val = tempo[0] if isinstance(tempo, np.ndarray) and tempo.size > 0 else tempo
        with state.lock:
            state.last_event_time = now
            if tempo_val > 40: bpm_hist.append(float(tempo_val)); state.librosa_bpm = float(np.mean(bpm_hist))
            win_dur = len(buf) / config.audio_rate
            abs_times = [now - (win_dur - librosa.frames_to_time(f, sr=config.audio_rate)) for f in beat_frames]
            for t in abs_times:
                if not state.beats or t - state.beats[-1] > 0.05: state.beats.append(t)
            if len(bpm_hist) < config.bpm_stability_buffer: state.state = 'Gathering'
            elif np.std(bpm_hist) < config.bpm_stability_threshold: state.state = 'Locked'
            else: state.state = 'Unstable'
        buf = buf[-int(config.audio_rate * 1.5):]
    stream.stop_stream(); stream.close(); pa.terminate()

def ui_thread(data_queue: Queue, shared_state: SharedState, stop_event: threading.Event):
    loop_dts, last_ui_print_time, last_data = [], time.time(), None
    while not stop_event.is_set():
        try:
            while True: data = data_queue.get_nowait(); loop_dts.append(data['loop_dt']); last_data = data
        except Empty: pass
        now = time.time()
        if not last_data or now - last_ui_print_time < (1.0 / config.ui_update_rate): time.sleep(0.1); continue
        last_ui_print_time = now
        if loop_dts: dts_ms = np.array(loop_dts)*1000; stats={'mean':np.mean(dts_ms),'var':np.var(dts_ms),'max':np.max(dts_ms),'num':len(dts_ms)}
        else: stats={'mean':0,'var':0,'max':0,'num':0}
        smart_offset_status = 'ON' if shared_state.smart_offset_enabled else 'OFF'
        manual_offset = shared_state.manual_offset
        print(f"\n--- Reachy Rhythm Controller ---\n"
              f"Loop (last {stats['num']}): Mean: {stats['mean']:.1f}ms | Var: {stats['var']:.2f} | Max: {stats['max']:.1f}ms\n"
              f"State: {last_data['state']:<10} [Smart Correction (P): {smart_offset_status}] [Manual Offset: {manual_offset:+.3f}]\n"
              f"Active BPM (Control): {last_data['active_bpm']:<6.1f}\n"
              f"Phase Error:          {last_data['phase_error']:+7.3f} [beat cycle]")
        sys.stdout.flush(); loop_dts.clear()
    print("\nUI thread stopped.")

# ───────────────────── Post-Run Plotting ──────────────────────────
def generate_final_plot(log, offset_log, config):
    if not log: print("No data logged, skipping plot generation."); return
    print("Generating final analysis plot...")
    plt.ioff(); fig,ax=plt.subplots(2,1,sharex=True,figsize=(15,8),constrained_layout=True); fig.suptitle("Post-Run Rhythm Analysis",fontsize=16)
    t = np.array([e['t'] for e in log]); start_time = t[0]; t -= start_time
    active_bpm = np.array([e['active_bpm'] for e in log])
    phase_error = np.array([e['phase_error'] for e in log])
    musical_time = np.array([e['musical_time'] for e in log])
    reference_time = np.array([e['reference_time'] for e in log])
    acc_beats = np.array([b for e in log for b in e['accepted_beats']]) - start_time
    
    ax[0].plot(t, active_bpm, '-', label='Active BPM (from Librosa)'); ax[0].set_ylabel('BPM'); ax[0].set_ylim(0, 200); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(t, np.sin(2 * np.pi * musical_time), '-', label='Corrected Phase (sin)')
    ax[1].plot(t, np.sin(2 * np.pi * reference_time), '--', label='Reference Phase (Metronome)', alpha=0.7)
    ax[1].vlines(acc_beats, -1, 1, colors='g', linestyles='solid', label='Accepted Beat', alpha=0.8)
    
    if offset_log:
        offset_times=[e[0]-start_time for e in offset_log]; offset_vals=[e[1] for e in offset_log]
        ax[1].vlines(offset_times, -1, 1, colors='m', linestyles=':', label='Manual Offset Adj.')
        for i, (ts, val) in enumerate(zip(offset_times, offset_vals)):
             ax[1].text(ts, -1.05, f"{val:+.2f}", color='m', fontsize=9, ha='center')

    ax[1].set_ylabel('Phase / Beat Cycle'); ax[1].set_xlabel('Time (s)'); ax[1].legend(); ax[1].grid(True,alpha=0.3)
    
    os.makedirs(config.save_dir, exist_ok=True)
    fname = f'reachy_analysis_{datetime.datetime.now():%Y%m%d_%H%M%S}.png'
    path = os.path.join(config.save_dir, fname); fig.savefig(path, dpi=150); print(f"Analysis plot saved to {path}"); plt.show()

# ───────────────────────────── Main Control Loop ─────────────────────────────
def main(config: Config) -> None:
    data_queue, stop_event, music, shared_state = Queue(), threading.Event(), MusicState(), SharedState()
    
    threading.Thread(target=audio_thread,args=(music,config,stop_event),daemon=True).start()
    threading.Thread(target=keyboard_listener_thread,args=(shared_state,config,stop_event),daemon=True).start()
    ui_thread_obj=threading.Thread(target=ui_thread,args=(data_queue,shared_state,stop_event),daemon=True); ui_thread_obj.start()
    
    move_fn=AVAILABLE_DANCE_MOVES[config.dance_move]; params=MOVE_SPECIFIC_PARAMS.get(config.dance_move,{})
    last_loop = time.time(); processed_beats, active_bpm = 0, 0.0
    filtered_beat_times = collections.deque(maxlen=config.beat_buffer_size)
    full_log, offset_change_log, last_manual_offset = [], [], 0.0
    musical_time, reference_time, phase_error = 0.0, 0.0, 0.0

    print('Connecting to Reachy Mini…')
    with ReachyMini() as bot:
        bot.set_target(head_pose(config.neutral_pos,config.neutral_eul),antennas=np.zeros(2)); time.sleep(1)
        print('Robot ready — play music! (P to toggle smart correction, Left/Right Arrows for manual offset)\n')
        
        try:
            t0 = time.time()
            while True:
                loop_start_time = time.time()
                dt = loop_start_time - last_loop
                last_loop = loop_start_time
                
                with music.lock:
                    librosa_bpm, state, last_event_time = music.librosa_bpm, music.state, music.last_event_time
                    new_beats = list(music.beats)[processed_beats:]
                processed_beats += len(new_beats)
                
                accepted_this_frame = []
                ref_bpm_filter = active_bpm if active_bpm > 0 else librosa_bpm
                if new_beats and ref_bpm_filter > 0:
                    expected_interval=60.0/ref_bpm_filter; min_interval=expected_interval*config.min_interval_factor; i=0
                    while i<len(new_beats):
                        last_beat=filtered_beat_times[-1] if filtered_beat_times else new_beats[i]-expected_interval
                        current_beat=new_beats[i]
                        if i+1<len(new_beats) and (new_beats[i+1]-current_beat)<min_interval:
                            competitor=new_beats[i+1]; err_current=abs((current_beat-last_beat)-expected_interval); err_competitor=abs((competitor-last_beat)-expected_interval)
                            if err_current<=err_competitor: accepted_this_frame.append(current_beat)
                            else: accepted_this_frame.append(competitor)
                            i+=2
                        else:
                            if(current_beat-last_beat)>min_interval: accepted_this_frame.append(current_beat)
                            i+=1
                elif new_beats: accepted_this_frame.extend(new_beats)
                filtered_beat_times.extend(accepted_this_frame)
                
                last_good_beat = filtered_beat_times[-1] if filtered_beat_times else 0

                active_bpm = librosa_bpm
                if time.time() - last_event_time > config.silence_tmo: active_bpm = 0.0
                
                smart_offset_on = shared_state.smart_offset_enabled
                manual_offset = shared_state.manual_offset
                if manual_offset != last_manual_offset:
                    offset_change_log.append((time.time(), manual_offset)); last_manual_offset = manual_offset
                
                time_for_dance = musical_time
                if active_bpm > 0:
                    # Always advance the reference metronome
                    reference_time += dt * (active_bpm / 60.0)

                    if smart_offset_on:
                        phase_error = calculate_phase_error(musical_time, last_good_beat, active_bpm)
                        correction = phase_error * config.offset_correction_rate
                        # Limit the correction to ensure smoothness
                        correction = np.clip(correction, -config.max_phase_correction_per_frame, config.max_phase_correction_per_frame)
                        musical_time += dt * (active_bpm / 60.0) + correction
                    else:
                        phase_error = 0.0
                        # When correction is off, the musical clock follows the perfect reference
                        musical_time = reference_time
                    
                    time_for_dance = musical_time + manual_offset
                    
                    offs = move_fn(time_for_dance, **params)
                    bot.set_target(head_pose(config.neutral_pos+offs.get('position_offset',np.zeros(3)),config.neutral_eul+offs.get('orientation_offset',np.zeros(3))),antennas=offs.get('antennas_offset',np.zeros(2)))
                else:
                    bot.set_target(head_pose(config.neutral_pos,config.neutral_eul),antennas=np.zeros(2))

                data_queue.put({'loop_dt':dt,'state':state,'active_bpm':active_bpm, 'phase_error':phase_error})
                full_log.append({'t':time.time(),'active_bpm':active_bpm,'phase_error':phase_error,'musical_time':musical_time,'reference_time':reference_time,'final_dance_time':time_for_dance,'accepted_beats':accepted_this_frame})
                
                time.sleep(max(0, config.control_ts - (time.time() - loop_start_time)))
        
        except KeyboardInterrupt: print("\nCtrl-C received, shutting down...")
        finally:
            stop_event.set()
            ui_thread_obj.join(timeout=2)
            print("Shutdown complete.")
            generate_final_plot(full_log, offset_change_log, config)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="Reachy Rhythm Controller",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dance',default='head_bob_z',choices=AVAILABLE_DANCE_MOVES.keys())
    parser.add_argument('--save-dir',default='.',help='Folder for PNG analysis plot')
    cli_args=parser.parse_args()
    config=Config(dance_move=cli_args.dance, save_dir=cli_args.save_dir)
    main(config)