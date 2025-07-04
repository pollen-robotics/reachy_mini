#!/usr/bin/env python3
# reachy_rhythm_controller.py  – v5.2 (Final)
"""
Real-time robot rhythm synchronization using a decoupled architecture.

This script synchronizes a Reachy Mini robot's movements to music detected
from a microphone. It features:
- A high-frequency (100 Hz), stable control loop.
- A separate thread for audio processing using Librosa to detect beat candidates.
- A live beat filtering algorithm to reject spurious detections.
- A robust BPM calculation based on the median of recent filtered beat intervals.
- A Phase-Locked Loop (PLL) to dynamically correct the robot's phase against
  the filtered beats, ensuring tight, long-term synchronization.
- A non-blocking UI thread for live plotting and statistics.
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
    audio_win: float = 2.0  # Duration (in seconds) of the audio buffer for beat analysis.
    audio_rate: int = 44100
    audio_chunk_size: int = 2048
    bpm_stability_buffer: int = 4  # Number of Librosa BPM estimates for stability check.
    bpm_stability_threshold: float = 3.0 # Std dev threshold to consider BPM 'Locked'.
    silence_tmo: float = 2.0 # Seconds of no audio event before stopping.

    # --- Beat Filtering & BPM Calculation ---
    beat_buffer_size: int = 20 # Number of recent accepted beats for BPM calculation.
    min_interval_factor: float = 0.5 # A beat is "too close" if interval is < this * expected.

    # --- Synchronization & PLL ---
    use_pll: bool = True # Master switch for the Phase-Locked Loop.
    pll_kp: float = 0.8  # Proportional gain for the PLL (faster correction).
    pll_max_rate: float = 0.15 # Max rate change (15%) to prevent jerky moves.
    pll_hold_threshold: float = 0.45 # If phase error > this, freeze correction.
    beat_time_offset: float = 0.0 # Manual offset (s) to add to beat timestamps for latency.

    # --- UI & Logging ---
    ui_update_rate: float = 2.0 # How many times per second to update UI.
    plot_window_duration: float = 5.0 # Duration (s) of the rolling plot.
    
    # --- Static ---
    neutral_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    neutral_eul: np.ndarray = field(default_factory=lambda: np.zeros(3))
    audio_buffer_len: int = field(init=False)

    def __post_init__(self):
        self.audio_buffer_len = int(self.audio_rate * self.audio_win)


# ───────────────────── Helper Classes and Functions ──────────────────────────
def head_pose(pos: np.ndarray, eul: np.ndarray) -> np.ndarray:
    m=np.eye(4); m[:3,3]=pos; m[:3,:3]=R.from_euler('xyz',eul).as_matrix(); return m

class MusicState:
    """Thread-safe state object for data from the audio thread."""
    def __init__(self) -> None:
        self.lock=threading.Lock(); self.librosa_bpm=0.0; self.last_event_time=0.0
        self.state='Init'; self.beats:collections.deque[float]=collections.deque(maxlen=512)

class PLL:
    """A Phase-Locked Loop to synchronize an internal oscillator to external beats."""
    def __init__(self) -> None:
        self.phase=0.0; self.rate=1.0

    def step(self, dt: float, bpm: float, last_beat_time: float, now: float, config: Config) -> tuple[float, float, float]:
        if bpm == 0.0: return self.phase, 0.0, 1.0
        if not config.use_pll or last_beat_time == 0.0:
            self.phase = (self.phase + dt * (bpm / 60.0)) % 1.0; return self.phase, 0.0, 1.0
        period = 60.0 / bpm
        expected_phase = ((now - last_beat_time) / period) % 1.0
        err = ((expected_phase - self.phase + 0.5) % 1.0) - 0.5
        self.rate = 1.0 + np.clip(config.pll_kp * err, -config.pll_max_rate, config.pll_max_rate)
        if abs(err) > config.pll_hold_threshold: return self.phase, err, 0.0
        self.phase = (self.phase + dt * (bpm / 60.0) * self.rate) % 1.0
        return self.phase, err, self.rate

# ───────────────────────────── Worker Threads ──────────────────────────────
def audio_thread(state: MusicState, config: Config, stop_event: threading.Event) -> None:
    """Dedicated thread to read audio and run Librosa's beat tracking."""
    pa = pyaudio.PyAudio(); stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=config.audio_rate, input=True, frames_per_buffer=config.audio_chunk_size)
    buf=np.empty(0,dtype=np.float32); bpm_hist=collections.deque(maxlen=config.bpm_stability_buffer)
    while not stop_event.is_set():
        try: buf=np.append(buf, np.frombuffer(stream.read(config.audio_chunk_size,exception_on_overflow=False),dtype=np.float32))
        except (IOError, ValueError): continue
        if len(buf) < config.audio_buffer_len: continue
        tempo,beat_frames=librosa.beat.beat_track(y=buf,sr=config.audio_rate,units='frames',tightness=100); now=time.time()
        tempo_val=tempo[0] if isinstance(tempo,np.ndarray) and tempo.size>0 else tempo
        with state.lock:
            state.last_event_time=now
            if tempo_val>40: bpm_hist.append(float(tempo_val)); state.librosa_bpm=float(np.mean(bpm_hist))
            win_dur=len(buf)/config.audio_rate; abs_times=[now-(win_dur-librosa.frames_to_time(f,sr=config.audio_rate)) for f in beat_frames]
            for t in abs_times:
                if not state.beats or t-state.beats[-1]>0.05: state.beats.append(t+config.beat_time_offset)
            if len(bpm_hist)<config.bpm_stability_buffer: state.state='Gathering'
            elif np.std(bpm_hist)<config.bpm_stability_threshold: state.state='Locked'
            else: state.state='Unstable'
        buf=buf[-int(config.audio_rate*1.5):]
    stream.stop_stream(); stream.close(); pa.terminate()

def ui_thread(data_queue: Queue, config: Config, stop_event: threading.Event):
    """Dedicated thread for all slow UI operations."""
    fig=None
    if config.live_plot:
        plt.ion(); fig,ax=plt.subplots(2,1,sharex=True,figsize=(10,6),constrained_layout=True)
        librosa_line,=ax[0].plot([],[],'o',ms=3,label='Librosa BPM',alpha=0.6); calc_line,=ax[0].plot([],[],'-',label='Calculated BPM')
        ax[0].set_ylabel('BPM'); ax[0].legend(); phase_line,=ax[1].plot([],[],'-',label='PLL Phase')
        ax[1].plot([],[],color='g',linestyle='-',label='Accepted Beat'); ax[1].plot([],[],color='r',linestyle='--',label='Rejected Beat')
        ax[1].set_ylabel('Phase/Beat'); ax[1].set_xlabel('Time (s)'); ax[1].legend()
    loop_dts,log_entries,last_ui_print_time,last_data=[],[],time.time(),None
    while not stop_event.is_set():
        try:
            while True:
                data=data_queue.get_nowait(); loop_dts.append(data['loop_dt']); log_entries.append(data); last_data=data
        except Empty: pass
        now=time.time()
        if not last_data or now-last_ui_print_time<(1.0/config.ui_update_rate): time.sleep(0.1); continue
        last_ui_print_time=now;
        if loop_dts:
            dts_ms=np.array(loop_dts)*1000
            stats={'mean':np.mean(dts_ms),'var':np.var(dts_ms),'max':np.max(dts_ms),'num':len(dts_ms)}
        else:
            stats={'mean':0,'var':0,'max':0,'num':0}
        if sys.stdout.isatty(): sys.stdout.write('\033[7F\033[J')
        print(f"--- Reachy Rhythm Controller ---\n"
              f"Control Loop (last {stats['num']} loops):\n  Mean: {stats['mean']:.1f}ms | Var: {stats['var']:.2f} | Max: {stats['max']:.1f}ms\n"
              f"State: {last_data['state']:<10} [PLL: {'ON' if config.use_pll else 'OFF'}]\n"
              f"Librosa BPM: {last_data['librosa_bpm']:6.1f}\nCalc. BPM:   {last_data['calculated_bpm']:6.1f}"); sys.stdout.flush(); loop_dts.clear()
        if config.live_plot and fig:
            t=np.array([e['t'] for e in log_entries]);
            if len(t)<2: continue
            start_time=t[0]; t-=start_time; mask=t>=t[-1]-config.plot_window_duration
            if not np.any(mask): continue
            librosa_bpm=np.array([e['librosa_bpm'] for e in log_entries]); calc_bpm=np.array([e['calculated_bpm'] for e in log_entries]); phase=np.sin(2*np.pi*np.array([e['phase'] for e in log_entries]))
            librosa_line.set_data(t[mask],librosa_bpm[mask]); calc_line.set_data(t[mask],calc_bpm[mask]); phase_line.set_data(t[mask],phase[mask])
            
            # FIXED: Correctly remove old vlines by iterating through the collections list.
            for collection in ax[1].collections[:]:
                collection.remove()

            acc_beats=np.array([b for e in log_entries for b in e['accepted_beats']])-start_time; rej_beats=np.array([b for e in log_entries for b in e['rejected_beats']])-start_time
            ax[1].vlines(acc_beats[acc_beats > t[mask][0]],-1,1,colors='g',linestyles='solid'); ax[1].vlines(rej_beats[rej_beats > t[mask][0]],-1,1,colors='r',linestyles='dashed')
            for a in ax: a.relim(); a.autoscale_view()
            y_min,y_max=ax[0].get_ylim(); ax[0].set_ylim(0,y_max*1.05); ax[1].set_ylim(-1.1,1.1); plt.pause(0.001)
    print("\nUI thread stopped.")
    if config.live_plot and fig:
        os.makedirs(config.save_dir,exist_ok=True); fname=f'reachy_live_{datetime.datetime.now():%Y%m%d_%H%M%S}.png'
        path=os.path.join(config.save_dir,fname); fig.savefig(path,dpi=150); print(f"Plot saved to {path}"); plt.ioff(); plt.show()

# ───────────────────────────── Main Control Loop ─────────────────────────────
def main(config: Config) -> None:
    data_queue, stop_event, music = Queue(), threading.Event(), MusicState()
    threading.Thread(target=audio_thread, args=(music, config, stop_event), daemon=True).start()
    ui=threading.Thread(target=ui_thread, args=(data_queue, config, stop_event), daemon=True); ui.start()
    pll=PLL(); neutral=head_pose(config.neutral_pos, config.neutral_eul); move_fn=AVAILABLE_DANCE_MOVES[config.dance_move]
    params=MOVE_SPECIFIC_PARAMS.get(config.dance_move,{}); phase_offset=1.0/(4*params.get('frequency_factor',1.0))
    last_loop=time.time(); processed_beats, active_bpm = 0, 0.0
    filtered_beat_times=collections.deque(maxlen=config.beat_buffer_size)
    print('Connecting to Reachy Mini…')
    with ReachyMini() as bot:
        bot.set_target(head=neutral,antennas=np.zeros(2)); time.sleep(1); print('Robot ready — play music!\n')
        try:
            while True:
                now,dt=time.time(),time.time()-last_loop; last_loop=now
                with music.lock:
                    librosa_bpm,state,last_event_time=music.librosa_bpm,music.state,music.last_event_time
                    new_beats=list(music.beats)[processed_beats:]
                processed_beats+=len(new_beats)
                accepted_this_frame,rejected_this_frame=[],[]
                if new_beats:
                    ref_bpm=active_bpm if active_bpm>0 else librosa_bpm
                    if ref_bpm>0:
                        expected_interval,min_interval=60.0/ref_bpm, (60.0/ref_bpm)*config.min_interval_factor; i=0
                        while i<len(new_beats):
                            last_beat=filtered_beat_times[-1] if filtered_beat_times else new_beats[i]-expected_interval
                            current_beat=new_beats[i]
                            if i+1<len(new_beats) and (new_beats[i+1]-current_beat)<min_interval:
                                competitor=new_beats[i+1]; err_current=abs((current_beat-last_beat)-expected_interval); err_competitor=abs((competitor-last_beat)-expected_interval)
                                if err_current<=err_competitor: accepted_this_frame.append(current_beat); rejected_this_frame.append(competitor)
                                else: rejected_this_frame.append(current_beat); accepted_this_frame.append(competitor)
                                i+=2
                            else:
                                if (current_beat-last_beat)>min_interval: accepted_this_frame.append(current_beat)
                                else: rejected_this_frame.append(current_beat)
                                i+=1
                    else: accepted_this_frame.extend(new_beats)
                filtered_beat_times.extend(accepted_this_frame)
                calculated_bpm=0.0
                if len(filtered_beat_times)>5:
                    intervals=np.diff(filtered_beat_times); median_interval=np.median(intervals)
                    if median_interval>0: calculated_bpm=60.0/median_interval
                if state=='Locked' or calculated_bpm>0: active_bpm=calculated_bpm if calculated_bpm>0 else librosa_bpm
                if now-last_event_time>config.silence_tmo: active_bpm=0.0
                if active_bpm==0.0: bot.set_target(head=neutral,antennas=np.zeros(2)); phase,err,rate=pll.phase,0.0,1.0
                else:
                    last_good_beat=filtered_beat_times[-1] if filtered_beat_times else 0
                    phase,err,rate=pll.step(dt,active_bpm,last_good_beat,now,config); beat_phase=(phase+phase_offset)%1.0
                    offs=move_fn(beat_phase,**params); bot.set_target(head=head_pose(config.neutral_pos+offs.get('position_offset',np.zeros(3)),config.neutral_eul+offs.get('orientation_offset',np.zeros(3))),antennas=offs.get('antennas_offset',np.zeros(2)))
                data_queue.put({'t':now,'loop_dt':dt,'state':state,'librosa_bpm':librosa_bpm,'calculated_bpm':active_bpm,'phase':phase,'accepted_beats':accepted_this_frame,'rejected_beats':rejected_this_frame})
                time.sleep(max(0,config.control_ts-(time.time()-now)))
        except KeyboardInterrupt: print("\nCtrl-C received, shutting down...")
        finally: stop_event.set(); ui.join(timeout=3); print("Shutdown complete.")

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