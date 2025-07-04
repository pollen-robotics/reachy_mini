#!/usr/bin/env python3
# reachy_rhythm_controller.py  – v3.2 (GUI Exit Fix)
"""
Fixes the "main thread is not in main loop" crash by moving final
GUI operations (savefig, show) into the UI thread.
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

# ... (CLI and Constants are unchanged) ...
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dance', default='head_bob_z')
parser.add_argument('--live-plot', action='store_true', help='Show rolling 5 s plot')
parser.add_argument('--save-dir', default='.', help='Folder for PNG on Ctrl-C')
args = parser.parse_args()
CONTROL_TS = 0.01; AUDIO_WIN = 2.0; BUF_LEN = int(44100 * AUDIO_WIN); BPM_BUF = 4
BPM_STD_TH = 3.0; UNSTABLE_TMO = 5.0; SILENCE_TMO = 2.0; USE_PLL_CORRECTION = False
UI_UPDATE_RATE = 2; LIVE_WIN = 5.0; NEUTRAL_POS = np.zeros(3); NEUTRAL_EUL = np.zeros(3)

# ... (Helper math, State objects, and audio_thread are unchanged) ...
def mod_diff(a: float, b: float) -> float: return ((a - b + 0.5) % 1.0) - 0.5
def head_pose(pos: np.ndarray, eul: np.ndarray) -> np.ndarray:
    m=np.eye(4); m[:3,3]=pos; m[:3,:3]=R.from_euler('xyz',eul).as_matrix(); return m
class MusicState:
    def __init__(self) -> None:
        self.lock=threading.Lock(); self.goal_bpm=0.0; self.last_beat_time=0.0;
        self.raw_bpm=np.nan; self.state='Init'; self.beats:collections.deque[float]=collections.deque(maxlen=512)
class PLL:
    def __init__(self) -> None: self.phase = 0.0; self.rate  = 1.0
    def step(self, dt: float, bpm: float, last_beat: float, now: float) -> tuple[float, float, float]:
        if bpm == 0.0: return self.phase, 0.0, 1.0
        if not USE_PLL_CORRECTION:
            self.phase = (self.phase + dt * (bpm / 60.0)) % 1.0; return self.phase, 0.0, 1.0
        if last_beat == 0.0: return self.phase, 0.0, self.rate
        period=60.0/bpm; expected_phase=((now-last_beat)/period)%1.0; err=mod_diff(expected_phase, self.phase)
        self.rate=1.0+np.clip(PHASE_KP*err, -MAX_RATE, MAX_RATE)
        if abs(err)>HOLD: return self.phase, err, 0.0
        self.phase=(self.phase+dt*(bpm/60.0)*self.rate)%1.0; return self.phase, err, self.rate
def audio_thread(state: MusicState) -> None:
    pa=pyaudio.PyAudio(); stream=pa.open(format=pyaudio.paFloat32,channels=1,rate=44100,input=True,frames_per_buffer=2048)
    buf=np.empty(0,dtype=np.float32); bpm_hist=collections.deque(maxlen=BPM_BUF)
    while True:
        buf=np.append(buf, np.frombuffer(stream.read(2048,exception_on_overflow=False),dtype=np.float32))
        if len(buf)<BUF_LEN: continue
        tempo,beat_frames=librosa.beat.beat_track(y=buf,sr=44100,units='frames',tightness=100); now=time.time()
        tempo_val = tempo[0] if isinstance(tempo, np.ndarray) and tempo.size > 0 else tempo
        with state.lock: state.raw_bpm=np.nan if tempo_val<40 else float(tempo_val)
        if tempo_val>40: bpm_hist.append(float(tempo_val))
        sr=44100; win_dur=len(buf)/sr; abs_times=[now-(win_dur-librosa.frames_to_time(f,sr=sr)) for f in beat_frames]
        with state.lock:
            for t in abs_times:
                if not state.beats or t-state.beats[-1]>0.05: state.beats.append(t)
            if len(bpm_hist)<BPM_BUF: state.state=f'Gathering {len(bpm_hist)}/{BPM_BUF}'
            elif np.std(bpm_hist)<BPM_STD_TH:
                state.goal_bpm=float(np.mean(bpm_hist)); state.state='Locked'; state.last_beat_time=now
            else: state.state='Unstable'
        buf=buf[-int(44100*1.5):]
@dataclass
class LogEntry: t:float; raw:float; stable:float; err:float; rate:float; phase:float
@dataclass
class Logger:
    entries: list[LogEntry] = field(default_factory=list)
    def add(self, **kw) -> None: self.entries.append(LogEntry(**kw))
    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.array([getattr(e,k) for e in self.entries]) for k in LogEntry.__annotations__}

# ───────────────────── Analysis & UI Thread ───────────────────
@dataclass
class UIData:
    loop_dt: float
    state: str; raw_bpm: float; active_bpm: float
    log_entry: LogEntry; new_beats: list[float]

def analyze_beats_on_exit(raw_beats: list[float], final_bpm: float):
    # This function is unchanged.
    def _print_analysis(beat_list: list[float], name: str):
        if len(beat_list) < 2: print(f"\n--- Analysis for '{name}' (Not enough data) ---"); return
        print(f"\n--- Analysis for '{name}' ({len(beat_list)} beats) ---")
        intervals = np.diff(beat_list)
        mean_interval = np.mean(intervals)
        mean_bpm = 60.0 / mean_interval if mean_interval > 0 else 0
        print(f"Intervals (s): {intervals}"); print(f"Mean Interval: {mean_interval:.4f} s")
        print(f"Mean BPM from Intervals: {mean_bpm:.2f} BPM")
    _print_analysis(raw_beats, "Raw Beats")
    if final_bpm == 0 or len(raw_beats) < 2: print("\nSkipping filtered analysis."); return
    expected_interval = 60.0 / final_bpm; min_interval_threshold = expected_interval * 0.5
    filtered_beats = [raw_beats[0]]; i = 1
    while i < len(raw_beats):
        current_beat = raw_beats[i]; last_accepted_beat = filtered_beats[-1]
        if i + 1 < len(raw_beats) and (raw_beats[i+1] - current_beat) < min_interval_threshold:
            competitor_beat = raw_beats[i+1]
            error_current = abs((current_beat - last_accepted_beat) - expected_interval)
            error_competitor = abs((competitor_beat - last_accepted_beat) - expected_interval)
            if error_current <= error_competitor: filtered_beats.append(current_beat); i += 2
            else: i += 1
        else:
            if (current_beat - last_accepted_beat) > min_interval_threshold: filtered_beats.append(current_beat)
            i += 1
    _print_analysis(filtered_beats, "Filtered Beats")

def ui_thread(data_queue: Queue, stop_event: threading.Event, live_win: float):
    # ... (function body is mostly the same until the end) ...
    logger = Logger(); accepted_beats: list[float] = []; loop_dts: list[float] = []
    fig = None
    if args.live_plot:
        plt.ion(); fig,ax=plt.subplots(2,1,sharex=True,figsize=(10,6)); raw_line,=ax[0].plot([],[],'o',ms=3,label='Raw',alpha=0.6)
        act_line,=ax[0].plot([],[],'-',label='Active'); ax[0].set_ylabel('BPM'); ax[0].legend(); sin_line,=ax[1].plot([],[],'-',label='sin phase')
        acc_up,=ax[1].plot([],[],'^',color='green',ms=5,label='beat detected'); acc_dn,=ax[1].plot([],[],'v',color='green',ms=5)
        ax[1].set_ylabel('amp'); ax[1].set_xlabel('s'); ax[1].legend(); fig.tight_layout(); lines=(raw_line,act_line,sin_line,acc_up,acc_dn)
    last_ui_print_time = time.time(); last_data = None
    while not stop_event.is_set():
        try:
            while True:
                data_point = data_queue.get_nowait(); loop_dts.append(data_point.loop_dt)
                logger.add(**data_point.log_entry.__dict__); accepted_beats.extend(data_point.new_beats)
                last_data = data_point
        except Empty: pass
        now = time.time()
        if not last_data or now - last_ui_print_time < (1.0 / UI_UPDATE_RATE): time.sleep(0.1); continue
        last_ui_print_time = now
        if sys.stdout.isatty(): sys.stdout.write('\033[6F\033[J')
        dts_ms=np.array(loop_dts)*1000; mean_dt=np.mean(dts_ms); var_dt=np.var(dts_ms); max_dt=np.max(dts_ms)
        print(f'--- Reachy Rhythm UI ---\nControl Loop (stats for last {len(dts_ms)} loops):\n'
              f'  Mean: {mean_dt:.1f}ms | Var: {var_dt:.2f} | Max: {max_dt:.1f}ms ({1000/max_dt if max_dt>0 else 0:.0f} Hz min)\n'
              f'State: {last_data.state} [PLL Correction: {"ON" if USE_PLL_CORRECTION else "OFF"}]\n'
              f'Raw BPM: {"--" if np.isnan(last_data.raw_bpm) else f"{last_data.raw_bpm:6.1f}"}\n'
              f'Stable BPM: {last_data.active_bpm:6.1f}'); sys.stdout.flush(); loop_dts.clear()
        if args.live_plot:
            arr=logger.arrays(); tvals=arr['t']-arr['t'][0]
            mask=tvals>=tvals[-1]-live_win if len(tvals)>0 else np.array([],dtype=bool)
            if np.any(mask):
                amp_vals=np.sin(2*np.pi*arr['phase']); acc_x=np.array([bt-arr['t'][0] for bt in accepted_beats if bt>=arr['t'][0]+tvals[mask][0]])
                rl,al,sl,au,ad=lines; rl.set_data(tvals[mask],arr['raw'][mask]); al.set_data(tvals[mask],arr['stable'][mask]); sl.set_data(tvals[mask],amp_vals[mask])
                au.set_data(acc_x,np.ones_like(acc_x)); ad.set_data(acc_x,-np.ones_like(acc_x));
                for a in ax: a.relim(); a.autoscale_view()
                y_min,y_max=ax[0].get_ylim(); ax[0].set_ylim(0,y_max*1.05); plt.pause(0.001)
    
    # --- FIXED: GUI operations must be done in the same thread that created the plot ---
    print("\nUI thread stopped. Performing final analysis...")
    if last_data:
        analyze_beats_on_exit(accepted_beats, last_data.active_bpm)
    if args.live_plot and fig:
        save_live_png(fig) # Pass the figure object
        plt.ioff() # Turn off interactive mode
        plt.show() # Now this will block until the window is closed
    print("Analysis complete.")

def main() -> None:
    data_queue=Queue(); stop_event=threading.Event(); music=MusicState()
    threading.Thread(target=audio_thread, args=(music,), daemon=True).start()
    ui=threading.Thread(target=ui_thread,args=(data_queue, stop_event, LIVE_WIN), daemon=True)
    ui.start()
    pll=PLL(); neutral=head_pose(NEUTRAL_POS,NEUTRAL_EUL); move_fn=AVAILABLE_DANCE_MOVES[args.dance]
    params=MOVE_SPECIFIC_PARAMS.get(args.dance,{}); phase_offset=1.0/(4*params.get('frequency_factor',1.0))
    last_loop=time.time(); active_bpm=0.0; processed_beats=0

    print('Connecting to Reachy Mini…')
    with ReachyMini() as bot:
        bot.set_target(head=neutral, antennas=np.zeros(2)); time.sleep(1)
        print('Robot ready — play music!\n')
        try:
            while True:
                now=time.time(); dt=now-last_loop; last_loop=now
                with music.lock:
                    goal_bpm,raw_bpm,state,last_beat=music.goal_bpm,music.raw_bpm,music.state,music.last_beat_time
                    new_beats=list(music.beats)[processed_beats:]
                processed_beats+=len(new_beats)
                if state=='Locked':
                    if active_bpm!=goal_bpm:
                        print(f"\n--- [CONTROL] BPM updated to: {goal_bpm:.1f} ---\n"); active_bpm=goal_bpm; pll.phase=0.0
                if now-last_beat>SILENCE_TMO: active_bpm=0.0
                if active_bpm==0.0: bot.set_target(head=neutral,antennas=np.zeros(2)); phase,err,rate=pll.phase,0.0,1.0
                else:
                    phase,err,rate=pll.step(dt,active_bpm,last_beat,now); beat_phase=(phase+phase_offset)%1.0
                    offs=move_fn(beat_phase,**params)
                    bot.set_target(head=head_pose(NEUTRAL_POS+offs.get('position_offset',np.zeros(3)),NEUTRAL_EUL+offs.get('orientation_offset',np.zeros(3))),antennas=offs.get('antennas_offset',np.zeros(2)))
                log_entry=LogEntry(t=now,raw=raw_bpm,stable=active_bpm,err=err,rate=rate*100.0,phase=phase)
                ui_data=UIData(loop_dt=dt,state=state,raw_bpm=raw_bpm,active_bpm=active_bpm,log_entry=log_entry,new_beats=new_beats)
                data_queue.put(ui_data)
                time.sleep(max(0,CONTROL_TS-(time.time()-now)))
        except KeyboardInterrupt:
            print("\nCtrl-C received, shutting down...")
        finally:
            # FIXED: The main thread's only job is to signal and wait.
            stop_event.set()
            ui.join(timeout=3) # Wait for UI thread to finish its work
            print("Shutdown complete.")

def save_live_png(fig: plt.Figure) -> None: # MODIFIED to accept figure
    if not args.live_plot: return
    os.makedirs(args.save_dir, exist_ok=True); fname=f'reachy_live_{datetime.datetime.now():%Y%m%d_%H%M%S}.png'
    path=os.path.join(args.save_dir, fname);
    fig.savefig(path, dpi=150) # Use the passed figure object
    print(f'PNG saved to {path}')

if __name__ == '__main__':
    main()