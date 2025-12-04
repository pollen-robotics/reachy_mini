#!/usr/bin/env python3
# coding: utf-8
"""
reachy_rhythm_controller.py - v12.0 (Simplified, no keyboard, no smart correction)

Real-time robot choreography driven by live BPM from the microphone.
- No manual controls (no keyboard listener).
- No "smart correction" phase alignment. The beat clock follows detected BPM directly.
- Auto-advances dance moves every N beats.
- Final analysis plot unchanged (BPM over time + corrected/reference beat clocks with accepted beats).

Dependencies: numpy, librosa, pyaudio, matplotlib, reachy_mini
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
from queue import Empty, Queue

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyaudio

from reachy_mini import ReachyMini, utils
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES


# ───────────────────────────── Move Amplitude Overrides ──────────────────────
# Per-move amplitude scaling (1.0 = full, 0.8 = 20% reduction, etc.)
# Moves not listed here use 1.0 (full amplitude)
MOVE_AMPLITUDE_OVERRIDES = {
    "headbanger_combo": 0.3,
    "dizzy_spin": 0.8,
    "pendulum_swing": 0.4,
    "jackson_square": 0.6,
    "side_to_side_sway": 0.7,
    "sharp_side_tilt": 0.35,
    "grid_snap": 0.5,
    "side_peakaboo": 0.4
}


# ───────────────────────────────── Config ────────────────────────────────────
@dataclass
class Config:
    # Where to save the final analysis PNG
    save_dir: str = "."

    # Main control loop period (s). Lower for tighter control, higher for lower CPU.
    control_ts: float = 0.01

    # Audio analysis window (s). Longer gives more stable BPM, but slower reactions.
    audio_win: float = 4.0

    # Microphone sample rate (Hz). 44100 is common and well supported.
    audio_rate: int = 44100

    # Mic buffer size per read. Smaller reduces latency; too small increases CPU/underruns.
    audio_chunk_size: int = 2048

    # Noise calibration duration (s). Records noise while robot does breathing.
    # 14s covers 2 full breathing cycles (slowest is roll at 6.67s/cycle).
    noise_calibration_duration: float = 14.0

    # Dance noise calibration duration (s). Records noise while robot does dance moves.
    dance_noise_calibration_duration: float = 8.0

    # How aggressively to subtract noise (1.0 = full subtraction, 0.5 = half).
    noise_subtraction_strength: float = 1.0

    # How many most recent BPM estimates to average for stability.
    bpm_stability_buffer: int = 4

    # BPM range clamping - forces BPM into this range by halving/doubling.
    # This fixes half-time/double-time detection issues.
    bpm_min: float = 70.0
    bpm_max: float = 140.0

    # Max allowed standard deviation over the stability buffer to consider "Locked".
    # Lower threshold = stricter lock; higher = looser (locks faster).
    bpm_stability_threshold: float = 15.0

    # If BPM becomes Unstable, how many consecutive unstable periods we tolerate before pausing motion.
    unstable_periods_before_stop: int = 8

    # If we haven't seen audio events for this many seconds, consider silence and stop motion.
    silence_tmo: float = 3.0

    # Buffer of recent accepted beat times used by the graph and control.
    beat_buffer_size: int = 20

    # Beat deduplication: beats closer than this fraction of the expected interval are considered duplicates.
    # Increase to remove double triggers; decrease if valid syncopations are dropped.
    min_interval_factor: float = 0.5

    # Auto-advance the dance after this many beats.
    beats_per_sequence: int = 8

    # How often the terminal UI refreshes (Hz).
    ui_update_rate: float = 1.0

    # Neutral pose (position in meters and Euler orientation in radians).
    # Adjust to fit your neutral posture in your setup.
    # Z offset of 0.01m added to prevent head-body collision.
    neutral_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.01]))
    neutral_eul: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # The following two are kept for completeness but are not used (smart correction removed).
    offset_correction_rate: float = 0.02
    max_phase_correction_per_frame: float = 0.005

    # Derived: number of samples in the rolling audio window.
    audio_buffer_len: int = field(init=False)

    def __post_init__(self):
        self.audio_buffer_len = int(self.audio_rate * self.audio_win)


# ───────────────────────────── Runtime State ─────────────────────────────────
class MusicState:
    def __init__(self):
        self.lock = threading.Lock()
        self.librosa_bpm = 0.0
        self.raw_librosa_bpm = 0.0
        self.last_event_time = 0.0
        self.state = "Init"
        self.beats: collections.deque[float] = collections.deque(maxlen=512)
        self.unstable_period_count = 0
        self.has_ever_locked = False  # Require stable lock before dancing
        self.bpm_std = 0.0  # Debug: track BPM standard deviation
        self.cleaned_amplitude = 0.0  # Debug: track audio amplitude after noise subtraction
        self.raw_amplitude = 0.0  # Debug: track raw audio amplitude (used for presence detection)
        self.is_breathing = True  # True when robot is in breathing/idle mode (set by main loop)


class Choreographer:
    def __init__(self):
        # Registry of available moves (name -> (fn, base_params, meta))
        self.move_names = list(AVAILABLE_MOVES.keys())
        # Default single waveform; moves that support it will consume it.
        self.waveforms = ["sin"]
        self.move_idx = 0
        self.waveform_idx = 0
        self.amplitude_scale = 1.0
        self.beat_counter_for_cycle = 0.0

    def current_move_name(self):
        return self.move_names[self.move_idx]

    def current_waveform(self):
        return self.waveforms[self.waveform_idx]

    def advance(self, beats_this_frame, config: Config):
        self.beat_counter_for_cycle += beats_this_frame
        if self.beat_counter_for_cycle >= config.beats_per_sequence:
            self.move_idx = (self.move_idx + 1) % len(self.move_names)
            self.beat_counter_for_cycle = 0.0


# ───────────────────────────── Idle Breathing Motion ─────────────────────────
def compute_breathing_pose(t: float, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute breathing/idle pose when music state is unstable.

    Based on pi-500-reachy-mini-client-1 BreathingMove with modifications:
    - Y sway: 8mm amplitude, 0.2 Hz (5 second cycle)
    - Head roll: 15% intensity sin wave at 0.15 Hz
    - Antenna sway: 24° amplitude (2x original 12°), 0.5 Hz

    Returns:
        (position, orientation, antennas) - all as numpy arrays
    """
    # Y sway - gentle side-to-side breathing motion (doubled from original 8mm)
    y_amplitude = 0.016  # 16mm (100% increase from 8mm)
    y_freq = 0.2  # 0.2 Hz = 5 second cycle
    y_offset = y_amplitude * np.sin(2.0 * np.pi * y_freq * t)

    # Head roll - 30 degree amplitude
    roll_amplitude = 0.222  # 30 degrees in radians
    roll_freq = 0.15  # Slow, gentle roll
    roll_offset = roll_amplitude * np.sin(2.0 * np.pi * roll_freq * t)

    # No antenna movement during breathing - motors are too close to microphone
    position = config.neutral_pos + np.array([0.0, y_offset, 0.0])
    orientation = config.neutral_eul + np.array([roll_offset, 0.0, 0.0])
    antennas = np.zeros(2)

    return position, orientation, antennas


# ───────────────────────────── Noise Calibration ─────────────────────────────
def calibrate_noise_with_breathing(mini: ReachyMini, config: Config) -> np.ndarray:
    """Record noise while robot does breathing motion to capture motor sounds.

    This calibrates the noise profile to include servo/motor noise so it can
    be subtracted during BPM analysis.
    """
    print(f"\n🎤 Calibrating noise floor with breathing motion ({config.noise_calibration_duration:.0f}s)...")
    print("   Robot will move during calibration to capture motor noise.")

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=config.audio_rate,
        input=True,
        frames_per_buffer=config.audio_chunk_size,
    )

    samples_needed = int(config.audio_rate * config.noise_calibration_duration)
    collected = []
    breathing_time = 0.0

    start_time = time.time()
    last_time = start_time

    while len(collected) * config.audio_chunk_size < samples_needed:
        now = time.time()
        dt = now - last_time
        last_time = now

        elapsed = now - start_time
        remaining = config.noise_calibration_duration - elapsed
        print(f"\r   Recording noise + motor: {remaining:.1f}s remaining...", end="", flush=True)

        # Run breathing motion during calibration
        breathing_time += dt
        breath_pos, breath_ori, breath_ant = compute_breathing_pose(breathing_time, config)
        mini.set_target(
            utils.create_head_pose(*breath_pos, *breath_ori, degrees=False),
            antennas=breath_ant,
        )

        try:
            chunk = np.frombuffer(
                stream.read(config.audio_chunk_size, exception_on_overflow=False),
                dtype=np.float32,
            )
            collected.append(chunk)
        except (IOError, ValueError):
            continue

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # Return to neutral after calibration
    mini.set_target(
        utils.create_head_pose(*config.neutral_pos, *config.neutral_eul, degrees=False),
        antennas=np.zeros(2),
    )

    # Combine all chunks and compute average spectral magnitude
    audio = np.concatenate(collected)

    # Use same FFT size as we'll use for subtraction (match librosa's default)
    n_fft = 2048
    hop_length = 512

    # Compute STFT and average the magnitude across all frames
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    noise_profile = np.mean(np.abs(stft), axis=1)

    print(f"\r✅ Noise calibration complete. Motor noise captured. Profile shape: {noise_profile.shape}     ")
    return noise_profile


def calibrate_dance_noise(mini: ReachyMini, config: Config) -> np.ndarray:
    """Record noise while robot does dance moves (like headbanger) to capture dance motor sounds.

    This calibrates a noise profile for when the robot is dancing, so motor noise
    can be subtracted during BPM analysis while dancing.
    """
    print(f"\n🎤 Calibrating dance noise ({config.dance_noise_calibration_duration:.0f}s)...")
    print("   Robot will do headbanger combo to capture dance motor noise.")

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=config.audio_rate,
        input=True,
        frames_per_buffer=config.audio_chunk_size,
    )

    samples_needed = int(config.audio_rate * config.dance_noise_calibration_duration)
    collected = []
    t_beats = 0.0

    # Use headbanger_combo at 120 BPM for calibration
    move_fn, base_params, _ = AVAILABLE_MOVES["headbanger_combo"]
    params = base_params.copy()
    amp_scale = MOVE_AMPLITUDE_OVERRIDES.get("headbanger_combo", 1.0)

    start_time = time.time()
    last_time = start_time

    while len(collected) * config.audio_chunk_size < samples_needed:
        now = time.time()
        dt = now - last_time
        last_time = now

        elapsed = now - start_time
        remaining = config.dance_noise_calibration_duration - elapsed
        print(f"\r   Recording dance noise: {remaining:.1f}s remaining...", end="", flush=True)

        # Run headbanger at ~120 BPM
        t_beats += dt * (120.0 / 60.0)
        offsets = move_fn(t_beats, **params)

        scaled_pos = offsets.position_offset * amp_scale
        scaled_ori = offsets.orientation_offset * amp_scale
        scaled_ant = offsets.antennas_offset * amp_scale

        mini.set_target(
            utils.create_head_pose(
                *(config.neutral_pos + scaled_pos),
                *(config.neutral_eul + scaled_ori),
                degrees=False,
            ),
            antennas=scaled_ant,
        )

        try:
            chunk = np.frombuffer(
                stream.read(config.audio_chunk_size, exception_on_overflow=False),
                dtype=np.float32,
            )
            collected.append(chunk)
        except (IOError, ValueError):
            continue

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # Return to neutral after calibration
    mini.set_target(
        utils.create_head_pose(*config.neutral_pos, *config.neutral_eul, degrees=False),
        antennas=np.zeros(2),
    )

    # Combine all chunks and compute average spectral magnitude
    audio = np.concatenate(collected)

    n_fft = 2048
    hop_length = 512

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    noise_profile = np.mean(np.abs(stft), axis=1)

    print(f"\r✅ Dance noise calibration complete. Profile shape: {noise_profile.shape}     ")
    return noise_profile


def subtract_noise(audio: np.ndarray, noise_profile: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Subtract noise profile from audio using spectral subtraction."""
    n_fft = 2048
    hop_length = 512

    # Compute STFT of input audio
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Subtract noise profile (broadcast across time frames)
    # Use strength parameter to control how aggressive the subtraction is
    cleaned_magnitude = magnitude - (noise_profile[:, np.newaxis] * strength)

    # Floor at zero (no negative magnitudes)
    cleaned_magnitude = np.maximum(cleaned_magnitude, 0.0)

    # Reconstruct complex STFT and invert
    cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
    cleaned_audio = librosa.istft(cleaned_stft, hop_length=hop_length, length=len(audio))

    return cleaned_audio.astype(np.float32)


def clamp_bpm(bpm: float, bpm_min: float, bpm_max: float) -> float:
    """Force BPM into range by halving or doubling.

    Fixes librosa's half-time/double-time confusion where it flip-flops
    between e.g. 83 and 166 BPM for the same song.
    """
    if bpm <= 0:
        return bpm

    # Double until we're at or above minimum
    while bpm < bpm_min:
        bpm *= 2.0

    # Halve until we're at or below maximum
    while bpm > bpm_max:
        bpm /= 2.0

    return bpm


# ───────────────────────────── Worker Threads ────────────────────────────────
def audio_thread(
    state: MusicState,
    config: Config,
    stop_event: threading.Event,
    breathing_noise_profile: np.ndarray | None = None,
    dance_noise_profile: np.ndarray | None = None,
) -> None:
    """Continuously read microphone audio and update BPM + beat times."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=config.audio_rate,
        input=True,
        frames_per_buffer=config.audio_chunk_size,
    )

    buf = np.empty(0, dtype=np.float32)
    bpm_hist = collections.deque(maxlen=config.bpm_stability_buffer)

    while not stop_event.is_set():
        try:
            audio_chunk = np.frombuffer(
                stream.read(config.audio_chunk_size, exception_on_overflow=False),
                dtype=np.float32,
            )
            buf = np.append(buf, audio_chunk)
        except (IOError, ValueError):
            continue

        if len(buf) < config.audio_buffer_len:
            continue

        # Apply appropriate noise subtraction based on robot state
        analysis_buf = buf
        with state.lock:
            is_breathing = state.is_breathing
        if is_breathing and breathing_noise_profile is not None:
            # Robot is breathing - filter breathing motor noise
            analysis_buf = subtract_noise(buf, breathing_noise_profile, config.noise_subtraction_strength)
        elif not is_breathing and dance_noise_profile is not None:
            # Robot is dancing - filter dance motor noise
            analysis_buf = subtract_noise(buf, dance_noise_profile, config.noise_subtraction_strength)

        tempo, beat_frames = librosa.beat.beat_track(
            y=analysis_buf, sr=config.audio_rate, units="frames", tightness=80
        )
        now = time.time()
        raw_tempo = float(tempo[0] if isinstance(tempo, np.ndarray) and tempo.size > 0 else tempo)

        # Clamp BPM to expected range (fixes half-time/double-time confusion)
        tempo_val = clamp_bpm(raw_tempo, config.bpm_min, config.bpm_max)

        # Check RAW audio for presence (not cleaned - noise subtraction can be too aggressive)
        raw_amplitude = np.abs(buf).mean()
        cleaned_amplitude = np.abs(analysis_buf).mean()  # Still track for debug
        has_audio = raw_amplitude > 0.005 and len(beat_frames) > 0

        with state.lock:
            # Only update last_event_time if we actually detected audio
            if has_audio:
                state.last_event_time = now
            state.raw_librosa_bpm = raw_tempo  # Show raw for debugging

            if tempo_val > 40:
                bpm_hist.append(tempo_val)  # Use clamped value for averaging
                state.librosa_bpm = float(np.mean(bpm_hist))

            win_dur = len(buf) / config.audio_rate
            abs_times = [
                now - (win_dur - librosa.frames_to_time(f, sr=config.audio_rate))
                for f in beat_frames
            ]
            for t in abs_times:
                if not state.beats or t - state.beats[-1] > 0.05:
                    state.beats.append(t)

            # Track debug values
            state.raw_amplitude = raw_amplitude
            state.cleaned_amplitude = cleaned_amplitude
            state.bpm_std = float(np.std(bpm_hist)) if len(bpm_hist) > 1 else 0.0

            if len(bpm_hist) < config.bpm_stability_buffer:
                state.state = "Gathering"
                state.unstable_period_count = 0
            elif np.std(bpm_hist) < config.bpm_stability_threshold:
                state.state = "Locked"
                state.unstable_period_count = 0
                state.has_ever_locked = True
            else:
                state.state = "Unstable"
                state.unstable_period_count += 1

        # Keep a rolling audio buffer of ~1.5 s to limit CPU
        buf = buf[-int(config.audio_rate * 1.5) :]

    stream.stop_stream()
    stream.close()
    pa.terminate()


def ui_thread(data_queue: Queue, config: Config, stop_event: threading.Event):
    """Lightweight terminal UI; refresh rate controlled by config.ui_update_rate."""
    last_ui_print_time, last_data = time.time(), None
    while not stop_event.is_set():
        try:
            while True:
                last_data = data_queue.get_nowait()
        except Empty:
            pass

        now = time.time()
        if not last_data or now - last_ui_print_time < (1.0 / config.ui_update_rate):
            time.sleep(0.1)
            continue
        last_ui_print_time = now

        paused_status = " | PAUSED (Music Unstable)" if last_data["unstable_pause"] else ""
        locked_status = "YES" if last_data.get("has_ever_locked", False) else "NO"

        print(
            "\n" + "─" * 80 + "\n"
            f"🎵 Music State: {last_data['state']:<10} | BPM (Active/Raw): "
            f"{last_data['active_bpm']:.1f}/{last_data['raw_bpm']:.1f}{paused_status}\n"
            f"📊 Debug: StdDev={last_data.get('bpm_std', 0):.2f} (need <{config.bpm_stability_threshold}) | "
            f"RawAmp={last_data.get('raw_amp', 0):.4f} (need >0.005) | EverLocked={locked_status}\n"
            f"🕺 Dance State: {last_data['move_name']:<25} | Wave: {last_data['waveform']:<8} | Amp: {last_data['amp_scale']:.1f}x\n"
            f"⚙️  Settings: Auto mode | Beats/sequence: {config.beats_per_sequence}\n"
            + "─" * 80
        )
        sys.stdout.flush()


# ───────────────────────────── Plot (unchanged) ─────────────────────────────
def generate_final_plot(log, config: Config):
    if not log:
        return
    t = np.array([e["t"] for e in log])
    start_time = t[0]
    t -= start_time
    t_beats = np.array([e["t_beats"] for e in log])
    reference_t_beats = np.array([e["reference_t_beats"] for e in log])
    acc_beats = np.array([b for e in log for b in e["accepted_beats"]]) - start_time

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 8), constrained_layout=True)
    ax[0].plot(t, [e["active_bpm"] for e in log], "-", label="Active BPM")
    ax[0].set_ylabel("BPM")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, np.sin(2 * np.pi * t_beats), "-", label="Corrected Beat Clock (sin)")
    ax[1].plot(
        t,
        np.sin(2 * np.pi * reference_t_beats),
        "--",
        label="Reference Beat Clock (Metronome)",
        alpha=0.7,
    )
    ax[1].vlines(
        acc_beats,
        -1,
        1,
        colors="g",
        linestyles="solid",
        label="Accepted Beat",
        alpha=0.8,
    )
    ax[1].set_ylabel("Beat Cycle")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    path = os.path.join(
        config.save_dir, f"reachy_analysis_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
    )
    fig.savefig(path, dpi=150)
    print(f"\nAnalysis plot saved to {path}")
    plt.show()


# ───────────────────────────── Main Control Loop ────────────────────────────
def main(config: Config) -> None:
    data_queue, stop_event = Queue(), threading.Event()
    music, choreographer = MusicState(), Choreographer()

    print("Connecting to Reachy Mini...")
    with ReachyMini() as mini:
        # Skip noise calibration - just use raw audio
        breathing_noise_profile = None
        dance_noise_profile = None
        print("Skipping noise calibration - using raw audio")

        # Start audio and UI threads after calibration
        threading.Thread(
            target=audio_thread,
            args=(music, config, stop_event, breathing_noise_profile, dance_noise_profile),
            daemon=True,
        ).start()
        threading.Thread(target=ui_thread, args=(data_queue, config, stop_event), daemon=True).start()

        last_loop = time.time()
        processed_beats, active_bpm = 0, 0.0
        filtered_beat_times = collections.deque(maxlen=config.beat_buffer_size)
        full_log = []
        t_beats, reference_t_beats = 0.0, 0.0
        breathing_time = 0.0  # Track time for idle breathing animation

        print("\nRobot ready — play music!\n")

        try:
            while True:
                loop_start_time = time.time()
                dt = loop_start_time - last_loop
                last_loop = loop_start_time

                with music.lock:
                    librosa_bpm, raw_bpm = music.librosa_bpm, music.raw_librosa_bpm
                    state, last_event_time = music.state, music.last_event_time
                    unstable_count = music.unstable_period_count
                    has_ever_locked = music.has_ever_locked
                    bpm_std = music.bpm_std
                    raw_amp = music.raw_amplitude
                    cleaned_amp = music.cleaned_amplitude
                    new_beats = list(music.beats)[processed_beats:]
                processed_beats += len(new_beats)

                active_bpm = librosa_bpm if time.time() - last_event_time < config.silence_tmo else 0.0

                # Beat filtering and deduplication
                accepted_this_frame = []
                if new_beats and active_bpm > 0:
                    expected_interval = 60.0 / active_bpm
                    min_interval = expected_interval * config.min_interval_factor
                    i = 0
                    while i < len(new_beats):
                        last_beat = (
                            filtered_beat_times[-1]
                            if filtered_beat_times
                            else new_beats[i] - expected_interval
                        )
                        current_beat = new_beats[i]
                        if i + 1 < len(new_beats) and (new_beats[i + 1] - current_beat) < min_interval:
                            c = new_beats[i + 1]
                            e1 = abs((current_beat - last_beat) - expected_interval)
                            e2 = abs((c - last_beat) - expected_interval)
                            accepted_this_frame.append(current_beat if e1 <= e2 else c)
                            i += 2
                        else:
                            if (current_beat - last_beat) > min_interval:
                                accepted_this_frame.append(current_beat)
                            i += 1
                filtered_beat_times.extend(accepted_this_frame)
                last_good_beat = filtered_beat_times[-1] if filtered_beat_times else 0.0  # kept for plot consistency

                # Start/continue criteria - require has_ever_locked before any dancing
                is_allowed_to_start = state == "Locked"
                is_stable_enough_to_continue = unstable_count < config.unstable_periods_before_stop
                can_dance = active_bpm > 0 and has_ever_locked and (is_allowed_to_start or (state == "Unstable" and is_stable_enough_to_continue))

                # Tell audio thread whether we're breathing (so it knows to filter motor noise)
                with music.lock:
                    music.is_breathing = not can_dance

                if can_dance:
                    beats_this_frame = dt * (active_bpm / 60.0)
                    reference_t_beats += beats_this_frame
                    t_beats = reference_t_beats  # no smart correction

                    choreographer.advance(beats_this_frame, config)
                    move_name = choreographer.current_move_name()

                    move_fn, base_params, _ = AVAILABLE_MOVES[move_name]
                    params = base_params.copy()

                    # If the move supports waveform, use the default single waveform.
                    if "waveform" in params:
                        params["waveform"] = choreographer.current_waveform()

                    offsets = move_fn(t_beats, **params)

                    # Apply per-move amplitude scaling
                    amp_scale = MOVE_AMPLITUDE_OVERRIDES.get(move_name, 1.0)
                    scaled_pos = offsets.position_offset * amp_scale
                    scaled_ori = offsets.orientation_offset * amp_scale
                    scaled_ant = offsets.antennas_offset * amp_scale

                    mini.set_target(
                        utils.create_head_pose(
                            *(config.neutral_pos + scaled_pos),
                            *(config.neutral_eul + scaled_ori),
                            degrees=False,
                        ),
                        antennas=scaled_ant,
                    )
                else:
                    # Idle state - use breathing motion instead of static neutral
                    breathing_time += dt
                    breath_pos, breath_ori, breath_ant = compute_breathing_pose(breathing_time, config)
                    mini.set_target(
                        utils.create_head_pose(*breath_pos, *breath_ori, degrees=False),
                        antennas=breath_ant,
                    )

                # UI + log
                ui_data = {
                    "state": state,
                    "active_bpm": active_bpm,
                    "raw_bpm": raw_bpm,
                    "move_name": choreographer.current_move_name(),
                    "waveform": choreographer.current_waveform(),
                    "amp_scale": choreographer.amplitude_scale,
                    "unstable_pause": not is_stable_enough_to_continue and state == "Unstable",
                    "bpm_std": bpm_std,
                    "raw_amp": raw_amp,
                    "cleaned_amp": cleaned_amp,
                    "has_ever_locked": has_ever_locked,
                }
                data_queue.put(ui_data)
                full_log.append(
                    {
                        "t": time.time(),
                        "active_bpm": active_bpm,
                        "t_beats": t_beats,
                        "reference_t_beats": reference_t_beats,
                        "accepted_beats": accepted_this_frame,
                        "last_good_beat": last_good_beat,
                    }
                )

                time.sleep(max(0.0, config.control_ts - (time.time() - loop_start_time)))

        except KeyboardInterrupt:
            print("\nCtrl-C received, shutting down...")
        finally:
            stop_event.set()
            print("Putting robot to sleep and cleaning up...")
            # try:
            #     mini.goto_sleep()
            # except Exception:
            #     pass
            print("Shutdown complete.")
            generate_final_plot(full_log, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reachy Rhythm Controller (simplified, no keyboard, no smart correction)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--save-dir", default=".", help="Folder for PNG analysis plot")
    args = parser.parse_args()
    cfg = Config(save_dir=args.save_dir)
    main(cfg)
