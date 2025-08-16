#!/usr/bin/env python3
"""
Reachy Mini — speech-driven 6-DoF sway (no accent detection)

What it does
- Taps system output (no mic) via soundcard loopback.
- Voice Activity Detection (VAD) from dBFS with attack/release.
- While VAD is ON: continuous 6-DoF "speech sway" modulated by loudness.
- When VAD turns OFF: sway fades out smoothly and returns to neutral.
- No accent/pitch-peaks at all (removed).
- Plots: (1) dBFS with thresholds and RT ratio, (2) commanded angles (deg).
- Two macro knobs to retune quickly across environments:
    SWAY_MASTER      → multiplies all sway amplitudes (default 1.5 ≈ +50%)
    SENS_DB_OFFSET   → shifts perceived loudness in dB (e.g., +6 makes it livelier)

Deps:
  pip install soundcard numpy matplotlib
"""

import time, math, threading
from collections import deque
from itertools import islice
import numpy as np
import soundcard as sc
import matplotlib.pyplot as plt
from queue import Queue

# ============================== TUNABLES =======================================
# Analysis cadence
SR = 16_000  # Hz
FRAME_MS = 20  # analysis frame (ms)
HOP_MS = 10  # hop (ms, 100 Hz)

# Macro knobs (fast tuning)
SWAY_MASTER = 1.5  # global multiplier for ALL motion amplitudes (1.5 ≈ +50%)
SENS_DB_OFFSET = (
    +4.0
)  # 0.0        # dB added to measured dBFS before loudness mapping (e.g. +6.0)

# VAD thresholds (durable lines on plot)
VAD_DB_ON = -35.0
VAD_DB_OFF = -45.0
VAD_ATTACK_MS = 40  # 120
VAD_RELEASE_MS = 250

# Faster envelope follow (add this next to other tunables)
ENV_FOLLOW_GAIN = 0.65

# Continuous 6-DoF "speech sway" while VAD is ON
# Frequencies (Hz) and *base* peak amplitudes (deg or mm). All amplitudes are
# multiplied by SWAY_MASTER and by a loudness envelope (see SWAY_DB_*).
SWAY_F_PITCH = 2.2
SWAY_A_PITCH_DEG = 4.5  # base 3.0 → +50% ≈ 4.5
SWAY_F_YAW = 0.6
SWAY_A_YAW_DEG = 7.5  # base 5.0 → +50% ≈ 7.5
SWAY_F_ROLL = 1.3
SWAY_A_ROLL_DEG = 2.25  # base 1.5 → +50% ≈ 2.25
SWAY_F_X = 0.35
SWAY_A_X_MM = 4.5  # base 3.0 → +50% ≈ 4.5
SWAY_F_Y = 0.45
SWAY_A_Y_MM = 3.75  # base 2.5 → +50% ≈ 3.75
SWAY_F_Z = 0.25
SWAY_A_Z_MM = 2.25  # base 1.5 → +50% ≈ 2.25

# Loudness mapping for sway (how quickly sway grows with dB)
SWAY_DB_LOW = -46.0  # below → almost no sway
SWAY_DB_HIGH = -18.0  # above → full sway
LOUDNESS_GAMMA = 0.9  # <1 slightly compresses; >1 expands

# Sway envelope (attack/release around VAD edges)
SWAY_ATTACK_MS = 50  # 120
SWAY_RELEASE_MS = 250

# Plot and monitoring
PLOT_HZ = 12
WINDOW_S = 10.0
RT_PRINT_EVERY = 2.0
RT_TOL = 0.06

# Device selection
SPEAKER_SUBSTR = "Headphones"  # "" for default device
# ==============================================================================

# Derived
FRAME = int(SR * FRAME_MS / 1000)
HOP = int(SR * HOP_MS / 1000)
ATTACK_FRAMES = max(1, int(VAD_ATTACK_MS / HOP_MS))
RELEASE_FRAMES = max(1, int(VAD_RELEASE_MS / HOP_MS))
SWAY_ATTACK_FR = max(1, int(SWAY_ATTACK_MS / HOP_MS))
SWAY_RELEASE_FR = max(1, int(SWAY_RELEASE_MS / HOP_MS))


# ------------------ helpers ------------------
def rms_dbfs(x: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(x * x) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)


def loudness_gain(db, offset=SENS_DB_OFFSET):
    """Map dBFS to 0..1 using knee [SWAY_DB_LOW, SWAY_DB_HIGH], with optional dB offset and gamma."""
    db_eff = db + offset
    t = (db_eff - SWAY_DB_LOW) / (SWAY_DB_HIGH - SWAY_DB_LOW)
    t = max(0.0, min(1.0, t))
    if LOUDNESS_GAMMA != 1.0:
        t = t**LOUDNESS_GAMMA
    return t


# ------------------ Analyzer (no F0, no accents) ------------------
class Analyzer:
    """Loopback audio → dBFS, VAD, sway envelope, RT timing."""

    def __init__(self, speaker_substr=SPEAKER_SUBSTR):
        self.speaker_substr = speaker_substr
        self.stop = False
        self.samples = deque(maxlen=10 * SR)

        # histories (for plots)
        self.t_hist, self.db_hist = [], []

        # state
        self.frame_idx = 0
        self.start_wall = None

        # VAD w/ hysteresis
        self.vad_on = False
        self.vad_above = 0
        self.vad_below = 0

        # Sway envelope (0..1), independent attack/release for smooth fade
        self.sway_env = 0.0
        self.sway_up = 0
        self.sway_down = 0

    def _choose_loopback(self):
        spk = None
        for s in sc.all_speakers():
            if self.speaker_substr and self.speaker_substr.lower() in s.name.lower():
                spk = s
                break
        if spk is None:
            spk = sc.default_speaker()
        try:
            mic = sc.get_microphone(id=spk.name, include_loopback=True)
        except Exception:
            mic = None
        if mic is None:
            for m in sc.all_microphones(include_loopback=True):
                nm = m.name.lower()
                if (
                    (self.speaker_substr and self.speaker_substr.lower() in nm)
                    or "monitor" in nm
                    or "loopback" in nm
                ):
                    mic = m
                    break
        if mic is None:
            raise RuntimeError("No loopback microphone found")
        return mic, spk

    def run(self):
        mic, spk = self._choose_loopback()
        block = max(HOP, 512)
        print(f"[audio] Loopback of: {spk.name}")
        with mic.recorder(samplerate=SR, channels=1, blocksize=block) as r:
            carry = np.zeros(0, dtype=np.float32)
            self.start_wall = time.time()
            while not self.stop:
                data = r.record(numframes=block).astype(np.float32).flatten()
                if data.size == 0:
                    continue
                carry = np.concatenate([carry, data])

                while carry.size >= HOP:
                    # advance one hop
                    self.samples.extend(carry[:HOP].tolist())
                    carry = carry[HOP:]
                    if len(self.samples) < FRAME:
                        continue

                    # analyze last frame
                    frame = np.fromiter(
                        islice(
                            self.samples, len(self.samples) - FRAME, len(self.samples)
                        ),
                        dtype=np.float32,
                        count=FRAME,
                    )
                    db = rms_dbfs(frame)

                    # VAD (hard gate)
                    if db >= VAD_DB_ON:
                        self.vad_above += 1
                        self.vad_below = 0
                        if not self.vad_on and self.vad_above >= ATTACK_FRAMES:
                            self.vad_on = True
                    elif db <= VAD_DB_OFF:
                        self.vad_below += 1
                        self.vad_above = 0
                        if self.vad_on and self.vad_below >= RELEASE_FRAMES:
                            self.vad_on = False

                    # Sway envelope follows VAD with smooth attack/release
                    if self.vad_on:
                        self.sway_up = min(SWAY_ATTACK_FR, self.sway_up + 1)
                        self.sway_down = 0
                    else:
                        self.sway_down = min(SWAY_RELEASE_FR, self.sway_down + 1)
                        self.sway_up = 0
                    up_gain = self.sway_up / SWAY_ATTACK_FR
                    down_gain = 1.0 - (self.sway_down / SWAY_RELEASE_FR)
                    target_env = up_gain if self.vad_on else down_gain
                    # 1st-order follow (critically damped-ish)
                    self.sway_env += ENV_FOLLOW_GAIN * (target_env - self.sway_env)
                    self.sway_env = max(0.0, min(1.0, self.sway_env))

                    # logs for plotting
                    t_rel = self.frame_idx * (HOP_MS / 1000.0)
                    self.t_hist.append(t_rel)
                    self.db_hist.append(db)

                    self.frame_idx += 1


class SpeechTapper:
    def __init__(self, plot=False):
        self.plot = plot
        self.queue = Queue(maxsize=1)
        self.last_commands = [0, 0, 0, 0, 0, 0]

    # ------------------ Control + plots ------------------
    def run(self):
        an = Analyzer()
        th = threading.Thread(target=an.run, daemon=True)
        th.start()
        if self.plot:
            # 2 panels now: top dBFS, bottom commanded angles
            plt.ion()
            fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(9, 6), sharex=False)

            # Top: dBFS + thresholds
            ax1.set_title("Energy dBFS with VAD thresholds")
            ax1.set_ylim(-80, 0)
            ax1.set_xlim(0, WINDOW_S)
            (line_db,) = ax1.plot([], [], label="dBFS")
            ax1.axhline(
                VAD_DB_ON,
                color="tab:green",
                linestyle="--",
                linewidth=1.0,
                label=f"VAD ON {VAD_DB_ON} dB",
            )
            ax1.axhline(
                VAD_DB_OFF,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                label=f"VAD OFF {VAD_DB_OFF} dB",
            )
            vad_poly = None
            rt_text = ax1.text(0.01, 0.05, "", transform=ax1.transAxes)
            ax1.legend(loc="lower right")

            # Bottom: commanded angles
            ax3.set_title("Commanded angles (deg)")
            ax3.set_ylim(-30, 30)
            ax3.set_xlim(0, WINDOW_S)
            (cmd_pitch_line,) = ax3.plot([], [], linestyle="--", label="pitch cmd")
            (cmd_yaw_line,) = ax3.plot([], [], linestyle=":", label="yaw cmd")
            (cmd_roll_line,) = ax3.plot([], [], linestyle="-.", label="roll cmd")
            ax3.legend(loc="lower right")

            last_draw = 0.0
            last_rt_print = 0.0

            # histories (audio-time base) for plot
            cmd_t_hist, cmd_pitch_deg_hist, cmd_yaw_deg_hist, cmd_roll_deg_hist = (
                [],
                [],
                [],
                [],
            )

        # random phases so sway isn’t locked
        rng = np.random.default_rng(7)
        phase_pitch = rng.random() * 2 * math.pi
        phase_yaw = rng.random() * 2 * math.pi
        phase_roll = rng.random() * 2 * math.pi
        phase_x = rng.random() * 2 * math.pi
        phase_y = rng.random() * 2 * math.pi
        phase_z = rng.random() * 2 * math.pi

        try:
            dt = 0.01  # 100 Hz
            offsets = [0, 0, 0, 0, 0, 0]
            while True:
                now = time.time()
                audio_time_now = (
                    an.frame_idx * (HOP_MS / 1000.0) if an.start_wall else 0.0
                )

                # Loudness (with offset) and envelope
                db_now = an.db_hist[-1] if an.db_hist else -80.0
                loud = loudness_gain(db_now) * SWAY_MASTER  # master gain applied here
                env = an.sway_env  # 0..1 fade with VAD

                # angular sways (radians)
                pitch_sway = (
                    math.radians(SWAY_A_PITCH_DEG)
                    * loud
                    * env
                    * math.sin(2 * math.pi * SWAY_F_PITCH * now + phase_pitch)
                )
                yaw_sway = (
                    math.radians(SWAY_A_YAW_DEG)
                    * loud
                    * env
                    * math.sin(2 * math.pi * SWAY_F_YAW * now + phase_yaw)
                )
                roll_sway = (
                    math.radians(SWAY_A_ROLL_DEG)
                    * loud
                    * env
                    * math.sin(2 * math.pi * SWAY_F_ROLL * now + phase_roll)
                )

                # translational sways (mm)
                x_sway = (
                    SWAY_A_X_MM
                    * loud
                    * env
                    * math.sin(2 * math.pi * SWAY_F_X * now + phase_x)
                )
                y_sway = (
                    SWAY_A_Y_MM
                    * loud
                    * env
                    * math.sin(2 * math.pi * SWAY_F_Y * now + phase_y)
                )
                z_sway = (
                    SWAY_A_Z_MM
                    * loud
                    * env
                    * math.sin(2 * math.pi * SWAY_F_Z * now + phase_z)
                )

                # final commands
                pitch_cmd = pitch_sway
                yaw_cmd = yaw_sway
                roll_cmd = roll_sway
                print(f"Commands: {x_sway}, {y_sway}, {z_sway}, {roll_cmd}, {pitch_cmd}, {yaw_cmd}")
                self.queue.put(
                    [
                        x_sway / 1000,
                        y_sway / 1000,
                        z_sway / 1000,
                        roll_cmd,
                        pitch_cmd,
                        yaw_cmd,
                    ]
                )

                if self.plot:
                    # record commands (deg) for plot
                    cmd_t_hist.append(audio_time_now)
                    cmd_pitch_deg_hist.append(math.degrees(pitch_cmd))
                    cmd_yaw_deg_hist.append(math.degrees(yaw_cmd))
                    cmd_roll_deg_hist.append(math.degrees(roll_cmd))

                # RT monitor
                if self.plot:
                    if an.start_wall is not None:
                        wall_time = now - an.start_wall
                        audio_time = an.frame_idx * (HOP_MS / 1000.0)
                        rt_ratio = (audio_time / wall_time) if wall_time > 0 else 0.0
                        if (now - last_rt_print) >= RT_PRINT_EVERY:
                            status = (
                                "OK"
                                if abs(rt_ratio - 1.0) <= RT_TOL
                                else ("SLOW" if rt_ratio < 1.0 - RT_TOL else "FAST")
                            )
                            print(
                                f"[rt] audio={audio_time:.2f}s wall={wall_time:.2f}s ratio={rt_ratio:.3f} [{status}]"
                            )
                            last_rt_print = now

                    # plots
                    if (now - last_draw) >= (1.0 / PLOT_HZ) and an.t_hist:
                        tmax = an.t_hist[-1]
                        tmin = max(0.0, tmax - WINDOW_S)

                        def last_window(t_all, y_all):
                            if not t_all:
                                return [], []
                            i0 = 0
                            for i in range(len(t_all) - 1, -1, -1):
                                if t_all[i] < tmin:
                                    i0 = i + 1
                                    break
                            return t_all[i0:], y_all[i0:]

                        tx, dbx = last_window(an.t_hist, an.db_hist)
                        tc, pcmd = last_window(cmd_t_hist, cmd_pitch_deg_hist)
                        _, ycmd = last_window(cmd_t_hist, cmd_yaw_deg_hist)
                        _, rcmd = last_window(cmd_t_hist, cmd_roll_deg_hist)

                        line_db.set_data(tx, dbx)
                        cmd_pitch_line.set_data(tc, pcmd)
                        cmd_yaw_line.set_data(tc, ycmd)
                        cmd_roll_line.set_data(tc, rcmd)

                        # --- autoscale commanded-angles panel (robust, symmetric, not jittery) ---
                        if tc:
                            vals = np.concatenate(
                                [
                                    np.asarray(pcmd, dtype=float),
                                    np.asarray(ycmd, dtype=float),
                                    np.asarray(rcmd, dtype=float),
                                ]
                            )
                            # 99th percentile of absolute value to ignore outliers
                            amax = (
                                float(np.percentile(np.abs(vals), 99))
                                if vals.size
                                else 5.0
                            )
                            amax = max(5.0, amax)  # never tighter than ±5°
                            ax3.set_ylim(
                                -1.2 * amax, 1.2 * amax
                            )  # small headroom to avoid clipping

                        for ax in (ax1, ax3):
                            ax.set_xlim(tmin, max(tmin + 0.5, tmax))

                        # VAD shading (current state)
                        if vad_poly is not None:
                            vad_poly.remove()
                            vad_poly = None
                        if tx:
                            y0 = [-80] * len(tx)
                            y1 = [0 if an.vad_on else -80] * len(tx)
                            vad_poly = ax1.fill_between(
                                tx, y0, y1, alpha=0.10, step="pre", color="gray"
                            )

                        # RT overlay
                        if an.start_wall is not None:
                            wall_time = now - an.start_wall
                            audio_time = an.frame_idx * (HOP_MS / 1000.0)
                            rt_ratio = (
                                (audio_time / wall_time) if wall_time > 0 else 0.0
                            )
                            status = (
                                "OK"
                                if abs(rt_ratio - 1.0) <= RT_TOL
                                else ("SLOW" if rt_ratio < 1.0 - RT_TOL else "FAST")
                            )
                            rt_text.set_text(f"RT ratio: {rt_ratio:.3f} [{status}]")

                        plt.pause(0.001)
                        last_draw = now

                time.sleep(dt)
        except KeyboardInterrupt:
            print("\nCtrl-C. Shutting down...")
        finally:
            an.stop = True

    def get_commands(self):
        try:
            self.last_commands = self.queue.get(False)  # non blocking
        except Exception:
            pass

        return self.last_commands


if __name__ == "__main__":
    st = SpeechTapper()
