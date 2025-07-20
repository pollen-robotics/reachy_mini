#!/usr/bin/env python3
"""
Reachy Mini Music System  •  v0.1

Degrees of freedom used
───────────────────────
  Antennas (2)   → drum BPM (60‑200) + groove style (0‑9)
  Base‑yaw (1)   → drum master volume
  Head joints (6)→ harmonic pad (root note, octave, chord colour, amp…)

File layout
───────────
• Drum section  …… complete, works exactly like v0.5 (BPM 60‑200)
• Pad section   …… very simple additive‑synthesis triad that always snaps
                    to a major‑pentatonic scale, so random motion sounds OK
• Mapping fns   …… **_map_drums()** and **_map_pad()** are the only bits
                    you’ll edit when iterating
"""

from __future__ import annotations
import math, queue, signal, sys, threading, time
from collections import deque
from typing import Deque, Tuple, List
from scipy.spatial.transform import Rotation as R


import numpy as np
import simpleaudio as sa
from reachy_mini import ReachyMini  # pip install reachy-mini

# ───────────────────────────── COMMON CFG ─────────────────────────────
RATE = 44_100  # Hz
SLICE_SEC = 0.05  # 50 ms → 20 fps
NS = int(RATE * SLICE_SEC)

# ───────────────────────────── DRUM ENGINE ────────────────────────────
LEFT_ANT_RANGE = (-math.pi, math.pi)  # BPM 60‑200
RIGHT_ANT_RANGE = (-math.pi, math.pi)  # style 0‑9
BASE_YAW_RANGE = (-2.0, 2.0)  # volume 0‑1
NUM_STYLES = 10
BPM_MIN, BPM_MAX = 60, 200
VOL_SMOOTH = 4  # frames


def _aff(x, a, b, y0, y1):  # linear mapper
    return y0 if x <= a else y1 if x >= b else (x - a) * (y1 - y0) / (b - a) + y0


# --- tiny synth building blocks (cached) ------------------------------
_cache: dict[str, np.ndarray] = {}


def _exp(n, tau):
    return np.exp(-np.arange(n) / RATE / tau).astype(np.float32)


def _kick():
    if "k" in _cache:
        return _cache["k"]
    n = int(RATE * 0.25)
    t = np.arange(n) / RATE
    body = np.sin(2 * np.pi * 80 * t * (1 - 0.5 * t / 0.25)) * _exp(n, 0.15)
    click = 0.15 * np.sin(2 * np.pi * 180 * t) * _exp(n, 0.004)
    _cache["k"] = body + click
    return _cache["k"]


def _snare():
    if "s" in _cache:
        return _cache["s"]
    n = int(RATE * 0.18)
    _cache["s"] = np.random.randn(n) * _exp(n, 0.05) + 0.3 * np.sin(
        2 * np.pi * 180 * np.arange(n) / RATE
    ) * _exp(n, 0.12)
    return _cache["s"]


def _hat(d=0.008):
    n = int(RATE * 0.05)
    return np.random.randn(n) * _exp(n, d)


def _rim():
    if "r" in _cache:
        return _cache["r"]
    n = int(RATE * 0.06)
    t = np.arange(n) / RATE
    _cache["r"] = 0.6 * np.sin(2 * np.pi * 1200 * t) * _exp(
        n, 0.01
    ) + 0.3 * np.random.randn(n) * _exp(n, 0.01)
    return _cache["r"]


def _clap():
    if "c" in _cache:
        return _cache["c"]
    n = int(RATE * 0.2)
    _cache["c"] = np.random.randn(n) * _exp(n, 0.03)
    return _cache["c"]


def _put(buf, s, idx):
    end = idx + s.size
    if end <= buf.size:
        buf[idx:end] += s
    else:
        split = buf.size - idx
        buf[idx:] += s[:split]
        buf[: end - buf.size] += s[split:]


def _build_loop(style: int, bpm: int) -> np.ndarray:
    q = 60 / bpm
    ln = int(RATE * q * 4)
    buf = np.zeros(ln, np.float32)
    for b in [0, 2]:
        _put(buf, _kick(), int(b * q * RATE))
    for b in [1, 3]:
        _put(buf, _snare(), int(b * q * RATE))
    if style == 1:
        [_put(buf, _hat(), int(b * q * RATE)) for b in np.arange(0, 4, 0.5)]
    elif style == 2:
        [_put(buf, _hat(), int(b * q * RATE)) for b in np.arange(0, 4, 0.5)]
        [_put(buf, _rim(), int(b * q * RATE)) for b in [1.75, 3.75]]
    elif style == 3:
        [_put(buf, _hat(0.02), int(b * q * RATE)) for b in np.arange(0, 4, 0.25)]
    elif style == 4:
        [_put(buf, _hat(), int(b * q * RATE)) for b in np.arange(0, 4, 0.5)]
        _put(buf, _clap(), int(3 * q * RATE))
        _put(buf, _hat(0.15), int(1.5 * q * RATE))
    elif style == 5:
        [_put(buf, _kick(), int(b * q * RATE)) for b in range(4)]
        [_put(buf, _hat(), int(b * q * RATE)) for b in np.arange(0, 4, 0.25)]
    elif style == 6:
        _put(buf, _kick(), 0)
        _put(buf, _snare(), int(2 * q * RATE))
        [_put(buf, _hat(0.02), int(b * q * RATE)) for b in np.arange(0, 4, 0.5)]
    elif style == 7:
        [_put(buf, _hat(0.015), int(b * q * RATE)) for b in np.arange(0, 4, 1 / 3)]
        for b in [1.5, 3.5]:
            for off in np.linspace(0, 0.125, 4):
                _put(buf, _snare(), int((b + off) * q * RATE))
    elif style == 8:
        [_put(buf, _rim(), int(b * q * RATE)) for b in [0, 1.5, 2, 3, 3.5]]
        [_put(buf, _hat(), int(b * q * RATE)) for b in np.arange(0.25, 4, 0.5)]
    elif style == 9:
        [_put(buf, _kick(), int(b * q * RATE)) for b in [0, 1.75, 2.5]]
        [_put(buf, _snare(), int(b * q * RATE)) for b in [1, 1.5, 3]]
        [_put(buf, _hat(0.015), int(b * q * RATE)) for b in np.arange(0, 4, 0.25)]
    pk = np.max(np.abs(buf))
    if pk > 0:
        buf *= 0.99 / pk
    return buf


def _map_drums(ants: List[float], base: float) -> Tuple[int, int, float]:
    bpm = int(round(_aff(ants[0], *LEFT_ANT_RANGE, BPM_MIN, BPM_MAX)))
    style = int(_aff(ants[1], *RIGHT_ANT_RANGE, 0, NUM_STYLES - 0.001))
    vol = _aff(base, *BASE_YAW_RANGE, 0, 1)
    return bpm, style, vol


# ───────────────────────────── PAD ENGINE ───────────────────────────
# major pentatonic degrees (nice & consonant)
PENTA = [0, 2, 4, 7, 9]  # semitone offsets w.r.t key
PHASES = [0.0, 0.0, 0.0]  # phase accumulators for triad


def _midi_to_hz(note: int) -> float:
    return 440 * 2 ** ((note - 69) / 12)


# ─── PAD MAPPING  (uses real‑world ranges) ────────────────────────────────
_last_root_deg = None  # hysteresis memory for root degree


def _map_pad(trans_mm: np.ndarray, rpy_deg: np.ndarray) -> Tuple[List[float], float]:
    """
    Inputs
    ──────
    • trans_mm … np.array([x, y, z])   in mm     (ranges ≈ −8…5, −10…10, −15…5)
    • rpy_deg  … np.array([roll, pitch, yaw]) in ° (ranges ≈ −10…10 each)

    Mapping
    ───────
    yaw     → root degree (0‑4, major‑pentatonic)  with 3° hysteresis
    pitch   → chord colour (0 power, 1 sus2, 2 sus4)
    z‑depth → octave (C1–C2)   deeper = lower
    y‑shift → pad amplitude (0–1)
    """
    global _last_root_deg
    # ---- root‑degree with hysteresis & clamping ----
    yaw = rpy_deg[2]
    deg_raw = (yaw - (-10)) / 20 * 5  # 0‑5 float
    deg_raw = max(0.0, min(4.999, deg_raw))  # hard‑clip
    deg = int(deg_raw)  # 0‑4 valid index

    if _last_root_deg is None or abs(_last_root_deg - yaw) > 3:
        _last_root_deg = yaw
    else:
        # keep previous snapped degree
        deg = int(((_last_root_deg - (-10)) / 20) * 5)

    pitch = rpy_deg[1]
    colour = int(_aff(pitch, -10, 10, 0, 2))

    z = trans_mm[2]
    octave = int(_aff(z, -15, 5, 1, 2))  # 1 or 2

    amp = _aff(trans_mm[1], -10, 10, 0, 1)  # y‑axis → loudness

    root_note = 24 + octave * 12 + [0, 2, 4, 7, 9][deg]  # C1=24
    offs = ([0, 7], [0, 2], [0, 5])[colour]  # power / sus2 / sus4
    freqs = [_midi_to_hz(root_note + o) for o in offs]
    return freqs, amp


# ─── PAD SYNTHESIS  (detuned saw → 1‑pole LPF) ───────────────────────────
_pad_phases = [0.0, 0.0]  # one per voice
_pad_lp_state = 0.0  # filter memory
DETUNE = 0.997  # second oscillator a few cents flat
ALPHA = 0.02  # low‑pass coefficient (smaller = darker)


def _synth_pad_slice(freqs: List[float], amp: float) -> np.ndarray:
    global _pad_phases, _pad_lp_state
    if amp < 1e-3:
        return np.zeros(NS, np.float32)

    out = np.zeros(NS, np.float32)
    for i, f in enumerate(freqs):
        phase_inc = 2 * math.pi * f / RATE
        ph = _pad_phases[i] + phase_inc * np.arange(NS)
        saw = ((ph / math.pi) % 2) - 1  # raw saw
        if i == 0:
            out += saw
        else:
            # detune the second oscillator slightly
            ph2 = ph * DETUNE
            saw2 = ((ph2 / math.pi) % 2) - 1
            out += 0.5 * (saw + saw2)
        _pad_phases[i] = (ph[-1] + phase_inc) % (2 * math.pi)

    # simple 1‑pole low‑pass for warmth
    y = np.empty_like(out)
    lp = _pad_lp_state
    for n, x in enumerate(out):
        lp += ALPHA * (x - lp)
        y[n] = lp
    _pad_lp_state = lp

    return (y * (amp * 0.4)).astype(np.float32)


# ───────────────────────────── THREADS ─────────────────────────────
DrumCtrl = Tuple[int, int, float]  # bpm, style, vol
PadCtrl = Tuple[List[float], float]  # freqs, amp
drum_q: "queue.Queue[DrumCtrl]" = queue.Queue()
pad_q: "queue.Queue[PadCtrl]" = queue.Queue()


def _audio_thread() -> None:
    # drum state
    bpm, style, vol = 120, 0, 1.0
    loop = _build_loop(style, bpm)
    idx = 0
    pend_bpm = None
    vbuf = deque([vol], maxlen=VOL_SMOOTH)
    # pad state
    freqs = [220, 275, 330]
    pad_amp = 0.0

    while True:
        # pull latest msgs
        try:
            while True:
                bpm_, style_, vol_ = drum_q.get_nowait()
                if style_ != style:
                    style = style_
                    loop = _build_loop(style, bpm)
                    idx %= loop.size
                if bpm_ != bpm:
                    pend_bpm = bpm_
                vol = vol_
        except queue.Empty:
            pass
        try:
            while True:
                freqs, pad_amp = pad_q.get_nowait()
        except queue.Empty:
            pass

        # bpm change on bar wrap
        if pend_bpm is not None and idx + NS >= loop.size:
            bpm = pend_bpm
            loop = _build_loop(style, bpm)
            idx = 0
            pend_bpm = None

        vbuf.append(vol)
        vs = sum(vbuf) / len(vbuf)

        # drum slice
        if idx + NS <= loop.size:
            sl = loop[idx : idx + NS]
        else:
            sl = np.concatenate((loop[idx:], loop[: NS - (loop.size - idx)]))
        idx = (idx + NS) % loop.size
        drums = sl * vs
        # pad slice
        pad = _synth_pad_slice(freqs, pad_amp)
        mix = drums + pad
        stereo = np.repeat(mix, 2)
        sa.play_buffer((stereo * 32767).astype(np.int16), 2, 2, RATE)
        time.sleep(SLICE_SEC * 0.9)


def _sensor_loop() -> None:
    with ReachyMini() as r:
        r.disable_motors()
        # initial defaults so print never fails
        last_drum = (120, 0, 1.0)  # bpm, style, vol
        last_pad_amp = 0.0

        while True:
            head, ants = r._get_current_joint_positions()
            base = head[0]

            # map → queues
            last_drum = _map_drums(ants, base)
            drum_q.put(last_drum)

            pose = r.head_kinematics.fk(head)
            trans_mm = pose[:3, 3] * 1000
            rpy_deg = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)

            freqs, last_pad_amp = _map_pad(trans_mm, rpy_deg)
            pad_q.put((freqs, last_pad_amp))

            # live dashboard
            bpm, style, vol = last_drum
            print(
                "\r"
                f"x {trans_mm[0]:+5.1f} mm | y {trans_mm[1]:+5.1f} mm | z {trans_mm[2]:+5.1f} mm | "
                f"roll {rpy_deg[0]:+5.1f}° | pitch {rpy_deg[1]:+5.1f}° | yaw {rpy_deg[2]:+5.1f}° | "
                f"L_ant {ants[0]:+.2f} rad | R_ant {ants[1]:+.2f} rad | "
                f"BPM {bpm:3d} | style {style} | vol {vol:.2f} | pad_amp {last_pad_amp:.2f}",
                end="",
                flush=True,
            )
            time.sleep(0.02)


# ───────────────────────────── MAIN ──────────────────────────────
def main() -> None:
    print("🎶 Reachy Mini Music System — Ctrl‑C quits")
    threading.Thread(target=_audio_thread, daemon=True).start()
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    try:
        _sensor_loop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
