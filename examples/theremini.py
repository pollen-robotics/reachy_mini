#!/usr/bin/env python3
"""
Reachy Mini Music System • v0.3  (robust)

DoF ➜ Music
───────────
Percussion (Drums)  ───────────────────────────────────────
    Left  antenna   — BPM      (60‑200)
    Right antenna   — groove   (style 0‑9)
    base_yaw        — master volume (0‑1)

Harmonic Pad  ─────────────────────────────────────────────
    head_yaw+base_yaw — root degree (major‑pentatonic)
    pitch            — chord colour (power / sus2 / sus4)
    Y (mm)           — octave (C1‑C2)
    Z (mm)           — pad loudness
Unmapped left‑overs: X (mm) & Roll ° – room for future use
"""

from __future__ import annotations
import math, queue, signal, sys, threading, time
from collections import deque
from typing import Deque, Tuple, List
from scipy.spatial.transform import Rotation as R

import numpy as np
import simpleaudio as sa
from reachy_mini import ReachyMini  # pip install reachy-mini

# ───────── CONSTANTS ────────────────────────────────────────────────────
RATE, SLICE_SEC = 44_100, 0.05
NS = int(RATE * SLICE_SEC)

LEFT_ANT_RANGE = (-math.pi, math.pi)
RIGHT_ANT_RANGE = (-math.pi, math.pi)
BASE_YAW_RANGE = (-2.0, 2.0)

BPM_MIN, BPM_MAX = 60, 200
NUM_STYLES = 10
VOL_SMOOTH_FR = 4

PENTA = [0, 2, 4, 7, 9]  # major‑pentatonic offsets


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _aff(x, a, b, y0, y1):
    return y0 if x <= a else y1 if x >= b else (x - a) * (y1 - y0) / (b - a) + y0


# ───────── DRUM SYNTH (unchanged, minor tidy) ────────────────────────────
_cache: dict[str, np.ndarray] = {}


def _exp(n, t):
    return np.exp(-np.arange(n) / RATE / t).astype(np.float32)


def _kick():
    if "k" in _cache:
        return _cache["k"]
    n = int(RATE * 0.25)
    t = np.arange(n) / RATE
    _cache["k"] = np.sin(2 * np.pi * 80 * t * (1 - 0.5 * t / 0.25)) * _exp(
        n, 0.15
    ) + 0.15 * np.sin(2 * np.pi * 180 * t) * _exp(n, 0.004)
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


def _loop(style: int, bpm: int) -> np.ndarray:
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
    buf *= 0.99 / pk if pk > 0 else 1
    return buf


def _map_drums(ants: List[float], base: float) -> Tuple[int, int, float]:
    bpm = int(_aff(ants[0], *LEFT_ANT_RANGE, BPM_MIN, BPM_MAX))
    style = int(_aff(ants[1], *RIGHT_ANT_RANGE, 0, NUM_STYLES - 0.001))
    vol = _aff(base, *BASE_YAW_RANGE, 0, 1)
    return bpm, style, vol


# ───────── PAD mapping & synth (robust) ───────────────────────────────────
_last_deg = 2  # start on degree 2 (nice)


def _midi_hz(n):
    return 440 * 2 ** ((n - 69) / 12)


def _map_pad(
    trans_mm: np.ndarray, rpy_deg: np.ndarray, base_yaw_deg: float
) -> Tuple[List[float], float]:
    global _last_deg
    # combined yaw
    yaw_deg_comb = rpy_deg[2] + base_yaw_deg
    norm = (yaw_deg_comb + 10) / 20 * 5  # 0‑5 un‑clamped
    cand_deg = int(_clip(norm, 0, 4.999))  # stay in [0‑4]
    # hysteresis 3°
    if abs(cand_deg - _last_deg) >= 1:  # changed cell
        _last_deg = cand_deg
    deg = _last_deg

    colour = int(_aff(rpy_deg[1], -10, 10, 0, 2))  # pitch °
    octave = int(_aff(trans_mm[1], -10, 10, 1, 2))  # Y mm
    amp = _aff(trans_mm[2], -15, 5, 0, 1)  # Z mm

    root = 24 + octave * 12 + PENTA[deg]
    offs = ([0, 7], [0, 2], [0, 5])[colour]
    freqs = [_midi_hz(root + o) for o in offs]
    return freqs, amp


# simple detuned saw → LPF
_pad_ph = [0.0, 0.0]
_lp = 0.0
DET = 0.997
ALPHA = 0.02


def _pad_slice(freqs: List[float], amp: float) -> np.ndarray:
    global _pad_ph, _lp
    if amp < 1e-3:
        return np.zeros(NS, np.float32)
    t = np.arange(NS)
    sig = np.zeros(NS, np.float32)
    for i, f in enumerate(freqs):
        inc = 2 * math.pi * f / RATE
        ph = _pad_ph[i] + inc * t
        saw = ((ph / math.pi) % 2) - 1
        if i:
            saw = (saw + (((ph * DET) / math.pi) % 2) - 1) * 0.5
        sig += saw
        _pad_ph[i] = (ph[-1] + inc) % (2 * math.pi)
    # LPF
    y = np.empty_like(sig)
    for n, x in enumerate(sig):
        _lp += ALPHA * (x - _lp)
        y[n] = _lp
    return (y * amp * 0.4).astype(np.float32)


# ───────── THREADS ────────────────────────────────────────────────────────
DrumMsg = Tuple[int, int, float]
PadMsg = Tuple[List[float], float]
dq: "queue.Queue[DrumMsg]" = queue.Queue()
pq: "queue.Queue[PadMsg]" = queue.Queue()


def _audio() -> None:
    bpm, style, vol = 120, 0, 1.0
    loop = _loop(style, bpm)
    idx = 0
    pend = None
    vbuf = deque([vol], maxlen=VOL_SMOOTH_FR)
    freqs = [220, 330]
    amp = 0.0
    while True:
        try:
            while True:
                nb, ns, nv = dq.get_nowait()
                style, vol = ns, nv
                bpm = nb if nb != bpm else bpm
                pend = nb
        except queue.Empty:
            pass
        try:
            while True:
                freqs, amp = pq.get_nowait()
        except queue.Empty:
            pass
        if pend and idx + NS >= loop.size:
            bpm = pend
            loop = _loop(style, bpm)
            idx = 0
            pend = None
        vbuf.append(vol)
        vol_s = sum(vbuf) / len(vbuf)
        slice_ = (
            loop[idx : idx + NS]
            if idx + NS <= loop.size
            else np.concatenate((loop[idx:], loop[: NS - (loop.size - idx)]))
        )
        idx = (idx + NS) % loop.size
        mix = _pad_slice(freqs, amp) + slice_ * vol_s
        sa.play_buffer((np.repeat(mix, 2) * 32767).astype(np.int16), 2, 2, RATE)
        time.sleep(SLICE_SEC * 0.9)


def _sensor() -> None:
    """Poll Reachy, push control queues, and print full state."""
    with ReachyMini() as r:
        r.disable_motors()

        while True:
            # ── raw sensors ────────────────────────────────────────────
            head, ants = r._get_current_joint_positions()
            base_rad = head[0]
            base_deg = math.degrees(base_rad)

            pose = r.head_kinematics.fk(head)
            trans_mm = pose[:3, 3] * 1000
            rpy_deg = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)

            # ── DRUM mapping & queue ──────────────────────────────────
            bpm, style, vol = _map_drums(ants, base_rad)
            dq.put((bpm, style, vol))

            # ── PAD mapping & queue ───────────────────────────────────
            freqs, pad_amp = _map_pad(trans_mm, rpy_deg, base_deg)
            pq.put((freqs, pad_amp))

            # Extra pad diagnostics
            yaw_sum = rpy_deg[2] + base_deg
            deg_norm = (_clip(yaw_sum, -10, 10) + 10) / 20 * 5  # 0–5
            root_deg = int(deg_norm)  # 0‑4
            chord_col = int(_aff(rpy_deg[1], -10, 10, 0, 2))  # 0/1/2
            octave = int(_aff(trans_mm[1], -10, 10, 1, 2))  # 1/2

            # ── live dashboard ───────────────────────────────────────
            sys.stdout.write("\x1b[2J\x1b[H")  # clear screen

            print("Percussion")
            print(f"  L‑antenna  (rad): {ants[0]:+.2f}  →  BPM      {bpm:3d}")
            print(f"  R‑antenna  (rad): {ants[1]:+.2f}  →  style    {style}")
            print(f"  base_yaw   (rad): {base_rad:+.2f} →  volume   {vol:.2f}")

            print("\nPad / Drone")
            print(f"  yaw_sum    (°) : {yaw_sum:+.1f}  →  degree   {root_deg}")
            print(f"  pitch      (°) : {rpy_deg[1]:+.1f}  →  colour   {chord_col}")
            print(f"  Y axis     (mm): {trans_mm[1]:+.1f}  →  octave   {octave}")
            print(f"  Z axis     (mm): {trans_mm[2]:+.1f}  →  amp      {pad_amp:.2f}")

            print("\nUnused")
            print(f"  X axis     (mm): {trans_mm[0]:+.1f}")
            print(f"  roll       (°) : {rpy_deg[0]:+.1f}")

            sys.stdout.flush()
            time.sleep(0.02)


def main() -> None:
    print("🎶 Reachy Mini Music System v0.3 — Ctrl‑C quits")
    threading.Thread(target=_audio, daemon=True).start()
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    try:
        _sensor()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
