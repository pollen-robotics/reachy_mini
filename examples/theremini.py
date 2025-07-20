#!/usr/bin/env python3
"""
Reachy Mini Drum‑Machine (v0.5)

Mappings (full −π…+π range)
---------------------------
• Base‑yaw  (head[0])    → master volume 0‑1
• Left antenna  (ants[0])→ tempo 1‑240 BPM (integer)
• Right antenna (ants[1])→ groove style 0‑9

• Styles switch instantly.
• Tempo changes apply on the next bar to avoid glitches.
• Ten built‑in grooves from minimal to drum‑and‑bass.
"""

from __future__ import annotations
import math, queue, signal, sys, threading, time
from collections import deque
from typing import Deque, Tuple

import numpy as np
import simpleaudio as sa
from reachy_mini import ReachyMini  # pip install reachy-mini

# ───────────────────────── constants ─────────────────────────
RATE = 44_100  # sample rate (Hz)
SLICE_SEC = 0.05  # audio slice = 50 ms ≈20 fps

LEFT_ANT_RANGE = (-math.pi, math.pi)  # rad → BPM
RIGHT_ANT_RANGE = (-math.pi, math.pi)  # rad → style
BASE_YAW_RANGE = (-2.0, 2.0)  # rad → volume

NUM_STYLES = 10
BPM_MIN, BPM_MAX = 60, 200
VOL_SMOOTH_FRAMES = 4


# ───────────────────── drum voice builders ──────────────────
def _exp_env(n: int, tau: float) -> np.ndarray:
    return np.exp(-np.arange(n) / RATE / tau).astype(np.float32)


_cache: dict[str, np.ndarray] = {}


def _kick() -> np.ndarray:
    if "kick" in _cache:
        return _cache["kick"]
    n = int(RATE * 0.25)
    t = np.arange(n) / RATE
    body = np.sin(2 * np.pi * 80 * t * (1 - 0.5 * t / 0.25)) * _exp_env(n, 0.15)
    click = 0.15 * np.sin(2 * np.pi * 180 * t) * _exp_env(n, 0.004)
    _cache["kick"] = (body + click).astype(np.float32)
    return _cache["kick"]


def _snare() -> np.ndarray:
    if "snare" in _cache:
        return _cache["snare"]
    n = int(RATE * 0.18)
    noise = np.random.randn(n) * _exp_env(n, 0.05)
    tone = 0.3 * np.sin(2 * np.pi * 180 * np.arange(n) / RATE) * _exp_env(n, 0.12)
    _cache["snare"] = (noise + tone).astype(np.float32)
    return _cache["snare"]


def _hat(decay=0.008) -> np.ndarray:
    n = int(RATE * 0.05)
    return (np.random.randn(n) * _exp_env(n, decay)).astype(np.float32)


def _rim() -> np.ndarray:
    if "rim" in _cache:
        return _cache["rim"]
    n = int(RATE * 0.06)
    t = np.arange(n) / RATE
    body = 0.6 * np.sin(2 * np.pi * 1200 * t) * _exp_env(n, 0.01)
    noise = 0.3 * np.random.randn(n) * _exp_env(n, 0.01)
    _cache["rim"] = (body + noise).astype(np.float32)
    return _cache["rim"]


def _clap() -> np.ndarray:
    if "clap" in _cache:
        return _cache["clap"]
    n = int(RATE * 0.2)
    _cache["clap"] = (np.random.randn(n) * _exp_env(n, 0.03)).astype(np.float32)
    return _cache["clap"]


# ───────────────────── loop construction helpers ────────────
def _put(buf: np.ndarray, sample: np.ndarray, idx: int) -> None:
    end = idx + sample.size
    if end <= buf.size:
        buf[idx:end] += sample
    else:
        split = buf.size - idx
        buf[idx:] += sample[:split]
        buf[: end - buf.size] += sample[split:]


def _beat_idx(beat: float, q_sec: float) -> int:
    return int(beat * q_sec * RATE)


def _build_loop(style: int, bpm: int) -> np.ndarray:
    q_sec = 60 / bpm
    length = int(RATE * q_sec * 4)
    buf = np.zeros(length, np.float32)

    # backbone
    for b in [0, 2]:
        _put(buf, _kick(), _beat_idx(b, q_sec))
    for b in [1, 3]:
        _put(buf, _snare(), _beat_idx(b, q_sec))

    if style == 0:
        pass
    elif style == 1:
        [_put(buf, _hat(), _beat_idx(b, q_sec)) for b in np.arange(0, 4, 0.5)]
    elif style == 2:
        [_put(buf, _hat(), _beat_idx(b, q_sec)) for b in np.arange(0, 4, 0.5)]
        [_put(buf, _rim(), _beat_idx(b, q_sec)) for b in [1.75, 3.75]]
    elif style == 3:
        [_put(buf, _hat(0.02), _beat_idx(b, q_sec)) for b in np.arange(0, 4, 0.25)]
    elif style == 4:
        [_put(buf, _hat(), _beat_idx(b, q_sec)) for b in np.arange(0, 4, 0.5)]
        _put(buf, _clap(), _beat_idx(3, q_sec))
        _put(buf, _hat(0.15), _beat_idx(1.5, q_sec))
    elif style == 5:
        [_put(buf, _kick(), _beat_idx(b, q_sec)) for b in range(4)]
        [_put(buf, _hat(), _beat_idx(b, q_sec)) for b in np.arange(0, 4, 0.25)]
    elif style == 6:
        _put(buf, _kick(), _beat_idx(0, q_sec))
        _put(buf, _snare(), _beat_idx(2, q_sec))
        [_put(buf, _hat(0.02), _beat_idx(b, q_sec)) for b in np.arange(0, 4, 0.5)]
    elif style == 7:
        [_put(buf, _hat(0.015), _beat_idx(b, q_sec)) for b in np.arange(0, 4, 1 / 3)]
        for b in [1.5, 3.5]:
            for off in np.linspace(0, 0.125, 4):
                _put(buf, _snare(), _beat_idx(b + off, q_sec))
    elif style == 8:
        [_put(buf, _rim(), _beat_idx(b, q_sec)) for b in [0, 1.5, 2, 3, 3.5]]
        [_put(buf, _hat(), _beat_idx(b, q_sec)) for b in np.arange(0.25, 4, 0.5)]
    elif style == 9:
        [_put(buf, _kick(), _beat_idx(b, q_sec)) for b in [0, 1.75, 2.5]]
        [_put(buf, _snare(), _beat_idx(b, q_sec)) for b in [1, 1.5, 3]]
        [_put(buf, _hat(0.015), _beat_idx(b, q_sec)) for b in np.arange(0, 4, 0.25)]

    peak = np.max(np.abs(buf))
    if peak > 0:
        buf *= 0.99 / peak
    return buf


# ───────────────────── mapping functions ─────────────────────
def _aff(x, a, b, y0, y1):
    return y0 if x <= a else y1 if x >= b else (x - a) * (y1 - y0) / (b - a) + y0


def _map(ants: list[float], base: float) -> Tuple[int, int, float]:
    bpm = max(1, int(round(_aff(ants[0], *LEFT_ANT_RANGE, BPM_MIN, BPM_MAX))))
    style = int(_aff(ants[1], *RIGHT_ANT_RANGE, 0, NUM_STYLES - 0.001))
    vol = _aff(base, *BASE_YAW_RANGE, 0, 1)
    return bpm, style, vol


# ───────────────────── audio thread ─────────────────────────
CtrlTuple = Tuple[int, int, float]
ctrl_q: "queue.Queue[CtrlTuple]" = queue.Queue()


def _audio_thread() -> None:
    bpm, style, vol = 120, 0, 1.0
    loop = _build_loop(style, bpm)
    idx, pending_bpm = 0, None
    vol_buf: Deque[float] = deque([vol], maxlen=VOL_SMOOTH_FRAMES)
    ns = int(RATE * SLICE_SEC)

    while True:
        # fetch latest controls
        try:
            while True:
                nbpm, nstyle, nvol = ctrl_q.get_nowait()
                if nstyle != style:
                    style = nstyle
                    loop = _build_loop(style, bpm)
                    idx %= loop.size
                if nbpm != bpm:
                    pending_bpm = nbpm
                vol = nvol
        except queue.Empty:
            pass

        # apply bpm on bar wrap
        if pending_bpm is not None and idx + ns >= loop.size:
            bpm = pending_bpm
            loop = _build_loop(style, bpm)
            idx = 0
            pending_bpm = None

        vol_buf.append(vol)
        vs = sum(vol_buf) / len(vol_buf)

        # slice audio
        if idx + ns <= loop.size:
            sl = loop[idx : idx + ns]
        else:
            p1 = loop[idx:]
            p2 = loop[: ns - p1.size]
            sl = np.concatenate((p1, p2))
        idx = (idx + ns) % loop.size

        stereo = np.repeat(sl * vs, 2)
        sa.play_buffer((stereo * 32767).astype(np.int16), 2, 2, RATE)
        time.sleep(SLICE_SEC * 0.9)


# ───────────────────── sensor loop ──────────────────────────
def _sensor_loop() -> None:
    with ReachyMini() as r:
        r.disable_motors()
        while True:
            head, ants = r._get_current_joint_positions()
            base = head[0]
            bpm, style, vol = _map(ants, base)
            ctrl_q.put((bpm, style, vol))
            print(
                f"\rBPM {bpm:3d} | style {style} | vol {vol:.2f} | "
                f"L {ants[0]:+.2f} | R {ants[1]:+.2f} | base {base:+.2f}",
                end="",
                flush=True,
            )
            time.sleep(0.02)


# ───────────────────── main ─────────────────────────────────
def main() -> None:
    print("🥁 Reachy Mini Drum‑Machine v0.5 — Ctrl‑C quits")
    threading.Thread(target=_audio_thread, daemon=True).start()
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    try:
        _sensor_loop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
