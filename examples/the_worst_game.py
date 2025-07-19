#!/usr/bin/env python3
"""Reachy Mini head‑pose game with groove feedback, coloured bars and CLI.

Default‑mode
------------
Speed‑run: clear **N** targets (default 4) as fast as possible.
A target is “hit” when the *score* (translation mm + rotation °) drops
below the chosen **difficulty threshold**:

* easy    → 25
* medium  → 12   (default)
* hard    → 6

Your game score is the elapsed time.

Other flags
-----------
--precision   30 s countdown – aim for the lowest score.
--cheats      Print per‑axis signed errors each frame instead of bars.
--levels N    Number of targets in speed‑run (default 4).

Visual HUD
----------
Three moving bars every frame (green < ½ thr | orange < thr | red ≥ thr):

1. **l2 (mm)** max‑range 40 mm
2. **angle (°)** max‑range 40 °
3. **score** against the difficulty threshold
"""

import argparse
import queue
import random
import sys
import threading
import time
from collections import deque
from typing import Deque, Tuple

import numpy as np
import simpleaudio as sa
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import distance_between_poses

# ────────────────────────────── audio constants ────────────────────────────
RATE: int = 44_100
SLICE_SEC: float = 0.05
BPM: int = 96
BAR_SEC: float = 60 / BPM * 4

BASE_DIST_VOL: float = 0.15
MAX_DIST_GAIN: float = 0.55
MAX_ANG_GAIN: float = 0.45

state_q: queue.Queue[Tuple[float, float]] = queue.Queue()  # (l2_mm, ang_deg)


# ────────────────────────── jingle helpers ────────────────────────────
def _env(length_s: float, decay: float = 6.0) -> np.ndarray:
    """Exponential decay envelope."""
    t = np.linspace(0.0, length_s, int(RATE * length_s), endpoint=False)
    return np.exp(-decay * t)


def _play(buf: np.ndarray) -> None:  # helper to fire and block
    sa.play_buffer((buf * 32_767).astype(np.int16), 1, 2, RATE).wait_done()


# ───────────────────────── satisfying victory jingles ──────────────────
def jingle_arpeggio() -> None:
    """Up‑arpeggiated triad • octave drop • little twinkle."""
    root = 392  # G4
    tones = [root, root * 2 ** (4 / 12), root * 2 ** (7 / 12), root * 2]
    part = []
    for f in tones:
        t = np.linspace(0, 0.12, int(RATE * 0.12), endpoint=False)
        seg = 0.5 * np.sin(2 * np.pi * f * t) * _env(0.12)
        part.append(seg)
    # final twinkle
    t = np.linspace(0, 0.25, int(RATE * 0.25), endpoint=False)
    twinkle = (
        0.3
        * (np.sin(2 * np.pi * 2 * root * t) + 0.5 * np.sin(2 * np.pi * 3 * root * t))
        * _env(0.25, 10)
    )
    buf = np.concatenate(part + [twinkle])
    _play(buf)


def jingle_minor_to_major() -> None:
    """Start on a dramatic minor triad, flips to major, ends with a bell."""
    root = 349  # F4
    t1 = np.linspace(0, 0.25, int(RATE * 0.25), endpoint=False)
    minor = (
        (
            np.sin(2 * np.pi * root * t1)
            + np.sin(2 * np.pi * root * 2 ** (3 / 12) * t1)
            + np.sin(2 * np.pi * root * 2 ** (7 / 12) * t1)
        )
        * 0.35
        * _env(0.25, 5)
    )

    t2 = np.linspace(0, 0.25, int(RATE * 0.25), endpoint=False)
    major = (
        (
            np.sin(2 * np.pi * root * 2 * t2)
            + np.sin(2 * np.pi * root * 2 ** (4 / 12) * 2 * t2)
            + np.sin(2 * np.pi * root * 2 ** (7 / 12) * 2 * t2)
        )
        * 0.35
        * _env(0.25, 5)
    )

    t3 = np.linspace(0, 0.18, int(RATE * 0.18), endpoint=False)
    bell = 0.45 * np.sin(2 * np.pi * root * 3 * t3) * _env(0.18, 10)

    _play(np.concatenate((minor, major, bell)))


def jingle_brass_fanfare() -> None:
    """Quick ascending brass‑style fanfare ending on a long tonic."""
    root = 392  # G4
    steps = [0, 2, 4, 5, 7, 9, 12]  # major scale climb
    buf_list: list[np.ndarray] = []

    for step in steps:
        f = root * 2 ** (step / 12)
        t = np.linspace(0, 0.07, int(RATE * 0.07), endpoint=False)
        tone = (
            0.4
            * (np.sin(2 * np.pi * f * t) * 0.7 + 0.3 * np.sin(2 * np.pi * f * 2 * t))
            * _env(0.07, 8)
        )
        buf_list.append(tone)

    # Final long tonic
    t_final = np.linspace(0, 0.4, int(RATE * 0.4), endpoint=False)
    final = (
        0.5
        * (
            np.sin(2 * np.pi * root * 2 * t_final) * 0.6
            + 0.4 * np.sin(2 * np.pi * root * 3 * t_final)
        )
        * _env(0.4, 3)
    )

    _play(np.concatenate(buf_list + [final]))


# ────────────────────────── drum synth helpers ─────────────────────────────
def _exp_env(n: int, tau: float) -> np.ndarray:
    return np.exp(-np.arange(n) / RATE / tau, dtype=np.float32)


def _kick() -> np.ndarray:
    n = int(RATE * 0.25)
    t = np.arange(n) / RATE
    body = np.sin(2 * np.pi * 80 * t * (1 - 0.5 * t / 0.25)) * _exp_env(n, 0.15)
    click = 0.14 * np.sin(2 * np.pi * 180 * t) * _exp_env(n, 0.004)
    return body + click


def _snare() -> np.ndarray:
    n = int(RATE * 0.18)
    noise = np.random.randn(n).astype(np.float32) * _exp_env(n, 0.05)
    tone = 0.3 * np.sin(2 * np.pi * 180 * np.arange(n) / RATE) * _exp_env(n, 0.12)
    return noise + tone


def _hat() -> np.ndarray:
    n = int(RATE * 0.05)
    return np.random.randn(n).astype(np.float32) * _exp_env(n, 0.008)


def _rim() -> np.ndarray:
    n = int(RATE * 0.06)
    t = np.arange(n) / RATE
    body = 0.6 * np.sin(2 * np.pi * 1200 * t) * _exp_env(n, 0.01)
    noise = 0.3 * np.random.randn(n).astype(np.float32) * _exp_env(n, 0.01)
    return body + noise


def _build_loop(base_only: bool = False) -> np.ndarray:
    """Build a one‑bar loop: base groove or angle layer."""
    buf = np.zeros(int(RATE * BAR_SEC), np.float32)
    q = 60 / BPM

    def put(sample: np.ndarray, pos: float) -> None:
        i = int(pos * RATE)
        buf[i : i + sample.size] += sample[: max(0, buf.size - i)]

    put(_kick(), 0)
    put(_kick(), 1.5 * q)
    put(_kick(), 2.5 * q)
    put(_snare(), 1 * q)
    put(_snare(), 3 * q)

    if not base_only:
        for off in np.arange(0, 4 * q, q / 2):
            put(_hat(), off)
        for off in [0.75 * q, 2.25 * q]:
            put(_rim(), off)
        shaker = np.random.randn(int(RATE * 0.04)).astype(np.float32) * _exp_env(
            int(RATE * 0.04), 0.005
        )
        for off in np.arange(0, 4 * q, q / 4):
            put(shaker, off)

    return buf / np.max(np.abs(buf))


DIST_LOOP = _build_loop(base_only=True)
ANG_LOOP = _build_loop(base_only=False) - DIST_LOOP
DLEN, ALEN = DIST_LOOP.size, ANG_LOOP.size


# ─────────────────────────── success jingle ───────────────────────────────-
def _make_jingle(level: int) -> np.ndarray:
    """Deterministic 0.4 s arpeggio for `level`."""
    root = 330 + level * 40
    freqs = [root, root * 2 ** (4 / 12), root * 2 ** (7 / 12)]
    seg_len = int(RATE * 0.4 / len(freqs))
    buf = np.zeros(seg_len * len(freqs), np.float32)
    for i, f in enumerate(freqs):
        t = np.arange(seg_len) / RATE
        segment = 0.45 * np.sin(2 * np.pi * f * t) * np.exp(-6 * t)
        buf[i * seg_len : (i + 1) * seg_len] = segment
    return (buf * 32_767).astype(np.int16)


def _play_jingle(level: int) -> None:
    sa.play_buffer(_make_jingle(level), 1, 2, RATE)


# ─────────────────────────── pose utilities ───────────────────────────────
def random_target_pose() -> np.ndarray:
    """Generate a random target pose for Reachy Mini."""
    x = random.uniform(-8, 5)
    y = random.uniform(-10, 10)
    z = random.uniform(-15, 5)
    roll = random.uniform(-10, 10)
    pitch = random.uniform(-10, 10)
    yaw = random.uniform(-10, 10)

    pose = np.eye(4)
    pose[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    pose[:3, 3] = np.array([x, y, z]) / 1_000  # mm → m
    return pose


def _pose_error_components(
    target: np.ndarray, pose: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    d_xyz = (pose[:3, 3] - target[:3, 3]) * 1_000
    rot_err = R.from_matrix(target[:3, :3]).inv() * R.from_matrix(pose[:3, :3])
    d_rpy = rot_err.as_euler("xyz", degrees=True)
    return d_xyz, d_rpy


# ────────────────────────── coloured bars ─────────────────────────────────
def _bar(value: float, colour_thresh: float, scale_max: float, width: int = 20) -> str:
    """Return an ANSI colour bar."""
    ratio = min(value / scale_max, 1.0)
    filled = int(ratio * width)
    empty = width - filled
    if value < 0.5 * colour_thresh:
        colour = "32"  # green
    elif value < colour_thresh:
        colour = "33"  # orange
    else:
        colour = "31"  # red
    return f"\x1b[0;{colour}m" + "█" * filled + "-" * empty + "\x1b[0m"


# ─────────────────────────── audio thread ─────────────────────────────────
def _audio_thread() -> None:
    """Stream 50 ms audio slices; keep PlayObjects alive."""
    l2_mm, ang_deg = 40.0, 40.0
    d_idx = a_idx = 0
    playing: Deque[sa.PlayObject] = deque(maxlen=32)

    while True:
        try:
            while True:
                l2_mm, ang_deg = state_q.get_nowait()
        except queue.Empty:
            pass

        dist_gain = (max(0.0, 1 - l2_mm / 40)) ** 0.7
        ang_gain = (max(0.0, 1 - ang_deg / 40)) ** 0.7

        ns = int(RATE * SLICE_SEC)

        def seg(loop: np.ndarray, idx: int) -> np.ndarray:
            return (
                loop[idx : idx + ns]
                if idx + ns < loop.size
                else np.concatenate((loop[idx:], loop[: (idx + ns) % loop.size]))
            )

        dist_slice = seg(DIST_LOOP, d_idx) * (BASE_DIST_VOL + dist_gain * MAX_DIST_GAIN)
        ang_slice = seg(ANG_LOOP, a_idx) * (ang_gain * MAX_ANG_GAIN)

        d_idx = (d_idx + ns) % DLEN
        a_idx = (a_idx + ns) % ALEN

        mix = dist_slice + ang_slice
        peak = np.max(np.abs(mix))
        if peak > 1:
            mix /= peak

        playing.append(sa.play_buffer((mix * 32_767).astype(np.int16), 1, 2, RATE))
        time.sleep(SLICE_SEC * 0.9)


# ─────────────────────────── game loops ───────────────────────────────────
def speed_run(threshold: float, cheats: bool, n_levels: int) -> None:
    """Speed‑run game: clear *n_levels* targets as fast as possible."""
    targets = [np.eye(4)] + [random_target_pose() for _ in range(n_levels - 1)]
    current = 0
    best_score = float("inf")
    start = time.monotonic()

    if cheats:
        xyz, rpy = (
            _pose_error_components(np.eye(4), targets[0])[0],
            R.from_matrix(targets[0][:3, :3]).as_euler("xyz", degrees=True),
        )
        print("Target 1 (x mm y mm z mm roll ° pitch ° yaw °):", *xyz, *rpy)

    with ReachyMini() as reachy:
        reachy.disable_motors()

        while current < n_levels:
            joints, _ = reachy._get_current_joint_positions()
            pose = reachy.head_kinematics.fk(joints)

            t_dist, a_dist, score = distance_between_poses(targets[current], pose)
            best_score = min(best_score, score)
            l2_mm = t_dist * 1_000
            ang_deg = np.degrees(a_dist)
            state_q.put((l2_mm, ang_deg))

            if cheats:
                d_xyz, d_rpy = _pose_error_components(targets[current], pose)
                print(
                    f"\rΔx={d_xyz[0]:6.1f} Δy={d_xyz[1]:6.1f} Δz={d_xyz[2]:6.1f} mm | "
                    f"Δr={d_rpy[0]:6.1f} Δp={d_rpy[1]:6.1f} Δy={d_rpy[2]:6.1f} °",
                    end="",
                )
            else:
                bars = (
                    _bar(l2_mm, threshold, 40),
                    _bar(ang_deg, threshold, 40),
                    _bar(score, threshold, threshold * 2),
                )
                print(
                    f"\r{bars[0]} {bars[1]} {bars[2]} "
                    f"score={score:6.2f} best={best_score:6.2f}",
                    end="",
                )

            if score < threshold:
                _play_jingle(current + 1)
                current += 1
                best_score = float("inf")
                if cheats and current < n_levels:
                    xyz, rpy = (
                        _pose_error_components(np.eye(4), targets[current])[0],
                        R.from_matrix(targets[current][:3, :3]).as_euler(
                            "xyz", degrees=True
                        ),
                    )
                    print("\nTarget", current + 1, ":", *xyz, *rpy)
                elif current < n_levels:
                    print("\nNext target!")

            time.sleep(0.02)

    print(f"\n🏁  Time for {n_levels} targets: {time.monotonic() - start:.2f} s")


def precision_mode() -> None:
    """30 s countdown; try for the lowest possible score."""
    deadline = time.monotonic() + 30
    best = float("inf")

    with ReachyMini() as reachy:
        reachy.disable_motors()
        while (remain := deadline - time.monotonic()) > 0:
            joints, _ = reachy._get_current_joint_positions()
            pose = reachy.head_kinematics.fk(joints)
            t_dist, a_dist, score = distance_between_poses(np.eye(4), pose)
            best = min(best, score)
            state_q.put((t_dist * 1_000, np.degrees(a_dist)))

            bars = (
                _bar(t_dist * 1_000, 40, 40),
                _bar(np.degrees(a_dist), 40, 40),
                _bar(score, 40, 40),
            )
            print(
                f"\r{bars[0]} {bars[1]} {bars[2]} "
                f"best={best:6.2f} time_left={remain:4.1f}s",
                end="",
            )
            time.sleep(0.02)

    print(f"\n⌛️  Precision best score = {best:.2f}")


# ─────────────────────────────── CLI ───────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Reachy Mini head‑pose game")
    parser.add_argument("--precision", action="store_true", help="30 s precision mode")
    parser.add_argument("--cheats", action="store_true", help="print per‑axis errors")
    parser.add_argument(
        "--difficulty",
        "-d",
        choices=("easy", "medium", "hard"),
        default="medium",
        help="threshold: 25|12|6 (default medium)",
    )
    parser.add_argument(
        "--levels",
        "-n",
        type=int,
        default=4,
        help="number of targets in speed‑run (default 4)",
    )
    return parser.parse_args()


# ─────────────────────────────── main ──────────────────────────────────────
def main() -> None:
    """Start audio thread and launch the chosen game mode."""
    state_q.put((40.0, 40.0))  # neutral seed for audio
    threading.Thread(target=_audio_thread, daemon=True).start()

    args = parse_args()
    if args.precision:
        precision_mode()
        sys.exit()

    thresholds = {"easy": 25.0, "medium": 12.0, "hard": 6.0}
    speed_run(thresholds[args.difficulty], cheats=args.cheats, n_levels=args.levels)
    if args.difficulty == "hard":
        jingle_brass_fanfare()
    elif args.difficulty == "medium":
        jingle_arpeggio()
    else:
        jingle_minor_to_major()

    time.sleep(1.0)


if __name__ == "__main__":
    main()
