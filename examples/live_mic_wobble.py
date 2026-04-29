"""Live mic to head wobble.

Captures the local computer's microphone, runs a chosen speech-tapper
version in-process, and pushes the resulting head-pose offsets to the
daemon via SetSpeechOffsetsCmd. Bypasses the playback-tee path
entirely: nothing is played through speakers; the offsets are computed
from mic audio and sent straight to the inverse-kinematics loop.

Requirements:
    - Reachy Mini repo installed in the active env (provides every
      speech_tapper version and SetSpeechOffsetsCmd).
    - `pip install sounddevice` (not in the core deps).
    - Daemon running and reachable.

Usage:
    python examples/live_mic_wobble.py
    python examples/live_mic_wobble.py --wobbler-version v5
    python examples/live_mic_wobble.py --device 2 --blocksize 400
    python examples/live_mic_wobble.py --list-devices
"""

import argparse
import importlib
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd

from reachy_mini import ReachyMini
from reachy_mini.io.protocol import SetSpeechOffsetsCmd

SAMPLE_RATE = 16_000  # SwayRollRT's preferred rate; matches HOP_MS arithmetic
ZERO_OFFSETS = [0.0] * 6

# Mapping from CLI version flag to the module that provides SwayRollRT.
# v0 is the original; v1 to v5 are the iterations documented in
# examples/wobbler_lab/BLOG.md.
VERSIONS = {
    "v0": "reachy_mini.motion.speech_tapper",
    "v1": "reachy_mini.motion.speech_tapper_v1",
    "v2": "reachy_mini.motion.speech_tapper_v2",
    "v3": "reachy_mini.motion.speech_tapper_v3",
    "v4": "reachy_mini.motion.speech_tapper_v4",
    "v5": "reachy_mini.motion.speech_tapper_v5",
}


def list_devices() -> None:
    """Print sounddevice's view of available input devices, then exit."""
    print(sd.query_devices())


def load_sway_class(version: str):
    """Return the SwayRollRT class for the requested wobbler version."""
    module = importlib.import_module(VERSIONS[version])
    return module.SwayRollRT


def main() -> None:
    """Entry point — parse args, open the mic, push offsets until Ctrl-C."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=None,
                        help="sounddevice input index (default: system default)")
    parser.add_argument("--blocksize", type=int, default=800,
                        help="frames per audio callback (default 800 = 50 ms = 1 hop)")
    parser.add_argument("--samplerate", type=int, default=SAMPLE_RATE,
                        help="mic capture rate (default 16000)")
    parser.add_argument("--latency", default="low",
                        help="sounddevice latency hint: 'low' | 'high' | seconds")
    parser.add_argument("--no-wake", action="store_true",
                        help="skip mini.wake_up() (use if the robot is already awake)")
    parser.add_argument("--list-devices", action="store_true",
                        help="print available audio devices and exit")
    parser.add_argument("--wobbler-version", type=str, default="v5",
                        choices=sorted(VERSIONS),
                        help="speech tapper version (default: v5)")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    sway_cls = load_sway_class(args.wobbler_version)
    sway = sway_cls(sample_rate=args.samplerate)

    # Single-slot queue: only the most-recent offsets matter. The audio
    # callback drops into this queue (non-blocking); a worker thread
    # picks the latest tuple and sends SetSpeechOffsetsCmd over the
    # WebSocket. This decouples the audio thread from any network
    # hiccup — a slow send can never glitch capture.
    offsets_q: "queue.Queue[list[float] | None]" = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):  # noqa: ANN001 (sd signature)
        if status:
            print(f"audio status: {status}", file=sys.stderr, flush=True)
        # indata is (frames, channels) float32. Take channel 0; SwayRollRT
        # expects 1-D float32 mono in [-1, 1] (sounddevice default).
        pcm = indata[:, 0] if indata.ndim == 2 else indata
        results = sway.feed(np.ascontiguousarray(pcm, dtype=np.float32))
        if not results:
            return
        # Apply only the latest hop. With blocksize=800 (one hop) this is
        # always exactly the one hop produced; with smaller blocks it's
        # the most recent of however many fit.
        r = results[-1]
        offsets = [
            r["x_mm"] / 1000.0,
            r["y_mm"] / 1000.0,
            r["z_mm"] / 1000.0,
            r["roll_rad"],
            r["pitch_rad"],
            r["yaw_rad"],
        ]
        # Replace any unsent stale offsets — only the latest matters.
        try:
            offsets_q.get_nowait()
        except queue.Empty:
            pass
        try:
            offsets_q.put_nowait(offsets)
        except queue.Full:
            pass

    def sender_loop(mini: ReachyMini) -> None:
        while not stop_event.is_set():
            try:
                offsets = offsets_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if offsets is None:
                break
            try:
                mini.client.send_command(SetSpeechOffsetsCmd(offsets=offsets))
            except Exception as exc:
                print(f"send_command failed: {exc}", file=sys.stderr, flush=True)

    with ReachyMini(log_level="WARNING") as mini:
        if not args.no_wake:
            print("Waking up...")
            mini.wake_up()

        sender = threading.Thread(target=sender_loop, args=(mini,), daemon=True)
        sender.start()

        print(
            f"Listening on device={args.device or 'default'} "
            f"@ {args.samplerate} Hz, blocksize={args.blocksize}, "
            f"wobbler={args.wobbler_version}. Ctrl-C to stop."
        )
        try:
            with sd.InputStream(
                samplerate=args.samplerate,
                channels=1,
                dtype="float32",
                blocksize=args.blocksize,
                latency=args.latency,
                device=args.device,
                callback=audio_callback,
            ):
                while True:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            stop_event.set()
            offsets_q.put(None)  # wake sender if blocked
            sender.join(timeout=1.0)
            # Park the head at zero offsets before exiting.
            try:
                mini.client.send_command(SetSpeechOffsetsCmd(offsets=ZERO_OFFSETS))
            except Exception:
                pass


if __name__ == "__main__":
    main()
