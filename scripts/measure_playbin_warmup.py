#!/usr/bin/env python3
"""Measure GStreamer playbin start latency on the robot: cold vs prerolled.

Context
-------
The daemon plays move-paired sounds through a playbin whose PLAYING
transition does lazy work (open the ALSA sink, decode, fill the ring
buffer). `audio_lead_ms` was historically used to paper over that warmup.
The daemon now prerolls (PAUSED first, then flips to PLAYING). This script
quantifies both paths in isolation, on the exact same pipeline shape as
`GstMediaServer._make_playbin_for`:

    playbin -> [tee -> queue -> audioconvert -> audioresample -> alsasink]
                    -> queue -> audioconvert -> audioresample -> appsink(sync=true)

The appsink branch mirrors the daemon's head-wobbler tap: with sync=True its
`new-sample` callback fires at the buffer's PTS on the pipeline clock, i.e.
the same instant the audiosink renders that audio. First `new-sample` after
the PLAYING transition == first audible sample (minus constant DAC latency,
identical in both scenarios).

Measured per run:
  cold_ms      wall time from set_state(PLAYING) (from NULL) to first sample
  preroll_ms   wall time spent in set_state(PAUSED) + get_state() (the warmup,
               paid ahead of the motion clock)
  primed_ms    wall time from set_state(PLAYING) (from prerolled PAUSED) to
               first sample  -> this is what the move actually waits for

Usage (on the robot, daemon MUST be stopped first - it owns the ALSA device):
    sudo systemctl stop reachy_mini
    python3 measure_playbin_warmup.py --runs 10
    python3 measure_playbin_warmup.py --file /path/to/other.wav --runs 5
    sudo systemctl start reachy_mini

No reachy_mini import needed; only pygobject + GStreamer (both ship with the
robot image).
"""

import argparse
import os
import statistics
import subprocess
import sys
import threading
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

DEFAULT_ASSETS = "/venvs/system_apps/lib/python3.11/site-packages/reachy_mini/assets"
PREROLL_TIMEOUT_NS = 2 * Gst.SECOND  # same budget as GstMediaServer
PLAY_TAIL_S = 0.25  # let the sound run briefly after first sample, then stop


def find_asset(name: str) -> str:
    """Resolve a sound name against known assets locations."""
    if os.path.exists(name):
        return os.path.abspath(name)
    for root in (DEFAULT_ASSETS, os.path.expanduser("~/reachy_mini/assets")):
        candidate = os.path.join(root, name)
        if os.path.exists(candidate):
            return candidate
    # Last resort: ask python where reachy_mini lives.
    try:
        import reachy_mini  # type: ignore

        candidate = os.path.join(os.path.dirname(reachy_mini.__file__), "assets", name)
        if os.path.exists(candidate):
            return candidate
    except ImportError:
        pass
    sys.exit(f"error: sound file not found: {name}")


def build_audiosink() -> Gst.Element:
    """Mirror GstMediaServer._build_audiosink_element (robot branch).

    Wireless CM4 defines the `reachymini_audio_sink` ALSA alias in ~/.asoundrc;
    fall back to autoaudiosink elsewhere (e.g. dry-running on a laptop).
    """
    asoundrc = os.path.expanduser("~/.asoundrc")
    if os.path.exists(asoundrc):
        with open(asoundrc) as f:
            if "reachymini_audio_sink" in f.read():
                sink = Gst.ElementFactory.make("alsasink")
                sink.set_property("device", "reachymini_audio_sink")
                return sink
    return Gst.ElementFactory.make("autoaudiosink")


class ProbeBin:
    """The daemon's tee bin, with the wobbler appsink used as a render probe."""

    def __init__(self) -> None:
        """Build the tee bin and wire the appsink probe."""
        self.first_sample_t: float | None = None
        self._bin = Gst.Bin.new("probe_audio_bin")

        tee = Gst.ElementFactory.make("tee")
        q_spk = Gst.ElementFactory.make("queue")
        ac_spk = Gst.ElementFactory.make("audioconvert")
        ar_spk = Gst.ElementFactory.make("audioresample")
        sink = build_audiosink()
        q_probe = Gst.ElementFactory.make("queue")
        ac_probe = Gst.ElementFactory.make("audioconvert")
        ar_probe = Gst.ElementFactory.make("audioresample")
        appsink = Gst.ElementFactory.make("appsink")
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", True)  # deliver at render time, like the wobbler
        appsink.set_property("max-buffers", 4)
        appsink.set_property("drop", True)
        appsink.connect("new-sample", self._on_sample)

        for el in (
            tee,
            q_spk,
            ac_spk,
            ar_spk,
            sink,
            q_probe,
            ac_probe,
            ar_probe,
            appsink,
        ):
            self._bin.add(el)
        tee.link(q_spk)
        q_spk.link(ac_spk)
        ac_spk.link(ar_spk)
        ar_spk.link(sink)
        tee.link(q_probe)
        q_probe.link(ac_probe)
        ac_probe.link(ar_probe)
        ar_probe.link(appsink)

        ghost = Gst.GhostPad.new("sink", tee.get_static_pad("sink"))
        self._bin.add_pad(ghost)

    def _on_sample(self, appsink: Gst.Element) -> Gst.FlowReturn:
        if self.first_sample_t is None:
            self.first_sample_t = time.monotonic()
        # Must pull, otherwise the streaming thread stalls.
        appsink.emit("pull-sample")
        return Gst.FlowReturn.OK

    @property
    def element(self) -> Gst.Bin:
        """The bin to install as the playbin audio-sink."""
        return self._bin


def make_playbin(file_path: str) -> tuple[Gst.Element, ProbeBin]:
    """Build a playbin wired to the probe bin, like the daemon does."""
    playbin = Gst.ElementFactory.make("playbin", "player")
    if playbin is None:
        sys.exit("error: failed to create playbin (GStreamer install issue)")
    playbin.set_property("uri", f"file://{file_path}")
    probe = ProbeBin()
    playbin.set_property("audio-sink", probe.element)
    return playbin, probe


def wait_first_sample(probe: ProbeBin, timeout_s: float = 5.0) -> float | None:
    """Busy-wait until the probe sees the first rendered sample."""
    deadline = time.monotonic() + timeout_s
    while probe.first_sample_t is None and time.monotonic() < deadline:
        time.sleep(0.001)
    return probe.first_sample_t


def run_cold(file_path: str) -> float:
    """NULL -> PLAYING in one go: the legacy play_sound path."""
    playbin, probe = make_playbin(file_path)
    t0 = time.monotonic()
    playbin.set_state(Gst.State.PLAYING)
    t_first = wait_first_sample(probe)
    latency = (t_first - t0) * 1000 if t_first is not None else float("nan")
    time.sleep(PLAY_TAIL_S)
    playbin.set_state(Gst.State.NULL)
    return latency


def run_prerolled(file_path: str) -> tuple[float, float]:
    """PAUSED + get_state (preroll), then PLAYING: the new prepare/start path."""
    playbin, probe = make_playbin(file_path)
    t0 = time.monotonic()
    playbin.set_state(Gst.State.PAUSED)
    ret, _, _ = playbin.get_state(PREROLL_TIMEOUT_NS)
    preroll_ms = (time.monotonic() - t0) * 1000
    if ret not in (Gst.StateChangeReturn.SUCCESS, Gst.StateChangeReturn.NO_PREROLL):
        playbin.set_state(Gst.State.NULL)
        return preroll_ms, float("nan")

    t1 = time.monotonic()
    playbin.set_state(Gst.State.PLAYING)
    t_first = wait_first_sample(probe)
    primed_ms = (t_first - t1) * 1000 if t_first is not None else float("nan")
    time.sleep(PLAY_TAIL_S)
    playbin.set_state(Gst.State.NULL)
    return preroll_ms, primed_ms


def stats(label: str, values: list[float]) -> None:
    """Print summary statistics for one measurement series."""
    clean = [v for v in values if v == v]  # drop NaNs
    if not clean:
        print(f"  {label:<12} all runs failed")
        return
    print(
        f"  {label:<12} n={len(clean):<3} "
        f"median={statistics.median(clean):7.1f} ms  "
        f"mean={statistics.fmean(clean):7.1f} ms  "
        f"min={min(clean):7.1f} ms  "
        f"max={max(clean):7.1f} ms"
        + (f"  stdev={statistics.stdev(clean):6.1f} ms" if len(clean) > 1 else "")
    )


def check_daemon() -> None:
    """Warn when the reachy_mini daemon holds the audio device."""
    try:
        out = subprocess.run(
            ["systemctl", "is-active", "reachy_mini"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if out == "active":
            print(
                "WARNING: the reachy_mini daemon is running and owns the audio "
                "device.\n         Results will be wrong or the sink will fail "
                "to open.\n         Run: sudo systemctl stop reachy_mini\n"
            )
    except FileNotFoundError:
        pass  # not on the robot (dry run on a dev machine)


def main() -> None:
    """Run the cold vs prerolled measurement campaign."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help="sound file(s) to test; name resolved against the assets dir "
        "(default: wake_up.wav and go_sleep.wav)",
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="runs per file (default 10)"
    )
    parser.add_argument("--mute", action="store_true", help="volume 0 (silent test)")
    args = parser.parse_args()

    Gst.init(None)
    check_daemon()

    files = args.file or ["wake_up.wav", "go_sleep.wav"]
    for name in files:
        path = find_asset(name)
        print(f"\n=== {path} ===")
        cold: list[float] = []
        preroll: list[float] = []
        primed: list[float] = []
        for i in range(args.runs):
            c = run_cold(path)
            pre, pri = run_prerolled(path)
            cold.append(c)
            preroll.append(pre)
            primed.append(pri)
            print(
                f"  run {i + 1:>2}: cold={c:7.1f} ms   "
                f"preroll={pre:7.1f} ms   primed={pri:7.1f} ms"
            )
            time.sleep(0.2)
        print()
        stats("cold", cold)
        stats("preroll", preroll)
        stats("primed", primed)
        clean_cold = [v for v in cold if v == v]
        clean_primed = [v for v in primed if v == v]
        if clean_cold and clean_primed:
            gain = statistics.median(clean_cold) - statistics.median(clean_primed)
            print(f"\n  -> preroll removes {gain:.1f} ms of start latency (median)")


if __name__ == "__main__":
    main()
