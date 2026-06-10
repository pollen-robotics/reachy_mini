#!/usr/bin/env python3
"""Measure Opus in-band FEC *audio recovery* quality under packet loss.

Companion to measure_jitter_fec.py (which measured latency). This one answers
the reviewer's real question: does FEC actually recover audio better than PLC
alone, on the phone->robot leg, and by how much?

Method (fully offline, deterministic, reproducible):
  1. Generate a fixed reference signal (frequency chirp 200->3400 Hz, the voice
     band) as S16LE 48k mono. PLC predicts a steady tone too well, so a sweep +
     light noise makes concealment errors visible.
  2. Push it through a representative receive chain:
       filesrc -> opusenc(inband-fec,packet-loss-percentage)
               -> rtpopuspay -> [seeded RTP loss injector]
               -> rtpjitterbuffer(do-lost=true)   (emits GstRTPPacketLost)
               -> rtpopusdepay -> opusdec(use-inband-fec, plc)
               -> S16LE 48k mono -> appsink (captured)
  3. Reference run = loss 0. Test runs = same SEEDED loss pattern, FEC off vs on
     (identical dropped seqnums => fair A/B).
  4. Align (constant codec delay) and compute SNR of the recovered audio vs the
     clean decode. Higher SNR = better recovery.

No numpy required (pure Python).

Usage:
  python3 measure_fec_quality.py                 # default A/B at 5% and 15%
  python3 measure_fec_quality.py --loss 10 --seconds 6
"""

import argparse
import array
import math
import os
import random
import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402

RATE = 48000
CH = 1
REF_PATH = "/tmp/_fec_ref.s16"


def gen_reference(seconds):
    """Chirp 200->3400 Hz + light deterministic noise, S16LE mono."""
    n = int(RATE * seconds)
    f0, f1 = 200.0, 3400.0
    a = array.array("h")
    amp = 0.6 * 32767
    noise_amp = 0.05 * 32767
    rng = random.Random(1234)  # deterministic
    for i in range(n):
        t = i / RATE
        # linear chirp
        k = (f1 - f0) / seconds
        phase = 2 * math.pi * (f0 * t + 0.5 * k * t * t)
        s = amp * math.sin(phase) + noise_amp * (rng.random() * 2 - 1)
        if s > 32767:
            s = 32767
        elif s < -32768:
            s = -32768
        a.append(int(s))
    with open(REF_PATH, "wb") as f:
        f.write(a.tobytes())
    return n


class Decode:
    def __init__(self, fec, loss_perc, seed):
        self.fec = fec
        self.loss_perc = loss_perc
        self.seed = seed
        self.rng = random.Random(seed)
        self.out = bytearray()
        self.injected = 0
        self.pipeline = None
        self.loop = None

    def _loss_probe(self, pad, info, _):
        if self.loss_perc <= 0:
            return Gst.PadProbeReturn.OK
        if self.rng.random() * 100.0 < self.loss_perc:
            self.injected += 1
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        ok, mi = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                self.out.extend(mi.data)
            finally:
                buf.unmap(mi)
        return Gst.FlowReturn.OK

    def build(self):
        fec_bool = "true" if self.fec else "false"
        ploss = int(self.loss_perc) if self.fec else 0
        desc = (
            f"filesrc location={REF_PATH} ! "
            "rawaudioparse format=pcm pcm-format=s16le "
            f"sample-rate={RATE} num-channels={CH} ! "
            "audioconvert ! audioresample ! "
            f"opusenc inband-fec={fec_bool} packet-loss-percentage={ploss} "
            "frame-size=20 ! rtpopuspay ! "
            "application/x-rtp,media=audio,encoding-name=OPUS,clock-rate=48000,payload=96 ! "
            "identity name=lossy silent=true ! "
            "rtpjitterbuffer name=jb latency=200 do-lost=true ! "
            "rtpopusdepay ! "
            f"opusdec name=dec use-inband-fec={fec_bool} plc=true ! "
            "audioconvert ! audioresample ! "
            f"audio/x-raw,format=S16LE,rate={RATE},channels={CH},layout=interleaved ! "
            "appsink name=out emit-signals=true sync=false max-buffers=0 drop=false"
        )
        self.pipeline = Gst.parse_launch(desc)
        self.pipeline.get_by_name("lossy").get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._loss_probe, None
        )
        sink = self.pipeline.get_by_name("out")
        sink.connect("new-sample", self._on_sample)

    def run(self):
        self.build()
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()

        def on_msg(_b, msg):
            if msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                print(f"  ERROR: {err.message} | {dbg}", file=sys.stderr)
                self.loop.quit()
            elif msg.type == Gst.MessageType.EOS:
                self.loop.quit()

        bus.connect("message", on_msg)
        self.pipeline.set_state(Gst.State.PLAYING)
        GLib.timeout_add(20000, lambda: self.loop.quit())  # safety
        try:
            self.loop.run()
        finally:
            self.pipeline.set_state(Gst.State.NULL)
        return array.array("h", bytes(self.out))


def best_offset_snr(ref, test):
    """Compute SNR(dB) of test vs ref, searching a small alignment offset."""
    n = min(len(ref), len(test))
    if n == 0:
        return None, 0
    best = None
    best_off = 0
    for off in (-160, -80, -40, 0, 40, 80, 160):
        # ref[i] vs test[i+off]
        lo = max(0, -off)
        hi = min(n, n - off)
        if hi - lo < RATE:  # need >=1s overlap
            continue
        num = 0.0
        den = 0.0
        # step to keep it fast in pure python (~ every sample is fine for 6s)
        for i in range(lo, hi):
            r = ref[i]
            t = test[i + off]
            num += r * r
            d = r - t
            den += d * d
        if den <= 0:
            snr = float("inf")
        else:
            snr = 10.0 * math.log10(num / den) if num > 0 else None
        if snr is not None and (best is None or snr > best):
            best = snr
            best_off = off
    return best, best_off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", type=float, default=None,
                    help="single loss %% to test (default: sweep 5 and 15)")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Gst.init(None)

    print(f"[gen] reference chirp {args.seconds}s ...", flush=True)
    gen_reference(args.seconds)

    # Per-config clean decode: compare each codec config to ITS OWN loss-free
    # output so the only difference measured is loss recovery (not the
    # baseline codec/delay difference between FEC on and off).
    print("[ref] clean decode  FEC off (loss 0) ...", flush=True)
    ref_off = Decode(fec=False, loss_perc=0.0, seed=args.seed).run()
    print("[ref] clean decode  FEC on  (loss 0) ...", flush=True)
    ref_on = Decode(fec=True, loss_perc=0.0, seed=args.seed).run()

    # Sanity: loss-0 test vs its ref should be ~identical (very high SNR).
    sanity, _ = best_offset_snr(ref_off, ref_off)
    print(f"[sanity] self-SNR (should be huge): {sanity}", flush=True)

    losses = [args.loss] if args.loss is not None else [5.0, 15.0, 30.0]

    rows = []
    for loss in losses:
        # Same seed => identical dropped seqnums for off vs on (fair A/B).
        seed = args.seed + int(loss)
        print(f"[test] loss {loss}%  FEC off ...", flush=True)
        off_run = Decode(fec=False, loss_perc=loss, seed=seed).run()
        print(f"[test] loss {loss}%  FEC on  ...", flush=True)
        on_run = Decode(fec=True, loss_perc=loss, seed=seed).run()

        snr_off, _ = best_offset_snr(ref_off, off_run)
        snr_on, _ = best_offset_snr(ref_on, on_run)
        rows.append((loss, snr_off, snr_on))

    print()
    print("=" * 60)
    print("  FEC AUDIO RECOVERY  (SNR of recovered audio vs clean decode)")
    print("=" * 60)
    print(f"{'loss':>6} | {'PLC only (FEC off)':>20} | {'FEC on':>10} | {'gain':>8}")
    print("-" * 60)
    for loss, soff, son in rows:
        def f(x):
            return f"{x:.1f} dB" if x is not None and x != float('inf') else "  -  "
        gain = (f"+{son - soff:.1f} dB"
                if (soff is not None and son is not None
                    and soff != float('inf') and son != float('inf'))
                else "  -  ")
        print(f"{loss:>5.0f}% | {f(soff):>20} | {f(son):>10} | {gain:>8}")
    print("=" * 60)
    print("Higher SNR = closer to the clean decode = better recovery.")
    print("A positive 'gain' means in-band FEC recovered audio that PLC alone could not.")
    print()

    try:
        os.remove(REF_PATH)
    except OSError:
        pass


if __name__ == "__main__":
    main()
