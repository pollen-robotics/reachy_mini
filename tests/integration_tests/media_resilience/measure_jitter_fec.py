#!/usr/bin/env python3
"""Standalone GStreamer harness to measure the latency contribution of the
jitter buffer and Opus in-band FEC on the Reachy Mini phone->robot voice leg.

It does NOT touch the running daemon. It rebuilds a representative receive
path:

    audiotestsrc(is-live)
      -> opusenc(inband-fec, packet-loss-percentage)
      -> rtpopuspay
      -> [optional probabilistic RTP loss injector]
      -> rtpjitterbuffer(latency, do-lost)   <-- same element webrtcbin wraps
      -> rtpopusdepay
      -> opusdec(use-inband-fec, plc)
      -> fakesink(sync=true)

For each config it reports:
  * pipeline LATENCY query (min/max) -- what GStreamer advertises as added latency
  * measured per-packet hold time through the jitter buffer (probe in vs out)
  * measured start delay (first packet in -> first packet out)
  * rtpjitterbuffer stats (num-pushed / num-lost / num-late / avg-jitter)
  * injected loss vs reconstructed (so FEC effect is visible via num-lost)

Usage:
  python3 measure_jitter_fec.py                 # default sweep
  python3 measure_jitter_fec.py --latency 300 --fec on --loss 0 --duration 8
  python3 measure_jitter_fec.py --sweep         # full matrix
"""

import argparse
import statistics
import sys
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402


def percentile(values, pct):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


class Run:
    """One measurement run for a single (latency, fec, loss) config."""

    def __init__(self, latency_ms, fec, loss_perc, duration_s):
        self.latency_ms = latency_ms
        self.fec = fec
        self.loss_perc = loss_perc
        self.duration_s = duration_s

        self.in_times = {}          # seqnum -> running-time (ns) at jb input
        self.hold_times_ms = []     # out_rt - in_rt per packet (ms)
        self.first_in_rt = None
        self.first_out_rt = None
        self.injected_drops = 0
        self.packets_in = 0
        self.packets_out = 0

        self.pipeline = None
        self.jb = None
        self.loop = None

    def _now_rt(self):
        """Pipeline running-time in ns."""
        clock = self.pipeline.get_clock()
        if clock is None:
            return 0
        return clock.get_time() - self.pipeline.get_base_time()

    @staticmethod
    def _seqnum(buf):
        """Extract RTP seqnum from a buffer (bytes 2-3 of the RTP header)."""
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            if mapinfo.size < 4:
                return None
            data = mapinfo.data
            return (data[2] << 8) | data[3]
        finally:
            buf.unmap(mapinfo)

    def _loss_probe(self, pad, info, _):
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        if self.loss_perc > 0:
            # Deterministic-ish drop based on seqnum to avoid RNG import cost.
            import random
            if random.random() * 100.0 < self.loss_perc:
                self.injected_drops += 1
                return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    def _jb_in_probe(self, pad, info, _):
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        seq = self._seqnum(buf)
        rt = self._now_rt()
        if seq is not None:
            self.in_times[seq] = rt
        if self.first_in_rt is None:
            self.first_in_rt = rt
        self.packets_in += 1
        return Gst.PadProbeReturn.OK

    def _jb_out_probe(self, pad, info, _):
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        seq = self._seqnum(buf)
        rt = self._now_rt()
        if self.first_out_rt is None:
            self.first_out_rt = rt
        if seq is not None and seq in self.in_times:
            hold_ms = (rt - self.in_times.pop(seq)) / 1e6
            if hold_ms >= 0:
                self.hold_times_ms.append(hold_ms)
        self.packets_out += 1
        return Gst.PadProbeReturn.OK

    def build(self):
        fec_bool = "true" if self.fec else "false"
        # 20ms ptime (frame-size=20) like WebRTC; voice path.
        desc = (
            "audiotestsrc is-live=true wave=ticks samplesperbuffer=960 ! "
            "audio/x-raw,rate=48000,channels=1 ! audioconvert ! "
            f"opusenc inband-fec={fec_bool} "
            f"packet-loss-percentage={int(self.loss_perc) if self.fec else 0} "
            "frame-size=20 ! rtpopuspay ! "
            "application/x-rtp,media=audio,encoding-name=OPUS,clock-rate=48000,payload=96 ! "
            "identity name=lossy silent=true ! "
            f"rtpjitterbuffer name=jb latency={self.latency_ms} do-lost=true ! "
            "rtpopusdepay ! "
            f"opusdec name=dec use-inband-fec={fec_bool} plc=true ! "
            "audioconvert ! fakesink name=sink sync=true"
        )
        self.pipeline = Gst.parse_launch(desc)
        self.jb = self.pipeline.get_by_name("jb")

        lossy = self.pipeline.get_by_name("lossy")
        lossy.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._loss_probe, None
        )
        self.jb.get_static_pad("sink").add_probe(
            Gst.PadProbeType.BUFFER, self._jb_in_probe, None
        )
        self.jb.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._jb_out_probe, None
        )

    def query_latency(self):
        q = Gst.Query.new_latency()
        if self.pipeline.query(q):
            live, mn, mx = q.parse_latency()
            mn_ms = mn / 1e6 if mn != Gst.CLOCK_TIME_NONE else None
            mx_ms = mx / 1e6 if mx != Gst.CLOCK_TIME_NONE else None
            return live, mn_ms, mx_ms
        return None, None, None

    def jb_stats(self):
        s = self.jb.get_property("stats")
        if s is None:
            return {}
        out = {}
        for key in ("num-pushed", "num-lost", "num-late",
                    "num-duplicates", "avg-jitter"):
            ok, val = (s.get_uint64(key) if key != "num-late"
                       else s.get_uint64(key))
            if ok:
                out[key] = val
        # avg-jitter is in ns
        return out

    def run(self):
        self.build()
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()

        def on_msg(_bus, msg):
            t = msg.type
            if t == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                print(f"  ERROR: {err.message} | {dbg}", file=sys.stderr)
                self.loop.quit()
            elif t == Gst.MessageType.EOS:
                self.loop.quit()

        bus.connect("message", on_msg)

        self.pipeline.set_state(Gst.State.PLAYING)
        # Wait for preroll so the LATENCY query is meaningful.
        self.pipeline.get_state(Gst.CLOCK_TIME_NONE)

        live, mn_ms, mx_ms = self.query_latency()

        GLib.timeout_add(int(self.duration_s * 1000), lambda: self.loop.quit())
        try:
            self.loop.run()
        finally:
            self.pipeline.set_state(Gst.State.NULL)

        stats = self.jb_stats()
        start_delay_ms = None
        if self.first_in_rt is not None and self.first_out_rt is not None:
            start_delay_ms = (self.first_out_rt - self.first_in_rt) / 1e6

        return {
            "latency_ms": self.latency_ms,
            "fec": self.fec,
            "loss_perc": self.loss_perc,
            "query_live": live,
            "query_min_ms": mn_ms,
            "query_max_ms": mx_ms,
            "start_delay_ms": start_delay_ms,
            "hold_n": len(self.hold_times_ms),
            "hold_mean_ms": statistics.mean(self.hold_times_ms)
            if self.hold_times_ms else None,
            "hold_p50_ms": percentile(self.hold_times_ms, 50),
            "hold_p95_ms": percentile(self.hold_times_ms, 95),
            "hold_max_ms": max(self.hold_times_ms) if self.hold_times_ms else None,
            "packets_in": self.packets_in,
            "packets_out": self.packets_out,
            "injected_drops": self.injected_drops,
            "jb_num_pushed": stats.get("num-pushed"),
            "jb_num_lost": stats.get("num-lost"),
            "jb_num_late": stats.get("num-late"),
            "jb_avg_jitter_ms": (stats.get("avg-jitter") / 1e6)
            if stats.get("avg-jitter") is not None else None,
        }


def fmt(v, unit=""):
    if v is None:
        return "  -  "
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.1f}{unit}"
    return f"{v}{unit}"


def print_report(rows):
    print()
    print("=" * 96)
    print("  RESULTS  (lat=jitterbuffer latency config, fec=Opus in-band FEC, loss=injected RTP loss)")
    print("=" * 96)
    header = (
        f"{'lat':>4} {'fec':>4} {'loss':>5} | "
        f"{'query_min':>9} {'start':>7} {'hold_p50':>8} {'hold_p95':>8} {'hold_max':>8} | "
        f"{'drop':>4} {'jb_lost':>7} {'jb_late':>7} {'avg_jit':>7}"
    )
    print(header)
    print("-" * 96)
    for r in rows:
        line = (
            f"{fmt(r['latency_ms']):>4} "
            f"{fmt(r['fec']):>4} "
            f"{fmt(r['loss_perc'],'%'):>5} | "
            f"{fmt(r['query_min_ms'],'ms'):>9} "
            f"{fmt(r['start_delay_ms'],'ms'):>7} "
            f"{fmt(r['hold_p50_ms'],'ms'):>8} "
            f"{fmt(r['hold_p95_ms'],'ms'):>8} "
            f"{fmt(r['hold_max_ms'],'ms'):>8} | "
            f"{fmt(r['injected_drops']):>4} "
            f"{fmt(r['jb_num_lost']):>7} "
            f"{fmt(r['jb_num_late']):>7} "
            f"{fmt(r['jb_avg_jitter_ms'],'ms'):>7}"
        )
        print(line)
    print("=" * 96)
    print("Reading guide:")
    print("  * query_min  : latency GStreamer advertises for the pipeline (jitter buffer dominates).")
    print("  * start      : measured delay first-packet-in -> first-packet-out (~ configured latency).")
    print("  * hold_*     : per-packet time spent inside the jitter buffer on THIS link.")
    print("  * jb_lost    : packets the jitter buffer gave up on (lower with FEC under loss).")
    print("  * avg_jit    : jitter the buffer actually measured on this (clean, local) link.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", type=int, default=300)
    ap.add_argument("--fec", choices=["on", "off"], default="on")
    ap.add_argument("--loss", type=float, default=0.0,
                    help="injected RTP loss percent")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--sweep", action="store_true",
                    help="run the full comparison matrix")
    args = ap.parse_args()

    Gst.init(None)

    rows = []
    if args.sweep:
        # Isolate each factor: latency sweep (no loss), then loss A/B at 300ms.
        configs = [
            (0, True, 0.0),
            (40, True, 0.0),
            (200, True, 0.0),
            (300, True, 0.0),
            (300, False, 5.0),
            (300, True, 5.0),
            (300, False, 15.0),
            (300, True, 15.0),
        ]
    else:
        configs = [(args.latency, args.fec == "on", args.loss)]

    for latency_ms, fec, loss in configs:
        tag = f"lat={latency_ms} fec={'on' if fec else 'off'} loss={loss}%"
        print(f"[run] {tag} ...", flush=True)
        r = Run(latency_ms, fec, loss, args.duration).run()
        rows.append(r)
        time.sleep(0.3)

    print_report(rows)


if __name__ == "__main__":
    main()
