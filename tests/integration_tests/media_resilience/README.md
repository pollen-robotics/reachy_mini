# Media resilience measurements (jitter buffer & Opus FEC)

Experimental harnesses used to characterise the Wi-Fi resilience tuning added
to the WebRTC media server in PR #1195
(`feat(media): harden conversation audio against Wi-Fi packet loss & jitter`).

That PR added, in `src/reachy_mini/media/media_server.py`:

| Knob | Leg | What |
|------|-----|------|
| `RX_JITTER_LATENCY_MS = 300` | phone → robot | consumer `webrtcbin` jitter buffer depth |
| `opusdec use-inband-fec=True, plc=True` | phone → robot | decoder loss recovery |
| `opusenc inband-fec=True, packet-loss-percentage=20` | robot mic → phone | encoder loss resilience |

These scripts answer the review question *"did you compare jitter buffer vs
FEC?"* They are standalone, do **not** touch the running daemon, and rebuild a
representative receive path with the same GStreamer elements `webrtcsink` /
`webrtcbin` wrap internally.

> **TL;DR**
> - **Jitter buffer latency is a fixed cost, paid even on a clean link** —
>   measured, see below. This is the main thing the review asked about.
> - **Jitter buffer and FEC are complementary, not alternatives** (delay vs
>   loss). Keeping both is the correct call.
> - **FEC recovery quality could not be quantified objectively offline** — the
>   reason is documented below; it needs a perceptual metric (PESQ/ViSQOL) or
>   decoder instrumentation.

## Environment

Measured on a wireless Reachy Mini (daemon `1.8.0`), GStreamer `1.26.2`,
Python `3.13`.

```bash
# Latency harness needs nothing beyond system GStreamer python bindings.
python3 measure_jitter_fec.py --sweep --duration 7

# The quality/spectral harnesses additionally need numpy:
python3 -m venv --system-site-packages /tmp/venv
/tmp/venv/bin/pip install numpy
/tmp/venv/bin/python measure_fec_spectral.py
```

## 1. `measure_jitter_fec.py` — latency (CONCLUSIVE)

Builds `audiotestsrc → opusenc → rtpopuspay → [loss] → rtpjitterbuffer →
rtpopusdepay → opusdec → fakesink(sync=true)` and measures, per config:

- the pipeline `LATENCY` query,
- the **start delay** (first packet in → first packet out), via pad probes on
  the jitter buffer sink/src,
- per-packet hold time inside the jitter buffer,
- `rtpjitterbuffer` stats (`num-pushed/num-lost/num-late/avg-jitter`).

### Result (clean local link, `avg-jitter = 0`)

| latency config | measured start delay |
|---:|---:|
| 0 ms | 1.5 ms |
| 40 ms | 36.8 ms |
| 200 ms (GStreamer default) | 197.4 ms |
| 300 ms (this PR) | 296.3 ms |

The start delay tracks the configured `latency` almost exactly, on a link with
zero measured jitter. **The GStreamer `rtpjitterbuffer` is static** (not
adaptive like a browser's): on a good Wi-Fi link it does *not* shrink towards
zero — it sits at the configured depth. Raising 200 → 300 ms therefore adds
~100 ms of permanent latency on the phone → robot leg, in exchange for
surviving jitter spikes.

This matches the GStreamer docs and maintainer statements (the `latency`
property is "the maximum latency of the jitterbuffer", and by default the
buffer "starts after the latency has elapsed").

## 2. `measure_fec_quality.py` — waveform SNR (EXPLORATORY, confounded)

Compares the lossy decode against its own clean decode (same codec config, loss
0) using waveform SNR. **Does not discriminate FEC from PLC.**

Reason: packet concealment (PLC *and* FEC) never reproduces the exact sample
phase of the original. On any time-sensitive signal the per-sample error is
large regardless of whether the audio sounds fine, so both FEC on and off score
the same low SNR. Kept to document why waveform SNR is the wrong tool here.

## 3. `measure_fec_spectral.py` — per-frame dominant frequency (EXPLORATORY, confounded)

Attempts a phase-insensitive metric: per-20 ms-frame dominant frequency of a
chirp, lossy vs clean. Also inconclusive, because `opusdec use-inband-fec`
delays output by one frame **only when it actually uses FEC**, introducing a
variable timeline shift between the lossy run and its loss-free reference. On a
chirp (frequency changes every frame) any time shift looks like a frequency
error, so the numbers are dominated by drift, not recovery quality.

## Why FEC efficacy is not numerically reported

Two independent objective approaches (waveform SNR, spectral) are both
confounded by the same root cause: concealment does not preserve exact
phase/timeline, and the deterministic test signals are time-sensitive. A
trustworthy FEC-vs-PLC number requires one of:

1. **PESQ / ViSQOL** on a real speech sample — perceptual metrics designed to be
   robust to concealment timing. This is the standard approach.
2. **Instrumenting `opusdec`** to count actual `decode_fec` reconstructions —
   a direct "how many frames did FEC save" count, not exposed by GStreamer.

Enabling Opus in-band FEC follows the WebRTC recommendation
([RFC 8854](https://www.rfc-editor.org/info/rfc8854/): *"use of the built-in
Opus FEC mechanism is RECOMMENDED"*); its receive-side cost is ~1 frame of
look-ahead and is only engaged on loss. The phone → robot leg was validated
subjectively in PR #1195 (assistant voice no longer stutters on a jittery
link). A formal perceptual A/B remains future work.

## Files

| File | Status | Needs numpy |
|------|--------|:-----------:|
| `measure_jitter_fec.py` | conclusive (latency) | no |
| `measure_fec_quality.py` | exploratory (confounded) | no |
| `measure_fec_spectral.py` | exploratory (confounded) | yes |
