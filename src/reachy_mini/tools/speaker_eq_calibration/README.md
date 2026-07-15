# Speaker EQ Calibration for Reachy Mini

The plastic head shell acts as a small enclosure that colors the speaker output
("boxy" — low-mid resonances and comb peaks). The daemon corrects this with a
GStreamer `equalizer-10bands` on the speaker branch, driven by 10 per-band gains
(dB). This directory holds the **offline** tool that derives those gains from
acoustic measurements. It is not used at runtime; the runtime side (default
gains + config lookup) lives in `reachy_mini/media/audio_utils.py`.

The gains are stored in the daemon config as `speaker_eq_gains` in
`~/.config/reachy_mini/daemon_config.json`. With no entry, the daemon falls back
to the tested default baked into
`media/audio_utils.py:DEFAULT_SPEAKER_EQ_GAINS`.

## Method: differential measurement

We measure the speaker **with** and **without** the head shell and correct only
the difference (the shell's contribution), preserving the bare speaker's
voicing. We use [hifiscan](https://github.com/erdewit/hifiscan) (chirp sweep →
frequency response). Because we take a differential, the microphone, amplifier
and room coloration cancel — so a **calibrated mic is not required**; only that
nothing moves between the two measurements.

## Workflow

### 1. Measure with hifiscan

```bash
uv pip install hifiscan
```

Connect an **external** mic at a fixed listening position (not the robot's own
mic — its XVF3800 DSP would color the measurement). Sweep the robot speaker
twice at the same volume/position, averaging several sweeps each (a single sweep
is noisy), and use hifiscan's "save" to write a pickled `Analyzer`:

- shell **on** (assembled robot) → `corr_on.pkl`
- shell **off** (speaker in its bare case) → `corr_off.pkl`

### 2. Derive the gains

```bash
uv run python src/reachy_mini/tools/speaker_eq_calibration/calibrate.py \
    gains corr_on.pkl corr_off.pkl
```

Put the printed list under `speaker_eq_gains` in `daemon_config.json`, or bake
it into `DEFAULT_SPEAKER_EQ_GAINS`. Useful flags:

- `--zero-above HZ` — zero bands above this (default `8000`, the voice-path
  Nyquist). If the speaker/mic geometry shifted more than ~1 cm between sweeps,
  the treble differential is unreliable; pass e.g. `3000`.
- `--max-boost DB` — cap positive gains (default `12`); noisy differentials can
  ask for unrealistic boosts that only add hiss.
- `--smoothing N` — hifiscan spectral smoothing (default `15`).

### 3. (Optional) Plot the calibration figure

```bash
uv run --with matplotlib python \
    src/reachy_mini/tools/speaker_eq_calibration/calibrate.py plot \
    corr_on.pkl corr_off.pkl corr_on_16k.pkl corr_off_16k.pkl \
    -o docs/assets/speaker_eq_calibration.png
```

## Caveats

- A 10-band graphic EQ has fixed octave bands, so a narrow resonance may fall
  between centers.
- It corrects the magnitude response only — not time-domain ringing or
  mechanical buzz.
- On the 16 kHz voice path only bands ≤ 8 kHz carry signal.

Verify by ear and re-measure with the EQ active; the residual coloration should
be measurably flatter.
