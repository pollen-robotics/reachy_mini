# The Reachy Mini Head Wobbler: v4 and v5 Explained

A detailed walkthrough of the audio-to-motion pipeline used by versions v4 and v5 of the Reachy Mini head wobbler, with all acronyms expanded.

## What v4 actually does, step by step

### Inputs and outputs

The wobbler receives a continuous stream of audio samples from the GStreamer media pipeline at 16 000 samples per second, single channel ("mono"), each sample a 32-bit floating-point number in the range -1.0 to +1.0. Concretely the speech tapper's `feed()` method is called repeatedly with chunks of these samples.

The output is a stream of "head pose offsets": six numbers per output, one output every 50 milliseconds. The six numbers are:

- `pitch_rad` (radians): nodding up (positive) or down (negative)
- `yaw_rad` (radians): turning left or right
- `roll_rad` (radians): tilting one ear toward a shoulder
- `x_mm`, `y_mm`, `z_mm` (millimetres): small translations of the head

These are added on top of whatever pose the application is otherwise commanding, then sent to the motors.

### Frame buffering

Audio samples arrive in chunks of arbitrary size. Inside `feed()` we maintain a rolling buffer (`self.samples`) and a small overflow buffer (`self.carry`) for partial hops. The algorithm operates on:

- A "frame" of 40 ms (640 samples), the unit of analysis. Each computed quantity uses the most recent 40 ms of audio.
- A "hop" of 50 ms (800 samples), the unit of output. Every 50 ms of audio consumed produces one output.

Each time `feed()` accumulates at least one full hop of new samples, the per-hop pipeline below runs once.

### Step 1: Loudness in decibels

Compute the root-mean-square of the most recent frame:

```
rms = sqrt(mean(frame * frame))
db  = 20 * log10(rms + tiny)
```

The result is in decibels relative to full scale. A value of 0 dB would be a sample at maximum amplitude continuously; minus infinity dB would be perfect silence. Normal speech sits around -25 to -35 dB. We use this as the basic loudness signal.

### Step 2: Voicing detection

A "voiced" sound is one where the vocal folds vibrate: vowels and consonants like m, n, l, r. An "unvoiced" sound has no folds vibration: s, f, sh, p. Voiced sounds are periodic; unvoiced sounds are noisy. We tell them apart by looking at how self-similar the audio is at a delay corresponding to a plausible vocal pitch.

The algorithm:

1. Subtract the mean of the frame (remove the slow drift).
2. Multiply by a Hann window (a smooth bell shape that tapers the ends to zero, so the next step doesn't see a discontinuity).
3. Take the Fast Fourier Transform of the windowed frame zero-padded to twice its length. This gives the spectrum.
4. Take the magnitude squared of the spectrum (the power spectrum).
5. Take the Inverse Fast Fourier Transform of the power spectrum. By the Wiener-Khinchin theorem this gives the autocorrelation of the signal: how much the signal looks like itself when shifted by various amounts of time.
6. Normalize so that zero shift equals 1.
7. Look at the autocorrelation values for shifts in a window corresponding to vocal pitches between 80 Hz and 400 Hz (which means shifts between 16000/400=40 samples and 16000/80=200 samples).
8. Find the maximum value in that window. If it is above 0.40, the frame is voiced. If not, unvoiced.

The Fast Fourier Transform is just an efficient way to compute the autocorrelation. The whole step costs about 50 microseconds on a Raspberry Pi 4. We do this 20 times per second (once per hop), so 0.1 % of one CPU core.

### Step 3: Strict voice activity detection

"Voice activity detection" is the binary decision "is the speaker currently making sound that should drive motion". We combine the loudness measurement with the voicing flag and add a small amount of memory so the decision doesn't flicker on and off frame by frame.

State variables:

- `last_voiced_age`: number of hops since the last voiced frame
- `last_loud_age`: number of hops since the last frame with loudness above -32 dB

Update each hop:

- if voiced this hop: reset `last_voiced_age` to 0; otherwise increment
- if loudness above -32 dB this hop: reset `last_loud_age` to 0; otherwise increment

Voice activity detection is "active" if and only if:

- current loudness is above -50 dB (an absolute floor we never cross; below this we consider the channel silent regardless of memory), AND
- either `last_voiced_age` is below 120 ms worth of hops, OR `last_loud_age` is below 100 ms worth of hops.

This combination is strict in the right direction: things that aren't voiced and aren't very loud (reverb tails, breath, room tone) don't keep the wobbler active. Within ~100 to 150 ms of true silence the detector turns off cleanly, even if the room is not perfectly quiet.

### Step 4: Speaker-relative automatic gain control

If the speaker is quiet, her absolute loudness numbers are smaller than a loud speaker's. Without compensation, her motion would be smaller too. We don't want that.

State variables:

- `speech_db`: a slowly-updated estimate of "what dB level does this speaker average". Initialized to -28 dB so we have a reasonable guess for the first second.
- `speech_db_init`: a flag that becomes true after the first voiced loud frame.

Update each hop:

- if voiced AND loudness > -42 dB (clearly voiced speech, not just a low-amplitude consonant), update `speech_db`:
  - if not yet initialized: snap `speech_db` to the current loudness
  - otherwise: `speech_db = speech_db + 0.040 * (current_db - speech_db)` (an exponential moving average with smoothing constant 0.040, time constant about 1.25 seconds)

When asked for the loudness gain (a number between 0 and 1 used downstream), compute it relative to the speaker's running average:

```
if absolute_db < -50:        return 0           # absolute silence floor
relative_db = current_db - speech_db
t = (relative_db - (-8)) / (6 - (-8))
t = clamp(t, 0, 1)
gain = t ** 0.7              # gentle gamma curve for perceptual softening
```

Reading: when the speaker is at her own average, relative_db = 0 and gain comes out to about 0.67. When she is 6 dB louder than her average, gain reaches 1.0. When she is 8 dB quieter than her average, gain falls to 0. The 6 and -8 numbers together set how quickly motion responds to volume changes inside speech.

### Step 5: Envelope smoothing with a floor

The "envelope" is a single number that drives the size of all motion. It is a smoothed version of the AGC gain.

```
if VAD is active:
    target_envelope = max(0.65, agc_gain(current_db))   # 0.65 is MIN_ENVELOPE
else:
    target_envelope = 0
```

The `max(0.65, ...)` is the floor. Without it, very quiet voiced moments would compute a tiny gain (say 0.2), the motor commands would be too small to overcome static friction, and the head would visibly stop. The floor guarantees that during any active speech the envelope is at least 65 % of full, so motors always get a clear command to act on.

Smoothing the envelope toward target each hop, with asymmetric attack and release:

- if rising: `envelope = envelope + 0.7 * (target - envelope)` (fast attack, time constant ~70 ms)
- if falling: `envelope = envelope + 0.3 * (target - envelope)` (slower release, time constant ~165 ms)

When VAD turns off mid-hop, additionally `envelope *= 0.6` for fast collapse to zero.

### Step 6: Rising-edge nucleus detection

A "syllable nucleus" is the loud voiced peak of each syllable, typically the vowel. We trigger a fresh gesture on each one.

State variables:

- `loud_history`: a small ring buffer holding the last 3 values of (AGC gain when voiced, otherwise 0)
- `last_nucleus_age`: hops since the last detection

Each hop:

1. Compute `loud_voiced = agc_gain(current_db) if voiced else 0`.
2. Look at the minimum of the last 3 entries of `loud_history`.
3. Fire a nucleus iff:
   - VAD is active, AND
   - `loud_voiced` is above 0.18 (an absolute floor on loudness so we don't fire on consonant noise), AND
   - `loud_voiced` exceeds the recent minimum by more than 0.08 (the rise threshold), AND
   - At least 110 ms have passed since the last firing (minimum spacing).
4. Push `loud_voiced` onto `loud_history`.

This is a "rising-edge" detector. An earlier version waited for the loudness to peak and looked back one hop to confirm. The rising-edge version fires as soon as a clear rise is detected, which in practice is 50 to 150 ms earlier in the syllable. That removes a chunk of perceived lag.

### Step 7: Direction tracking

This is the core of how nuclei translate into motion. Two 6-dimensional vectors:

- `target_dir`: where the head wants to be heading (six floats, one per axis, magnitudes typically 0 to 1).
- `current_dir`: where the head currently is (smoothly chases `target_dir`).

When a nucleus fires, sample a fresh `target_dir`:

1. Draw 6 random numbers uniformly from -1 to +1.
2. Sharpen by raising each to the 1.5 power (with sign preservation): `d[i] = sign(d[i]) * |d[i]| ** 1.5`. This makes 1 or 2 axes dominate while the others stay smaller, so each gesture has a clear directional intent rather than spreading across all six axes.
3. Normalize so the largest absolute value equals 1.
4. Multiply by `0.7 + 0.3 * loud_voiced`. Every gesture starts at 70 % of the per-axis amplitude, with a small loudness bonus on top. (Without the floor, quiet syllables produced tiny gestures that disappeared on the robot.)
5. Assign to `target_dir`.

Each hop, regardless of whether a nucleus fired:

1. Decay `target_dir`: `target_dir = target_dir * 0.88`. This is the "drift back to neutral" between nuclei. Time constant about 8 hops (400 ms).
2. Lerp `current_dir` toward `target_dir`: `current_dir = current_dir + 0.40 * (target_dir - current_dir)`. Time constant about 3 hops (150 ms). This is what makes motion smooth instead of stepped.

When VAD is inactive:

- `current_dir *= 0.6` (fast collapse toward neutral)
- `target_dir = 0` (so when speech resumes, we don't snap back to a stale gesture)

### Step 8: Continuous breath layer

Six small sinusoids, one per axis, with different frequencies and randomized initial phases (so each session looks slightly different):

| axis | frequency | amplitude |
|------|-----------|-----------|
| pitch | 1.5 Hz | 2.0 deg |
| yaw | 0.6 Hz | 4.0 deg |
| roll | 1.0 Hz | 1.3 deg |
| x | 0.4 Hz | 2.0 mm |
| y | 0.45 Hz | 1.6 mm |
| z | 0.3 Hz | 1.0 mm |

Each oscillator is multiplied by the current envelope, so it goes to zero in silence. The amplitudes are about 30 % of the nucleus-driven peaks, so the breath provides a continuous undercurrent without dominating the discrete gestures.

The breath is a deliberate borrow from v0. v0's whole approach (continuous low-frequency sine wave) was its weakness as the main signal because it had no relationship to the actual speech, but as a small always-on layer underneath the syllable-driven gestures it gives the head an "alive, breathing" feel even when no nucleus is firing.

### Step 9: Composing the output

Final per-axis values:

```
pitch_rad = radians(16) * current_dir[0] * envelope + breath_pitch
yaw_rad   = radians(28) * current_dir[1] * envelope + breath_yaw
roll_rad  = radians(10) * current_dir[2] * envelope + breath_roll
x_mm      =          16 * current_dir[3] * envelope + breath_x
y_mm      =          13 * current_dir[4] * envelope + breath_y
z_mm      =           8 * current_dir[5] * envelope + breath_z
```

The big numbers (16, 28, 10, etc.) are the per-axis maximum amplitudes at full envelope and direction. Real-world peaks are smaller because envelope hovers around 0.65 to 1.0 and the dominant `current_dir` axis hovers around 0.5 to 0.7. Typical peak yaw on the robot ends up around 12 to 20 degrees.

### Step 10: Scheduling motion against audio

The dictionary `{pitch_rad, yaw_rad, ...}` produced for each hop is then handed back to `HeadWobbler.feed()`, which schedules each hop's offsets to fire at a precise wall-clock time using GLib's main-loop timer:

```
target_ns = play_at_ns + i * hop_ns - LATENCY_COMPENSATION_NS
```

Where `play_at_ns` is "when the first sample of the audio chunk will physically come out of the speaker", `hop_ns` is 50 milliseconds in nanoseconds, and `LATENCY_COMPENSATION_NS` is the 300 ms head-start we settled on. The audiosink is configured with a matching `ts-offset` of 300 ms so the audio is delayed by exactly that amount; the wobbler then schedules motion 300 ms before the corresponding sound is heard, which gives the head time to actually reach the gesture peak just as the audio peaks.

## What v5 adds on top of v4

v5 keeps everything from v4 unchanged. It only adds a small extra signal that affects the pitch axis.

### Why

When humans speak, their head naturally tilts up slightly on rising intonation (questions, emphasis, the ends of words said with energy) and dips down slightly on falling intonation (declarative statement endings, downstep at phrase end). It is a small cue but very recognizable; without it, the head looks like it has motion but no expression.

### Streaming pitch estimation

The same Fast Fourier Transform autocorrelation that v4 uses for voicing also tells us the fundamental frequency (often abbreviated F0): the rate at which the vocal folds are vibrating, in Hertz. We extract it almost for free.

In step 2 of v4, we found the maximum autocorrelation value in the lag window 40 to 200 samples. v5 also remembers the position (the lag in samples) where that maximum occurred. Then:

- Convert lag to fundamental frequency: `f0 = sample_rate / lag` Hz.
- Refine the position with parabolic interpolation. We have three autocorrelation values around the maximum: y0 at lag-1, y1 at lag, y2 at lag+1. Fit a parabola through them. Its analytic peak is at `lag + 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)`. This gives sub-sample accuracy in the lag (so accuracy down to fractions of a hertz in F0) without enlarging the FFT.

For typical adult speech this gives an F0 reading around 80 to 300 Hz, accurate to about 1 to 2 Hz frame-to-frame.

### Smoothing F0

Raw F0 readings jitter by ±5 Hz frame-to-frame even on steady vowels. We smooth with an exponential moving average:

```
f0_smoothed = f0_smoothed + 0.5 * (raw_f0 - f0_smoothed)
```

Smoothing constant 0.5, time constant about 100 ms. Tradeoff: too much smoothing and we miss fast intonation changes; too little and we tilt the head on every glottal tick. 100 ms is a reasonable compromise for syllable-level intonation.

### Converting to semitones

F0 in Hz is hard to compare across speakers because the perceptual difference between 100 and 110 Hz is the same as between 200 and 220 Hz (it is a logarithmic scale). We convert to semitones above a 100 Hz reference:

```
f0_in_semitones = 12 * log2(f0_smoothed / 100)
```

For a typical adult male speaker around 110 Hz, this is around 1 to 2 semitones. For a typical female speaker around 220 Hz, it is around 13 to 14 semitones. The actual reference (100 Hz) does not matter; we use a relative measure below.

### Running F0 baseline

We track a slowly-adapting baseline of "what semitone level is this speaker speaking at on average":

```
if first voiced frame:
    f0_baseline = f0_in_semitones
else:
    f0_baseline = f0_baseline + 0.020 * (f0_in_semitones - f0_baseline)
```

Smoothing constant 0.020, time constant about 2.5 seconds. We snap to the first reading rather than starting from zero so the tilt is not wrong for the first second of speech.

### Tilt computation

```
tilt_in_semitones = f0_in_semitones - f0_baseline
tilt_in_degrees   = clamp(tilt_in_semitones * 1.5, ±8)
tilt_target_radians = radians(tilt_in_degrees) * envelope
```

Reading: positive when current intonation is above the speaker's running mean (rising, emphatic), negative when below (falling, downstep). The `* 1.5` says "head tilts 1.5 degrees per semitone above baseline". Clamped at ±8 degrees so a sudden dramatic intonation does not bottom out the kinematic range. Multiplied by envelope so the tilt fades to zero in silence.

### Tilt smoothing and silence handling

```
tilt_current = tilt_current + 0.40 * (tilt_target - tilt_current)
if VAD inactive: tilt_current = 0   # hard zero in silence
```

Smoothing constant 0.40, time constant about 150 ms. The hard zero in silence is important: without it, a residual non-zero tilt from the end of one phrase would leak into the silence before the next phrase, which would visibly break the strict-silence guarantee.

### Sign

The tilt is added to the pitch axis as `TILT_SIGN * tilt_current`. We empirically found that `TILT_SIGN = -1` matches the convention of "rising intonation tilts head up" on this kinematic setup. The sign is a simple flip; if you ever change motors or chassis and the sign feels wrong, change it from -1 to +1.

### Output (pitch axis only changes)

```
pitch_rad = radians(16) * current_dir[0] * envelope + breath_pitch + (-1) * tilt_current
yaw_rad   = ...   # unchanged from v4
...
```

The other five axes are byte-identical to v4.

## Why v5 felt much better at 300 ms

v4 already gives the head discrete gestures synchronized to syllables. v5 adds a slow, prosody-driven baseline movement to the pitch axis. With latency compensation right (300 ms), the gesture timing of v4 is good and the prosody of v5 lifts the head on rising tones and drops it on falling tones at the right moments. The two effects compose cleanly because they live on different timescales (v4 gestures are on the 100 to 500 ms scale, v5 tilt is on the 500 to 2000 ms scale).
