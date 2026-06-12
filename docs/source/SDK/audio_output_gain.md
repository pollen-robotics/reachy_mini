# Audio Output Gain (REACHY_AUDIO_GAIN_DB)

## Purpose

Reachy Mini is not well audible in noisy environments, outdoors, convention booths, ... even with volume at 100%.
Proposed here is a daemon-level output gain stage that boosts audio sent to the robot's speaker. This is intended for repeatable, cross-platform signal-level correction.

## How it works

- At daemon startup the environment variable `REACHY_AUDIO_GAIN_DB` is read and parsed as a floating dB value.
- The value is converted to a linear multiplier (gain) and applied to a GStreamer `volume` element placed before the platform audio sink.
- The gain is applied to **all three** audio output paths:
  1. **Daemon playbin** — used by `play_sound()`, recorded/move audio, emotion sounds (via `_build_audiosink_tee_bin`)
  2. **WebRTC incoming audio** — browser mic played on the robot's speaker (per-peer pipeline)
  3. **Python SDK `push_audio_sample()`** — used by on-robot apps like the conversation app (via `_init_pipeline_playback` in `audio_gstreamer.py`)

## Files

- Implementation: [src/reachy_mini/media/audio_gain.py](src/reachy_mini/media/audio_gain.py)
- Daemon playbin insertion: [src/reachy_mini/media/media_server.py](src/reachy_mini/media/media_server.py) (`_build_audiosink_tee_bin`)
- WebRTC incoming insertion: [src/reachy_mini/media/media_server.py](src/reachy_mini/media/media_server.py) (`_on_consumer_pad_added`)
- Python SDK push path: [src/reachy_mini/media/audio_gstreamer.py](src/reachy_mini/media/audio_gstreamer.py) (`_init_pipeline_playback`)
- Runtime API (REST): [src/reachy_mini/daemon/app/routers/audio_gain.py](src/reachy_mini/daemon/app/routers/audio_gain.py)
- Unit tests: [tests/unit_tests/test_audio_gain.py](tests/unit_tests/test_audio_gain.py)

## Configuration

- Env var: `REACHY_AUDIO_GAIN_DB`
  - Value: floating-point decibels (e.g. `12`, `-6`, `0`)
  - Missing or invalid values default to `0 dB` (no change)

## Runtime control (HTTP)

The daemon exposes a simple REST API to read and update the gain at runtime (no persistence):

- GET `/api/audio/gain`
  - Response: `{ "gain_db": <float>, "gain_linear": <float> }`
- POST `/api/audio/gain` with JSON `{ "gain_db": <float> }`
  - The request is clamped to a safe range and applied immediately to active **daemon** pipelines.
  - Safe range: -20 dB .. +24 dB

> **Important:** The REST API updates daemon-owned pipelines (paths 1 & 2) live. Python apps (path 3) read the gain once at pipeline creation. To change their gain, set `REACHY_AUDIO_GAIN_DB` before the app starts, or restart the app.

Example using `curl` (Linux / macOS / WSL):

```bash
# Read current gain
curl -s http://localhost:8000/api/audio/gain | jq

# Set gain to +12 dB
curl -X POST -H "Content-Type: application/json" \
  -d '{"gain_db": 12.0}' http://localhost:8000/api/audio/gain
```

Example using `curl.exe` on **Windows CMD / PowerShell** (escape inner quotes):

```bat
curl.exe -X POST -H "Content-Type: application/json" -d "{\"gain_db\": 12.0}" http://localhost:8000/api/audio/gain
```

Or using **PowerShell** `Invoke-WebRequest` (no escaping needed):

```powershell
# Read current gain
(Invoke-WebRequest -Uri "http://localhost:8000/api/audio/gain" -UseBasicParsing).Content

# Set gain to +12 dB
Invoke-WebRequest -Uri "http://localhost:8000/api/audio/gain" -Method POST `
  -ContentType "application/json" -Body '{"gain_db": 12.0}' `
  -UseBasicParsing | Select-Object -ExpandProperty Content
```

## Recommended values

- **+12 dB** was found to produce "about twice as loud" perceived volume on the Reachy Mini wireless hardware with typical TTS speech with some quality degradation
- `0 dB` means neutral (no boost).
- Negative values are permitted to attenuate the signal.
- Values up to +24 dB are accepted; above +20 dB clipping becomes likely with hot sources.

## Persistence across restarts

The env var is the persistent mechanism. To make it survive daemon restarts on the wireless robot:

```bash
# Add to the systemd service (one-time setup)
sudo sed -i '/\[Service\]/a Environment=REACHY_AUDIO_GAIN_DB=12' \
  /etc/systemd/system/reachy-mini-daemon.service
sudo systemctl daemon-reload
sudo systemctl restart reachy-mini-daemon.service
```

## Per-app gain (app-level override)

The daemon-level gain applies globally to all audio. If you need a **different gain for one specific app**, you can add gain at the app level:

1. **App-specific env var** — define your own variable (e.g. `MY_APP_AUDIO_GAIN_DB`) in the app code and apply it to the `volume` element after pipeline init:
   ```python
   import os, math
   from reachy_mini.media.audio_gstreamer import AudioGstreamer

   audio = AudioGstreamer(...)
   app_gain_db = float(os.environ.get("MY_APP_AUDIO_GAIN_DB", "0"))
   audio._volume_element.set_property("volume", 10 ** (app_gain_db / 20))
   ```

2. **Scale PCM samples directly** — multiply the audio buffer by a linear factor before calling `push_audio_sample()`. This works without touching GStreamer internals.

> **Note:** App-level gain stacks with the daemon-level gain (they are independent `volume` elements in series). If the daemon is at +12 dB and the app adds +6 dB, total boost is +18 dB — be careful about clipping.

## Limiter (anti-clipping)

A `rglimiter` (brickwall limiter from gst-plugins-good) is placed **after** the `volume` element in every pipeline. It prevents samples from exceeding 0 dBFS after the gain boost, eliminating the harsh distortion that occurs when amplified peaks clip.

Pipeline chain: `... → volume → rglimiter → tee → ...`

The limiter is always active (zero-config). When gain is 0 dB it has no audible effect — it only engages when samples exceed full scale, which can't happen without a positive gain. It is kept unconditionally in the pipeline because:

1. **No cost at unity gain** — `rglimiter` is a passthrough when no sample exceeds 1.0 (negligible CPU, ~microseconds per buffer on CM4).
2. **Runtime safety** — the REST API can raise gain from 0 to +12 dB at any time; having the limiter already linked means no pipeline rebuild is needed.
3. **Simplicity** — a single pipeline topology regardless of config avoids conditional element insertion/removal bugs.

## Risks & Caveats

- At very high gain (+18 dB+) the limiter will engage frequently, which can sound "squashed" or reduce dynamic range. For best quality keep gain at +10 dB or below.
- Gain and device/OS volume are cumulative in perceived loudness.
- The ALSA master volume on the robot is already at 100% — there is no hidden headroom to recover via mixer settings.

## Architecture note: three pipelines

The Reachy Mini has three independent audio output pipelines:

| Pipeline | Built in | Used by |
|---|---|---|
| Daemon playbin (tee bin) | `media_server.py` | `play_sound()`, emotion library, uploaded moves |
| WebRTC incoming (per-peer) | `media_server.py` | Browser mic → robot speaker |
| Python SDK push | `audio_gstreamer.py` | On-robot apps (`push_audio_sample()`) like conversation app |

All three now include a `volume` element reading from `audio_gain.get_output_gain_linear()`.

## Testing

- Unit tests for parsing and conversion live in `tests/unit_tests/test_audio_gain.py`.
- Manual checks:
  - Start the daemon with `REACHY_AUDIO_GAIN_DB=12` and launch the conversation app — speech should be noticeably louder.
  - Play emotions via the desktop app — should also be louder.
  - Test with a sine wave: `gst-launch-1.0 audiotestsrc freq=440 num-buffers=100 ! volume volume=3.98 ! pulsesink` vs without the volume element to verify the pipeline works independently.
  - Verify the REST `GET /api/audio/gain` returns the expected value.

## Developer notes

- The `volume` element is created with the linear multiplier from `audio_gain.get_output_gain_linear()` and stored on `GstMediaServer._gain_elements` (daemon paths) or as `self._volume_element` (SDK path) so runtime updates can be pushed.
- `GstMediaServer.update_output_gain(linear)` updates daemon-owned elements live.
- To persist runtime changes across restarts, set `REACHY_AUDIO_GAIN_DB` in the systemd unit (not implemented as auto-persist in code).