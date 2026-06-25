# Sound Playback

This example demonstrates two ways to play audio through Reachy Mini's speaker:
- **`--file`**: Play a sound file (WAV, OGG, MP3, FLAC, ...) using the `play_sound()` API.
- **`--live`**: Push a continuous sine tone using the low-level `push_audio_sample()` API, useful for real-time audio sources such as text-to-speech engines or microphone input.

**Usage:**
```bash
# Play a sound file
python sound_play.py --file /path/to/file.ogg --backend webrtc

# Push a continuous sine tone (Ctrl+C to stop)
python sound_play.py --live --backend webrtc --tone-hz 440
```

**Options:**
- `--file <path>`: Path to a sound file to play (WAV, OGG, MP3, FLAC, ...).
- `--live`: Push a continuous sine tone.
- `--tone-hz <freq>`: Sine wave frequency in Hz (`--live` mode only, default: 440).
- `--backend`: Media backend to use (`default`, `local`, or `webrtc`).

<literalinclude>
{"path": "../../../examples/sound_play.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
