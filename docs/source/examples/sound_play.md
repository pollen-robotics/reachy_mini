# Sound Playback

This example demonstrates two ways to play audio through Reachy Mini's speaker:
- **`--wav`**: Play a WAV file using the `play_sound()` API.
- **`--live`**: Push a continuous sine tone using the low-level `push_audio_sample()` API, useful for real-time audio sources such as text-to-speech engines or microphone input.

**Usage:**
```bash
# Play a wav file
python sound_play.py --wav /path/to/file.wav --backend webrtc

# Push a continuous sine tone (Ctrl+C to stop)
python sound_play.py --live --backend webrtc --tone-hz 440
```

**Options:**
- `--wav <path>`: Path to a WAV file to play.
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
