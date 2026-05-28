# Sound Recording

This example demonstrates how to record audio from Reachy Mini's microphone array and save it to a WAV file. The script records for 5 seconds and saves the audio to `recorded_audio.wav`.

**How it works:**
1. Starts audio recording using `start_recording()`
2. Continuously retrieves audio samples using `get_audio_sample()`
3. Collects samples until the desired duration is reached
4. Stops recording and concatenates all samples
5. Saves the audio data to a WAV file using `soundfile`

**Features:**
- Configurable recording duration (default: 5 seconds)
- Automatic sample rate detection
- Timeout protection to prevent infinite loops
- Support for different media backends

**Usage:**
```bash
python sound_record.py --backend [default|local|webrtc]
```

The recorded audio will be saved to `recorded_audio.wav` in the current directory.

<literalinclude>
{"path": "../../../examples/sound_record.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
