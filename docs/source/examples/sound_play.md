# Sound Playback

This example demonstrates how to play audio through Reachy Mini's speaker by loading a WAV file and pushing audio samples to the audio device. This shows the low-level audio playback API that can be used with any audio source, such as text-to-speech engines or microphone input.

**How it works:**
1. Loads a WAV file using `soundfile`
2. Resamples the audio to match the robot's output sample rate if needed
3. Converts stereo to mono if necessary
4. Pushes audio samples in chunks to the speaker using `push_audio_sample()`

**Features:**
- Automatic sample rate conversion
- Stereo to mono conversion
- Chunked audio streaming (1024 samples per chunk)
- Support for different media backends


**Usage:**
```bash
python sound_play.py --backend [default_no_video|gstreamer_no_video|webrtc]
```

The example uses the built-in `wake_up.wav` file from the assets folder.

<literalinclude>
{"path": "../../../examples/sound_play.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
