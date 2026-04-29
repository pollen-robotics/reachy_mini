# Sound TTS (with head wobbling)

This example synthesises speech from text via Alibaba's
[Qwen3-TTS](https://huggingface.co/spaces/Qwen/Qwen3-TTS) Hugging Face
Space, plays the returned audio on Reachy Mini, and wobbles the head in
sync with the speech.

The "voice design" endpoint accepts a free-form `voice_description`
prompt, so you can style the voice by just describing it.

**Usage:**

```bash
# Default English prompt
uv run python examples/sound_tts.py --text "Hello, I can wobble my head!"

# Describe a voice style
uv run python examples/sound_tts.py \
    --text "No way, that's impossible!" \
    --voice-description "Speak with panic creeping into your voice."
```

**Options:**

- `--text <str>`: Text to synthesize.
- `--lang <code>`: Language (`Auto`, `English`, `French`, `Chinese`,
  `Japanese`, `Korean`, `German`, `Spanish`, `Portuguese`, `Russian`).
  `Auto` lets the model detect it.
- `--voice-description <str>`: Natural-language description of the
  voice style (tone, pace, emotion).

<literalinclude>
{"path": "../../../examples/sound_tts.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
