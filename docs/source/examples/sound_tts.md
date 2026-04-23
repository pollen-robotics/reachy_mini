# Sound TTS (with head wobbling)

This example synthesises speech from text via Alibaba's
[Qwen3-TTS](https://huggingface.co/spaces/Qwen/Qwen3-TTS) Hugging Face
Space, plays the returned audio on Reachy Mini, and wobbles the head in
sync with the speech.

The "voice design" endpoint accepts a free-form `voice_description`
prompt, so you can style the voice by just describing it.

**Usage:**

```bash
# Default English prompt, wobbler v0
uv run python examples/sound_tts.py --text "Hello, I can wobble my head!"

# Describe a voice style and pick a wobbler variant
uv run python examples/sound_tts.py \
    --text "No way, that's impossible!" \
    --voice-description "Speak with panic creeping into your voice." \
    --wobbler-version v2
```

**Options:**

- `--text <str>`: Text to synthesize.
- `--lang <code>`: Language (`Auto`, `English`, `French`, `Chinese`,
  `Japanese`, `Korean`, `German`, `Spanish`, `Portuguese`, `Russian`).
  `Auto` lets the model detect it.
- `--voice-description <str>`: Natural-language description of the
  voice style (tone, pace, emotion).
- `--wobbler-version {v0,v1,v2,v3}`: Speech-tapper variant —
  `v0`=original, `v1`=direct envelope, `v2`=multi-band, `v3`=onset impulse.

<literalinclude>
{"path": "../../../examples/sound_tts.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
