# Sound TTS (with head wobbling)

This example synthesises speech from text via ResembleAI's
[Chatterbox Multilingual TTS](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS)
Hugging Face Space, plays the returned audio on Reachy Mini, and
wobbles the head in sync with the speech.

Chatterbox supports zero-shot voice cloning: pass a short reference
audio file and the synthesis matches that voice. 23 languages are
supported.

**Usage:**

```bash
# Default English voice
uv run python examples/sound_tts.py --text "Hello, I can wobble my head!"

# Different language
uv run python examples/sound_tts.py --text "Bonjour, je suis Reachy Mini" --lang fr

# Clone a voice from a local sample
uv run python examples/sound_tts.py \
    --text "Hello world" \
    --ref-audio ~/Downloads/my_voice.wav
```

**Options:**

- `--text <str>`: Text to synthesize (max 300 chars per request).
- `--lang <code>`: ISO 639-1 language code. Supported: `ar`, `da`,
  `de`, `el`, `en`, `es`, `fi`, `fr`, `he`, `hi`, `it`, `ja`, `ko`,
  `ms`, `nl`, `no`, `pl`, `pt`, `ru`, `sv`, `sw`, `tr`, `zh`.
- `--ref-audio <path|url>`: Reference audio for zero-shot voice
  cloning. Local paths and URLs both work; defaults to a Gradio
  sample voice.

Synthesis runs on the Space's shared GPU and typically takes
60–90 s per sentence.

<literalinclude>
{"path": "../../../examples/sound_tts.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
