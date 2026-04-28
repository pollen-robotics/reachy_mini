"""TTS demo with head wobbling.

Sends text to Alibaba's Qwen3-TTS Hugging Face Space using the
"voice design" endpoint (style the voice with a natural-language
prompt), plays the returned audio on Reachy Mini, and wobbles the
head in sync.

Usage::

    uv run python examples/sound_tts.py --text "Hello world"
    uv run python examples/sound_tts.py --text "..." --voice-description "Speak with panic creeping in." --wobbler-version v2

Browse the Space: https://huggingface.co/spaces/Qwen/Qwen3-TTS
"""

# START doc_example

import argparse
import os
import time

import gi
from gradio_client import Client

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
from gi.repository import Gst, GstPbutils  # noqa: E402

from reachy_mini import ReachyMini  # noqa: E402

HF_SPACE = "Qwen/Qwen3-TTS"

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "German", "Spanish", "Portuguese", "Russian",
]


def synthesize(text: str, language: str, voice_description: str) -> str:
    """Submit *text* to the Qwen3-TTS Space; return a path to a local audio file."""
    client = Client(HF_SPACE)
    audio_path, _status = client.predict(
        text=text,
        language=language,
        voice_description=voice_description,
        api_name="/generate_voice_design",
    )
    return str(audio_path)


def probe_duration_s(path: str) -> float:
    """Return the media duration of *path* in seconds via GStreamer."""
    Gst.init([])
    disc = GstPbutils.Discoverer.new(10 * Gst.SECOND)
    info = disc.discover_uri(f"file://{path}")
    return float(info.get_duration() / Gst.SECOND)


def main(text: str, language: str, voice_description: str, wobbler_version: str) -> None:
    """Synthesize *text*, play it on Reachy Mini with wobbling enabled."""
    os.environ["WOBBLER_VERSION"] = wobbler_version

    print(f"Synthesizing {len(text)} chars ({language}) with Qwen3-TTS...")
    audio_path = synthesize(text, language, voice_description)
    duration = probe_duration_s(audio_path)
    print(f"Got {audio_path} ({duration:.1f}s)")

    with ReachyMini(log_level="INFO") as mini:
        mini.enable_wobbling()
        mini.media.play_sound(audio_path)
        time.sleep(duration + 0.5)
        mini.disable_wobbling()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS + head wobbler demo — voice-design endpoint.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, I am Reachy Mini. Let me wobble my head while I speak.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="Auto",
        choices=LANGUAGES,
        help="Language — 'Auto' lets the model detect it.",
    )
    parser.add_argument(
        "--voice-description",
        type=str,
        default="Speak in a warm, friendly tone.",
        help="Natural-language description shaping the voice style.",
    )
    parser.add_argument(
        "--wobbler-version",
        type=str,
        default="v0",
        choices=["v0", "v1", "v2", "v3"],
        help="Speech tapper version: v0=original, v1=direct envelope, v2=multi-band, v3=onset impulse.",
    )
    args = parser.parse_args()
    main(
        text=args.text,
        language=args.lang,
        voice_description=args.voice_description,
        wobbler_version=args.wobbler_version,
    )

# END doc_example
