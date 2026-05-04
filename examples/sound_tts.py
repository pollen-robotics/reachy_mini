"""TTS demo with head wobbling.

Sends text to ResembleAI's Chatterbox Multilingual TTS Hugging Face
Space (zero-shot voice cloning, 23 languages), plays the returned
audio on Reachy Mini, and wobbles the head in sync.

Usage::

    uv run python examples/sound_tts.py --text "Hello world"
    uv run python examples/sound_tts.py --text "Bonjour" --lang fr
    uv run python examples/sound_tts.py --text "..." --ref-audio /path/to/voice.wav

Browse the Space: https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS
"""

# START doc_example

import argparse
import os
import time

import gi
from gradio_client import Client, handle_file

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
from gi.repository import Gst, GstPbutils  # noqa: E402

from reachy_mini import ReachyMini  # noqa: E402

HF_SPACE = "ResembleAI/Chatterbox-Multilingual-TTS"
LANGUAGES = [
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
    "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
]
DEFAULT_REF_AUDIO = (
    "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav"
)


def synthesize(text: str, lang: str, ref_audio: str) -> str:
    """Submit *text* to Chatterbox; return a path to a local audio file."""
    if not ref_audio.startswith(("http://", "https://")):
        ref_audio = os.path.expanduser(ref_audio)
    client = Client(HF_SPACE)
    audio_path = client.predict(
        text_input=text,
        language_id=lang,
        audio_prompt_path_input=handle_file(ref_audio),
        api_name="/generate_tts_audio",
    )
    return str(audio_path)


def probe_duration_s(path: str) -> float:
    """Return the media duration of *path* in seconds via GStreamer."""
    Gst.init([])
    disc = GstPbutils.Discoverer.new(10 * Gst.SECOND)
    info = disc.discover_uri(f"file://{path}")
    return float(info.get_duration() / Gst.SECOND)


def main(text: str, lang: str, ref_audio: str) -> None:
    """Synthesize *text*, play it on Reachy Mini with wobbling enabled."""
    print(f"Synthesizing {len(text)} chars ({lang}) with Chatterbox...")
    audio_path = synthesize(text, lang, ref_audio)
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
        description="Chatterbox Multilingual TTS + head wobbler demo.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, I am Reachy Mini. Let me wobble my head while I speak.",
        help="Text to synthesize (max 300 chars per request).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=LANGUAGES,
        help="Language code (ISO 639-1).",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=DEFAULT_REF_AUDIO,
        help="Reference audio (URL or local path) for zero-shot voice cloning.",
    )
    args = parser.parse_args()
    main(
        text=args.text,
        lang=args.lang,
        ref_audio=args.ref_audio,
    )

# END doc_example
