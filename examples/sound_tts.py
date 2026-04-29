"""TTS demo with head wobbling — cached multi-speech playback.

Synthesizes three longer speeches (Kennedy, MLK, De Gaulle) via
Alibaba's Qwen3-TTS Hugging Face Space, caches the resulting WAV
files locally, then plays each clip on Reachy Mini with the head
wobbler running. Subsequent runs reuse the cached WAVs so you can
iterate on wobbler modes without re-synthesizing.

Usage::

    uv run python examples/sound_tts.py
    uv run python examples/sound_tts.py --wobbler-version v2
    uv run python examples/sound_tts.py --regenerate    # force resynthesis

Browse the Space: https://huggingface.co/spaces/Qwen/Qwen3-TTS
"""

# START doc_example

import argparse
import os
import shutil
import time
from pathlib import Path

import gi
from gradio_client import Client

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
from gi.repository import Gst, GstPbutils  # noqa: E402

from reachy_mini import ReachyMini  # noqa: E402

HF_SPACE = "Qwen/Qwen3-TTS"
CACHE_DIR = Path(__file__).parent / ".tts_cache"

SPEECHES = [
    {
        "key": "kennedy",
        "text": (
            "And so, my fellow Americans: ask not what your country can do for you "
            "— ask what you can do for your country. My fellow citizens of the world: "
            "ask not what America will do for you, but what together we can do for the "
            "freedom of man."
        ),
        "language": "English",
        "voice_description": "Speak in a confident, statesmanlike tone with a measured, deliberate cadence.",
    },
    {
        "key": "mlk",
        "text": (
            "I have a dream that one day this nation will rise up and live out the true "
            "meaning of its creed: we hold these truths to be self-evident, that all men "
            "are created equal. I have a dream that my four little children will one day "
            "live in a nation where they will not be judged by the color of their skin, "
            "but by the content of their character."
        ),
        "language": "English",
        "voice_description": "Speak with warm conviction, rising emotion, and a preacher's rhythm.",
    },
    {
        "key": "degaulle",
        "text": (
            "Moi, Général de Gaulle, actuellement à Londres, j'invite les officiers et "
            "les soldats français qui se trouvent en territoire britannique à se mettre "
            "en rapport avec moi. Quoi qu'il arrive, la flamme de la résistance française "
            "ne doit pas s'éteindre et ne s'éteindra pas."
        ),
        "language": "French",
        "voice_description": "Speak as a seventy-year-old French general issuing a command — deep baritone, serious, unhurried, no warmth.",
    },
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


def get_or_generate(speech: dict, regenerate: bool) -> Path:
    """Return the cached WAV path for *speech*, synthesizing on cache miss."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{speech['key']}.wav"
    if cached.exists() and not regenerate:
        print(f"[{speech['key']}] cache hit → {cached.name}")
        return cached
    print(f"[{speech['key']}] synthesizing {len(speech['text'])} chars ({speech['language']})...")
    src = synthesize(speech["text"], speech["language"], speech["voice_description"])
    shutil.copy(src, cached)
    print(f"[{speech['key']}] saved → {cached.name}")
    return cached


def main(wobbler_version: str, regenerate: bool, gap_s: float) -> None:
    """Synthesize (or load) all speeches and play them with wobbling enabled."""
    os.environ["WOBBLER_VERSION"] = wobbler_version

    clips = []
    for speech in SPEECHES:
        path = get_or_generate(speech, regenerate)
        duration = probe_duration_s(str(path))
        clips.append((speech["key"], path, duration))
        print(f"  duration: {duration:.1f}s")

    with ReachyMini(log_level="INFO") as mini:
        mini.enable_wobbling()
        for key, path, duration in clips:
            print(f"Playing [{key}] ({duration:.1f}s) with wobbler={wobbler_version}")
            mini.media.play_sound(str(path))
            time.sleep(duration + gap_s)
        mini.disable_wobbling()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS + head wobbler demo — cached multi-speech playback.",
    )
    parser.add_argument(
        "--wobbler-version",
        type=str,
        default="v0",
        choices=["v0", "v1", "v2", "v3", "v4"],
        help="Speech tapper version: v0=original, v1=direct envelope, v2=multi-band, v3=onset impulse, v4=strict gate + nucleus tracking.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force re-synthesis even if cached WAVs exist (delete a single cache "
        "file under examples/.tts_cache/ to regenerate just that clip).",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.8,
        help="Pause in seconds between clips.",
    )
    args = parser.parse_args()
    main(
        wobbler_version=args.wobbler_version,
        regenerate=args.regenerate,
        gap_s=args.gap,
    )

# END doc_example
