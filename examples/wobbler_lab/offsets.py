"""Export the head offsets a wobbler version produces for a wav, as JSON.

Single wav::

    python examples/wobbler_lab/offsets.py --wav X.wav --version v6 \
        --emotion angry --energy 1.2 --out X.v6.json

Batch (one JSON per wav and per version, ``<out-dir>/<stem>.<version>.json``)::

    python examples/wobbler_lab/offsets.py --wav-dir DIR --out-dir DIR \
        --versions v0,v5,v6 [--scene scene.json]

With ``--scene``, versions that accept a colouring (v6 and later) take
``emotion`` / ``energy`` from the beat whose ``id`` equals the wav stem;
wavs without a beat get neutral at 1.0. Wavs whose stem is listed in
``--exclude`` (default: ``scene_mix``) are skipped in batch mode.

JSON format::

    {"version": "v6", "emotion": "angry", "energy": 1.2,
     "hop_ms": 50, "sample_rate": 16000,
     "t": [...], "pitch": [...], "yaw": [...], "roll": [...],
     "x": [...], "y": [...], "z": [...]}

Angles in radians, translations in millimetres, ``t`` in seconds from the
start of the wav.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from wobbler_lab import simulate as sim_mod  # noqa: E402

DEFAULT_EMOTION = "neutral"
DEFAULT_ENERGY = 1.0
NDIGITS = 5


def load_scene_beats(scene_path: Path | None) -> dict[str, dict[str, Any]]:
    """Map beat id to beat dict. Empty when no scene is given."""
    if scene_path is None:
        return {}
    with open(scene_path, encoding="utf-8") as f:
        scene = json.load(f)
    beats = scene["beats"] if isinstance(scene, dict) and "beats" in scene else scene
    return {str(b["id"]): b for b in beats if isinstance(b, dict) and "id" in b}


def colouring_for(version: str, beat: dict[str, Any] | None, emotion: str | None, energy: float | None) -> tuple[str, float]:
    """Resolve (emotion, energy) for one export. Explicit flags win over the scene."""
    if not sim_mod.supports_emotion(version):
        return DEFAULT_EMOTION, DEFAULT_ENERGY
    emo = emotion
    en = energy
    if beat is not None:
        if emo is None:
            emo = beat.get("emotion")
        if en is None and beat.get("energy") is not None:
            en = float(beat["energy"])
    return (emo or DEFAULT_EMOTION), (DEFAULT_ENERGY if en is None else float(en))


def export(wav: Path, version: str, emotion: str, energy: float, out: Path) -> dict[str, Any]:
    """Run *version* on *wav* and write the offsets JSON to *out*."""
    pcm, sr = sim_mod.load_audio(str(wav))
    sim = sim_mod.run_tapper(version, pcm, sr, emotion=emotion, energy=energy)
    m = sim.motion
    payload = {
        "version": version,
        "emotion": emotion,
        "energy": float(energy),
        "hop_ms": int(sim.hop_ms),
        "sample_rate": int(sr),
        "t": [round(float(v), NDIGITS) for v in sim.motion_time],
        "pitch": [round(float(v), NDIGITS) for v in m[:, 0]],
        "yaw": [round(float(v), NDIGITS) for v in m[:, 1]],
        "roll": [round(float(v), NDIGITS) for v in m[:, 2]],
        "x": [round(float(v), NDIGITS) for v in m[:, 3]],
        "y": [round(float(v), NDIGITS) for v in m[:, 4]],
        "z": [round(float(v), NDIGITS) for v in m[:, 5]],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    return payload


def main() -> None:
    """Export one wav, or every wav of a directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav", type=Path, help="single wav to export")
    parser.add_argument("--out", type=Path, help="output JSON for --wav")
    parser.add_argument("--version", default="v6", choices=sorted(sim_mod.VERSIONS),
                        help="wobbler version for --wav (default v6)")
    parser.add_argument("--emotion", default=None,
                        help="colouring for versions that accept it (overrides the scene)")
    parser.add_argument("--energy", type=float, default=None,
                        help="amplitude scale for versions that accept it (overrides the scene)")
    parser.add_argument("--wav-dir", type=Path, help="batch: directory of wavs")
    parser.add_argument("--out-dir", type=Path, help="batch: output directory")
    parser.add_argument("--versions", default="v0,v5,v6",
                        help="batch: comma-separated versions (default v0,v5,v6)")
    parser.add_argument("--scene", type=Path, default=None,
                        help="batch: scene.json whose beats carry emotion / energy per id")
    parser.add_argument("--exclude", default="scene_mix",
                        help="batch: comma-separated wav stems to skip (default scene_mix)")
    args = parser.parse_args()

    if args.wav is not None:
        if args.out is None:
            parser.error("--wav needs --out")
        beats = load_scene_beats(args.scene)
        beat = beats.get(args.wav.stem)
        emotion, energy = colouring_for(args.version, beat, args.emotion, args.energy)
        p = export(args.wav, args.version, emotion, energy, args.out)
        print(f"{args.out}  {args.version} {emotion} {energy:.2f}  T={len(p['t'])} hop={p['hop_ms']} ms")
        return

    if args.wav_dir is None or args.out_dir is None:
        parser.error("either --wav and --out, or --wav-dir and --out-dir")

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    unknown = [v for v in versions if v not in sim_mod.VERSIONS]
    if unknown:
        parser.error(f"unknown versions: {unknown}")
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    beats = load_scene_beats(args.scene)

    wavs = sorted(p for p in args.wav_dir.glob("*.wav") if p.stem not in exclude)
    if not wavs:
        parser.error(f"no wav in {args.wav_dir}")
    for wav in wavs:
        beat = beats.get(wav.stem)
        for version in versions:
            emotion, energy = colouring_for(version, beat, args.emotion, args.energy)
            out = args.out_dir / f"{wav.stem}.{version}.json"
            p = export(wav, version, emotion, energy, out)
            print(f"{out.name:32s} {version} {emotion:9s} {energy:.2f}  T={len(p['t'])}")


if __name__ == "__main__":
    main()
