"""Replay the recorded secret-handshake moves through CollisionDetector.

This is the offline calibration / regression harness. It downloads the
recordings the human made with Marionette, runs the pure detector over them,
and prints per-file results so thresholds can be tuned on real data instead of
guesses.

Dataset: https://huggingface.co/datasets/RemiFabre/secret-handshake
  - default*.json           : human's "valid first step" recordings
  - collision-definition*.json : antennas swept through the full contact range

Run:
    python examples/secret_handshake_lab/replay_validate.py
"""

from __future__ import annotations

import json
import os

from collision import CollisionConfig, CollisionDetector
from handshake import HandshakeConfig, HandshakeStateMachine
from pose_gate import head_in_sleep_pose

REPO_ID = "RemiFabre/secret-handshake"


def load_moves() -> dict[str, dict]:
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
    data_dir = os.path.join(root, "data")
    moves = {}
    for name in sorted(os.listdir(data_dir)):
        if name.endswith(".json"):
            with open(os.path.join(data_dir, name)) as f:
                moves[name] = json.load(f)
    return moves


def replay(move: dict, cfg: CollisionConfig) -> dict:
    det = CollisionDetector(cfg)
    t = [row for row in move["time"]]
    t0 = t[0]
    onsets = []
    contact_ticks = 0
    for i, frame in enumerate(move["set_target_data"]):
        ant0, ant1 = frame["antennas"]
        if det.update(ant0, ant1):
            onsets.append(round(t[i] - t0, 2))
        if det.in_contact:
            contact_ticks += 1
    dt = (t[-1] - t[0]) / max(1, len(t) - 1)
    return {"onsets": onsets, "contact_s": round(contact_ticks * dt, 2)}


def replay_machine(move: dict, counter: str = "edge") -> list[tuple[float, str]]:
    """Run the FULL handshake state machine over a recording."""
    machine = HandshakeStateMachine(HandshakeConfig(counter=counter))
    t0 = move["time"][0]
    events = []
    for t, frame in zip(move["time"], move["set_target_data"]):
        ant0, ant1 = frame["antennas"]
        e = machine.update(
            t - t0,
            ant0,
            ant1,
            torque_off=True,
            head_in_sleep_pose=head_in_sleep_pose(frame["head"]),
        )
        if e is not None:
            events.append((round(t - t0, 2), e.value))
    return events


def main() -> None:
    cfg = CollisionConfig()
    print(f"CollisionConfig(t_on={cfg.t_on}, t_off={cfg.t_off})  diff = ant0 - ant1\n")
    moves = load_moves()
    for name, move in moves.items():
        r = replay(move, cfg)
        print(f"  {name:28s} onsets={len(r['onsets'])} at {r['onsets']}  contact={r['contact_s']}s")

    print(
        "\nFull state machine, counter='edge' (regression: at most 1 onset per\n"
        "file, so it must never prime, let alone act):\n"
    )
    bad = False
    for name, move in moves.items():
        events = replay_machine(move, counter="edge")
        kinds = {e for _, e in events}
        fired = kinds - {"armed"}
        if fired:
            bad = True
        verdict = "FALSE POSITIVE!" if fired else "ok"
        print(f"  {name:28s} events={events}  {verdict}")

    print(
        "\nFull state machine, counter='knock' (informative: default*.json ARE\n"
        "the intended round-1 gesture, so PRIMED there means the knock counter\n"
        "hears the 3 knocks that edge counting misses):\n"
    )
    for name, move in moves.items():
        events = replay_machine(move, counter="knock")
        print(f"  {name:28s} events={events}")

    if bad:
        raise SystemExit("false positive on recorded data, tune thresholds")
    print(
        "\nNote: in the recorded joint-angle data the antennas stay together\n"
        "once brought into contact, so edge-counting sees 1 onset per file even\n"
        "though 3 knocks are audible. These recordings validate the CONTACT\n"
        "scalar and the rest/sleep pose. The 3-tap RHYTHM must be confirmed\n"
        "live with live_contact_probe.py during a real hit-release-hit-release-\n"
        "hit gesture (tune t_on/t_off, or switch to the knock-peak counter if\n"
        "the natural gesture keeps light contact between knocks). See the spec."
    )


if __name__ == "__main__":
    main()
