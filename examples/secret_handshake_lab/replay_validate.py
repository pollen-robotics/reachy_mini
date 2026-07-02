"""Replay the recorded secret-handshake moves through the detector + machine.

Offline regression for the coupled-motion collision law. It downloads the
recordings the human made with Marionette and asserts:

  - the collision detector finds the knocks on every default*.json (the
    intended gesture: bring-together clack + 3 taps => 3 to 7 collisions)
  - it finds ZERO collisions in the 30 s of collision-definition*.json
    (antennas touching and sliding the whole time: coupling, but no knocks)
  - the FULL state machine (pose gate included) reaches PRIMED on the
    default recordings whose head pose arms, and never fires an action

Dataset: https://huggingface.co/datasets/RemiFabre/secret-handshake

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

DEFAULTS = ["default.json", "default2.json", "default3.json", "default4.json"]
SLIDES = ["collision-definition.json", "collision-definition2.json"]


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


def count_collisions(move: dict) -> list[float]:
    det = CollisionDetector(CollisionConfig())
    t0 = move["time"][0]
    knocks = []
    for t, frame in zip(move["time"], move["set_target_data"]):
        ant0, ant1 = frame["antennas"]
        if det.update(t - t0, ant0, ant1):
            knocks.append(round(t - t0, 2))
    return knocks


def replay_machine(move: dict) -> list[tuple[float, str]]:
    machine = HandshakeStateMachine(HandshakeConfig())
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
    moves = load_moves()
    failures = []

    print("Collision detector (coupled-motion law) on the recordings:\n")
    for name, move in moves.items():
        knocks = count_collisions(move)
        if name in SLIDES and knocks:
            failures.append(f"{name}: {len(knocks)} collisions on slide data (want 0)")
        if name in DEFAULTS and not 3 <= len(knocks) <= 7:
            failures.append(f"{name}: {len(knocks)} collisions (want 3-7)")
        print(f"  {name:28s} collisions={len(knocks)} at {knocks}")

    print(
        "\nFull state machine replay (pose gate from recorded head pose).\n"
        "default*.json ARE the round-1 gesture: PRIMED must fire when the\n"
        "head pose arms. No recording contains a full two-round handshake,\n"
        "so no ACTION event may ever fire:\n"
    )
    for name, move in moves.items():
        events = replay_machine(move)
        kinds = [e for _, e in events]
        if any(k.startswith("action") for k in kinds):
            failures.append(f"{name}: fired {kinds} (no action may fire)")
        if name in SLIDES and "primed" in kinds:
            failures.append(f"{name}: primed on slide data")
        if name in DEFAULTS and "armed" in kinds and "primed" not in kinds:
            failures.append(f"{name}: armed but never primed on the gesture")
        note = ""
        if name in DEFAULTS and "armed" not in kinds:
            note = "  (head pose never armed: known outlier, see README)"
        print(f"  {name:28s} events={events}{note}")

    if failures:
        print("\nREGRESSION FAILURES:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print("\nAll regressions pass.")


if __name__ == "__main__":
    main()
