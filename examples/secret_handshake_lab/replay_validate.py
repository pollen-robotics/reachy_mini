"""Replay the recorded secret-handshake moves through the detector + machine.

Offline regression for the GEOMETRIC collision law (Remi's measured
definition, see collision.py). It downloads the recordings and asserts:

  - the detector finds EXACTLY 3 collisions on every default*.json, at the
    audible knock times (the approach does not count: condition 2 filters it)
  - the full state machine (pose gate included) reaches PRIMED on the
    default recordings whose head pose arms, and never fires an action on them

The slide recordings (collision-definition*.json) are printed for judgment,
not asserted: a slow slide passes through the collision region repeatedly
and lingers in it, so it can prime and even fire the hold action. That is
inherent to a purely geometric law and must be judged live (the two-round
structure plus the sleep-pose gate are the real protection).

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
    onsets = []
    for t, frame in zip(move["time"], move["set_target_data"]):
        ant0, ant1 = frame["antennas"]
        if det.update(t - t0, ant0, ant1):
            onsets.append(round(t - t0, 2))
    return onsets


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

    print("Geometric collision detector on the recordings:\n")
    for name, move in moves.items():
        onsets = count_collisions(move)
        if name in DEFAULTS and len(onsets) != 3:
            failures.append(f"{name}: {len(onsets)} collisions (want exactly 3)")
        print(f"  {name:28s} collisions={len(onsets)} at {onsets}")

    print(
        "\nFull state machine replay (pose gate from recorded head pose).\n"
        "default*.json ARE the round-1 gesture: PRIMED must fire when the\n"
        "head pose arms, and no action may fire on them:\n"
    )
    for name, move in moves.items():
        events = replay_machine(move)
        kinds = [e for _, e in events]
        if name in DEFAULTS:
            if any(k.startswith("action") for k in kinds):
                failures.append(f"{name}: fired {kinds} (no action may fire)")
            if "armed" in kinds and "primed" not in kinds:
                failures.append(f"{name}: armed but never primed on the gesture")
        note = ""
        if name in DEFAULTS and "armed" not in kinds:
            note = "  (head pose never armed: known outlier, see README)"
        if name not in DEFAULTS and any(k.startswith("action") for k in kinds):
            note = "  <-- slide play completed a handshake, judge live"
        print(f"  {name:28s} events={events}{note}")

    if failures:
        print("\nREGRESSION FAILURES:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print("\nAll hard regressions pass (defaults: 3 collisions + primed).")


if __name__ == "__main__":
    main()
