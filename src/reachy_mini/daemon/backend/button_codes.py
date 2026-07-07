"""Torque-ON antenna-button code state machine.

Pure, dependency-free, fast: runs every tick on the 50 Hz control loop. It
consumes the Press edge events from `antenna_buttons.AntennaButtonDetector`
and returns a `CodeEvent` when a full coded sequence completes; the caller
decides what each event does (start WiFi provisioning, disable torque, play
an emotion).

THE THREE CODES (design spec 2026-07-07, section 4)

  WiFi        BOTH antennas external together, then BOTH internal together
  Torque off  3x left-external, then 2x right-external
  Emotion     left-ext, right-ext, left-ext, right-ext (alternating)

DESIGN

Presses are grouped into "moves". Presses landing within a small
simultaneity window collapse into ONE move: two antennas pressed the same
direction together become a symmetric move (used only by the WiFi code); a
lone press becomes a single-antenna move (used by the torque-off and emotion
codes). Each completed move is appended to a rolling buffer and matched
against the codes:

  - exact match of the whole buffer against a code -> fire that event, reset;
  - otherwise keep the buffer only while it is a PREFIX of some code, sliding
    off leading moves that break every prefix (so a stray press before a
    valid sequence does not block it).

Safety comes from external-only single-press codes, exact sequences, the
<1 s gap reset, a release between counted presses (enforced upstream by the
detector's latch), and the torque-ON gate.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .antenna_buttons import Direction, Press

# A "move" is one of:
#   ("single", antenna_index, Direction)   -- one antenna pressed
#   ("sym", Direction)                      -- both antennas pressed together
#   ("invalid",)                            -- an ambiguous group (never matches)
_Move = tuple


class CodeEvent(enum.Enum):
    """A completed button code; the caller runs the matching action."""

    WIFI = "wifi"  # both external, then both internal
    TORQUE_OFF = "torque_off"  # 3x left-external, then 2x right-external
    EMOTION = "emotion"  # left, right, left, right (external, alternating)


@dataclass(frozen=True)
class ButtonCodeConfig:
    """Tunables for the code matcher."""

    left_index: int = 0  # collision-code convention ([0]=left); confirm on robot
    right_index: int = 1
    max_gap_s: float = 1.0  # >1 s between moves resets the sequence
    simultaneity_window_s: float = 0.08  # both antennas within this = one move


class ButtonCodeMachine:
    """Feed one tick of press events; get a CodeEvent when a code completes.

        event = machine.update(t, presses, torque_on)

    t          monotonic seconds (any origin, must only increase)
    presses    the list AntennaButtonDetector.update() returned this tick
    torque_on  True when motors are torqued on (the codes are armed only then)
    """

    def __init__(self, config: ButtonCodeConfig | None = None) -> None:
        """Build the matcher and the code table from the left/right mapping."""
        self.cfg = config or ButtonCodeConfig()
        left, right = self.cfg.left_index, self.cfg.right_index
        ext, internal = Direction.EXTERNAL, Direction.INTERNAL
        # (event, sequence of moves). No code is a prefix of another.
        self._codes: list[tuple[CodeEvent, list[_Move]]] = [
            (CodeEvent.WIFI, [("sym", ext), ("sym", internal)]),
            (
                CodeEvent.TORQUE_OFF,
                [("single", left, ext)] * 3 + [("single", right, ext)] * 2,
            ),
            (
                CodeEvent.EMOTION,
                [
                    ("single", left, ext),
                    ("single", right, ext),
                    ("single", left, ext),
                    ("single", right, ext),
                ],
            ),
        ]
        self._groups: list[_Move] = []
        self._pending: list[Press] = []
        self._pending_open_t: float = 0.0
        self._last_press_t: float | None = None

    def reset(self) -> None:
        """Forget all in-progress sequence and pending-move state."""
        self._groups = []
        self._pending = []
        self._last_press_t = None

    def update(
        self, t: float, presses: list[Press], torque_on: bool
    ) -> CodeEvent | None:
        """Advance one tick; return a CodeEvent when a full code completes."""
        if not torque_on:
            self.reset()
            return None

        # A >max_gap silence since the last press abandons the sequence.
        if (
            self._last_press_t is not None
            and (self._groups or self._pending)
            and t - self._last_press_t > self.cfg.max_gap_s
        ):
            self._clear_sequence()

        if presses:
            if not self._pending:
                self._pending_open_t = t
            self._pending.extend(presses)
            self._last_press_t = t

        # Close the pending move once the simultaneity window has elapsed, so
        # a companion press on the other antenna has had time to arrive.
        if self._pending and t - self._pending_open_t >= self.cfg.simultaneity_window_s:
            move = _form_move(self._pending)
            self._pending = []
            return self._append_and_match(move)
        return None

    def _append_and_match(self, move: _Move) -> CodeEvent | None:
        self._groups.append(move)
        while self._groups:
            for event, code in self._codes:
                if self._groups == code:
                    self._clear_sequence()
                    return event
            if self._is_prefix_of_any(self._groups):
                break
            self._groups.pop(0)  # slide off a leading move that breaks all codes
        return None

    def _is_prefix_of_any(self, groups: list[_Move]) -> bool:
        n = len(groups)
        return any(code[:n] == groups for _, code in self._codes)

    def _clear_sequence(self) -> None:
        self._groups = []
        self._pending = []


def _form_move(pending: list[Press]) -> _Move:
    """Collapse the presses collected in one window into a single move."""
    if len(pending) == 1:
        p = pending[0]
        return ("single", p.antenna, p.direction)
    antennas = {p.antenna for p in pending}
    directions = {p.direction for p in pending}
    if len(antennas) == 2 and len(directions) == 1:
        return ("sym", pending[0].direction)
    return ("invalid",)
