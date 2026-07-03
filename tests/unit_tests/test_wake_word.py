"""Unit tests for the wake-word detector's chunking / gating / cooldown logic.

These drive ``WakeWordDetector._process`` directly with a fake interpreter, so
they need neither GStreamer audio nor the nanowakeword model.
"""

from types import SimpleNamespace

import numpy as np

from reachy_mini.media.wake_word import CHUNK, WakeWordDetector


class FakeInterpreter:
    """Scores a frame hot iff its first sample is non-zero."""

    def __init__(self) -> None:
        self.reset_calls = 0
        self.predict_calls = 0

    def predict(self, chunk: np.ndarray) -> SimpleNamespace:
        self.predict_calls += 1
        return SimpleNamespace(score=0.99 if chunk[0] != 0 else 0.0)

    def reset(self) -> None:
        self.reset_calls += 1


def _make_detector() -> tuple[WakeWordDetector, list[int], FakeInterpreter]:
    hits: list[int] = []
    det = WakeWordDetector(on_detection=lambda: hits.append(1), threshold=0.85)
    interp = FakeInterpreter()
    det._interpreter = interp
    return det, hits, interp


HOT = np.ones(CHUNK, dtype=np.int16)
COLD = np.zeros(CHUNK, dtype=np.int16)


def test_no_fire_when_not_listening() -> None:
    det, hits, _ = _make_detector()
    det._listening = False
    det._process(HOT)
    assert hits == []


def test_fires_once_when_listening() -> None:
    det, hits, interp = _make_detector()
    det._listening = True
    det._process(HOT)
    assert hits == [1]
    assert interp.reset_calls == 1  # reset after firing


def test_cold_audio_never_fires() -> None:
    det, hits, _ = _make_detector()
    det._listening = True
    det._process(COLD)
    assert hits == []


def test_cooldown_suppresses_second_detection() -> None:
    det, hits, _ = _make_detector()
    det._listening = True
    det._process(HOT)
    det._process(HOT)  # within cooldown_s -> ignored
    assert hits == [1]


def test_buffering_reassembles_across_small_samples() -> None:
    det, hits, _ = _make_detector()
    det._listening = True
    # Feed one hot frame split into sub-CHUNK pieces; detection should still
    # fire exactly once once a full CHUNK has accumulated.
    for i in range(0, CHUNK, 256):
        det._process(HOT[i : i + 256])
    assert hits == [1]


def test_partial_buffer_does_not_predict() -> None:
    det, hits, interp = _make_detector()
    det._listening = True
    det._process(HOT[: CHUNK - 1])  # one sample short of a frame
    assert interp.predict_calls == 0
    assert hits == []
