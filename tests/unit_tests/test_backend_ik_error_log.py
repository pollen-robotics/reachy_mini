"""The control loops log IK errors at most once every 0.5 s."""

from typing import Any

import pytest

from reachy_mini.daemon.backend import abstract


def test_ik_errors_are_logged_at_most_every_half_second(
    sim_backend: Any, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First error logged, repeats within 0.5 s dropped, the next one logged again."""
    clock = [100.0]
    monkeypatch.setattr(abstract.time, "monotonic", lambda: clock[0])

    with caplog.at_level("WARNING"):
        for t in (100.0, 100.2, 100.49, 100.5, 100.6):
            clock[0] = t
            sim_backend._log_ik_error(ValueError(f"at {t}"))

    assert [r.getMessage() for r in caplog.records] == [
        "IK error: at 100.0",
        "IK error: at 100.5",
    ]
