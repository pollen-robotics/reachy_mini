"""Cover the shared recorded-move library cache.

Exercises ``motion/recorded_move.py``'s ``get_recorded_moves``: the memoization
and the per-dataset single-flight lock that stop a cold HF cache from
rebuilding — and re-downloading — a library on every playback request. No
hardware and no network: ``snapshot_download`` is stubbed out.
"""

import json
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pytest

import reachy_mini.motion.recorded_move as recorded_move
from reachy_mini.motion.recorded_move import get_recorded_moves

DATASET = "pollen-robotics/some-library"


@pytest.fixture(autouse=True)
def clear_library_cache() -> Iterator[None]:
    """Isolate the module-level caches so tests don't leak into each other."""
    recorded_move._libraries.clear()
    recorded_move._library_locks.clear()
    yield
    recorded_move._libraries.clear()
    recorded_move._library_locks.clear()


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """A minimal on-disk dataset: one two-frame move, no sidecar sound."""
    frame: dict[str, Any] = {
        "head": np.eye(4).tolist(),
        "antennas": [0.0, 0.0],
        "body_yaw": 0.0,
    }
    (tmp_path / "hello.json").write_text(
        json.dumps(
            {
                "description": "test move",
                "time": [0.0, 0.1],
                "set_target_data": [frame, frame],
            }
        )
    )
    return tmp_path


def _stub_snapshot_download(
    monkeypatch: pytest.MonkeyPatch,
    dataset_dir: Path,
    on_call: Callable[[], None] | None = None,
) -> list[str]:
    """Replace snapshot_download with a local stub, returning its call log."""
    calls: list[str] = []

    def fake_snapshot_download(repo_id: str, **kwargs: Any) -> str:
        calls.append(repo_id)
        if on_call is not None:
            on_call()
        return str(dataset_dir)

    monkeypatch.setattr(recorded_move, "snapshot_download", fake_snapshot_download)
    return calls


def test_library_is_built_once_and_reused(
    monkeypatch: pytest.MonkeyPatch, dataset_dir: Path
) -> None:
    """Repeated requests reuse one instance instead of re-reading the dataset."""
    calls = _stub_snapshot_download(monkeypatch, dataset_dir)

    first = get_recorded_moves(DATASET)
    second = get_recorded_moves(DATASET)

    assert first is second
    assert calls == [DATASET], "library rebuilt on the second request"
    assert first.list_moves() == ["hello"]


def test_distinct_datasets_get_distinct_libraries(
    monkeypatch: pytest.MonkeyPatch, dataset_dir: Path
) -> None:
    """The cache keys on the dataset name, so libraries don't alias."""
    _stub_snapshot_download(monkeypatch, dataset_dir)

    assert get_recorded_moves(DATASET) is not get_recorded_moves("other/library")


def test_concurrent_requests_share_a_single_build(
    monkeypatch: pytest.MonkeyPatch, dataset_dir: Path
) -> None:
    """Simultaneous requests for a cold dataset trigger one download, not N.

    Without the per-dataset lock, every in-flight request starts its own
    ``snapshot_download`` — which on a cold cache means N concurrent downloads.
    """
    started = threading.Event()

    def slow_download() -> None:
        # Hold the first build open long enough for the others to pile up.
        started.set()
        threading.Event().wait(0.2)

    calls = _stub_snapshot_download(monkeypatch, dataset_dir, on_call=slow_download)

    results: list[recorded_move.RecordedMoves] = []
    results_lock = threading.Lock()

    def worker() -> None:
        library = get_recorded_moves(DATASET)
        with results_lock:
            results.append(library)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    threads[0].start()
    assert started.wait(timeout=5), "first build never started"
    for thread in threads[1:]:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert calls == [DATASET], f"expected 1 download, got {len(calls)}"
    assert len(results) == 8
    assert all(library is results[0] for library in results)
