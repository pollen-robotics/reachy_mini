"""Cover the inline-move / audio upload, play, and cancel handlers.

Exercises ``daemon/backend/abstract.py``'s fire-and-forget upload state
machine (start/chunk/finish for moves and audio), the stale-slot
eviction, and the play/cancel paths — all with no hardware or network.
"""

import base64
import gzip
import json
import os
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from reachy_mini.daemon.backend.abstract import _PlaybackCancelToken
from reachy_mini.io.protocol import (
    CancelAudioCmd,
    CancelMoveCmd,
    PlayUploadedAudioCmd,
    PlayUploadedMoveCmd,
    UploadAudioChunkCmd,
    UploadAudioFinishCmd,
    UploadAudioStartCmd,
    UploadMoveChunkCmd,
    UploadMoveFinishCmd,
    UploadMoveStartCmd,
)
from reachy_mini.motion.recorded_move import RecordedMove


def _minimal_move_dict() -> dict:
    """Smallest RecordedMove-parsable dict: 2 frames, identity head poses."""
    return {
        "description": "unit-test move",
        "time": [0.0, 0.1],
        "set_target_data": [
            {"head": np.eye(4).tolist(), "antennas": [0.0, 0.0], "body_yaw": 0.0},
            {"head": np.eye(4).tolist(), "antennas": [0.0, 0.0], "body_yaw": 0.0},
        ],
    }


def _split(payload: str, n: int) -> list[str]:
    """Split a string into n roughly-equal contiguous chunks (>=1 each)."""
    size = max(1, len(payload) // n + 1)
    parts = [payload[i : i + size] for i in range(0, len(payload), size)]
    return parts or [""]


def _feed_move(backend, upload_id, chunks, encoding) -> None:
    """Drive start + in-order chunks for a move upload slot."""
    backend._handle_upload_start(
        UploadMoveStartCmd(
            upload_id=upload_id, total_chunks=len(chunks), encoding=encoding
        )
    )
    for i, part in enumerate(chunks):
        backend._handle_upload_chunk(
            UploadMoveChunkCmd(upload_id=upload_id, chunk_index=i, chunk=part)
        )


# --------------------------------------------------------------------------
# Move upload — happy paths
# --------------------------------------------------------------------------


def test_move_upload_json_happy(sim_backend):
    """JSON-encoded move assembled from >=2 chunks parses into a RecordedMove."""
    payload = json.dumps(_minimal_move_dict())
    chunks = _split(payload, 2)
    assert len(chunks) >= 2
    _feed_move(sim_backend, "m-json", chunks, "json")
    sim_backend._handle_upload_finish(UploadMoveFinishCmd(upload_id="m-json"))
    move = sim_backend._uploaded_moves["m-json"]
    assert isinstance(move, RecordedMove)
    # slot state is cleaned up on finish
    assert "m-json" not in sim_backend._upload_chunks


def test_move_upload_gzip_base64_happy(sim_backend):
    """gzip+base64 move round-trips through decode + decompress."""
    raw = gzip.compress(json.dumps(_minimal_move_dict()).encode())
    payload = base64.b64encode(raw).decode()
    chunks = _split(payload, 2)
    assert len(chunks) >= 2
    _feed_move(sim_backend, "m-gz", chunks, "gzip+base64")
    sim_backend._handle_upload_finish(UploadMoveFinishCmd(upload_id="m-gz"))
    assert isinstance(sim_backend._uploaded_moves["m-gz"], RecordedMove)


# --------------------------------------------------------------------------
# Move upload — error branches
# --------------------------------------------------------------------------


def test_move_chunk_no_slot_dropped(sim_backend):
    """Chunk for an unopened slot is dropped without crashing."""
    sim_backend._handle_upload_chunk(
        UploadMoveChunkCmd(upload_id="ghost", chunk_index=0, chunk="x")
    )
    assert "ghost" not in sim_backend._upload_chunks


def test_move_chunk_out_of_order_drops_slot(sim_backend):
    """A misordered chunk discards the whole slot."""
    sim_backend._handle_upload_start(
        UploadMoveStartCmd(upload_id="ooo", total_chunks=2)
    )
    sim_backend._handle_upload_chunk(
        UploadMoveChunkCmd(upload_id="ooo", chunk_index=1, chunk="x")
    )
    assert "ooo" not in sim_backend._upload_chunks
    assert "ooo" not in sim_backend._upload_meta


def test_move_chunk_index_exceeds_total_drops_slot(sim_backend):
    """A chunk_index past the declared total discards the slot."""
    sim_backend._handle_upload_start(
        UploadMoveStartCmd(upload_id="ovf", total_chunks=1)
    )
    sim_backend._handle_upload_chunk(
        UploadMoveChunkCmd(upload_id="ovf", chunk_index=0, chunk="a")
    )
    # index 1 == expected len, but >= total_chunks (1) -> dropped
    sim_backend._handle_upload_chunk(
        UploadMoveChunkCmd(upload_id="ovf", chunk_index=1, chunk="b")
    )
    assert "ovf" not in sim_backend._upload_chunks


def test_move_finish_count_mismatch_stores_nothing(sim_backend):
    """Finishing with fewer chunks than declared stores no move."""
    sim_backend._handle_upload_start(
        UploadMoveStartCmd(upload_id="mm", total_chunks=2)
    )
    sim_backend._handle_upload_chunk(
        UploadMoveChunkCmd(upload_id="mm", chunk_index=0, chunk="{}")
    )
    sim_backend._handle_upload_finish(UploadMoveFinishCmd(upload_id="mm"))
    assert "mm" not in sim_backend._uploaded_moves


def test_move_finish_unknown_slot(sim_backend):
    """Finishing an id that was never started is a no-op."""
    sim_backend._handle_upload_finish(UploadMoveFinishCmd(upload_id="nope"))
    assert "nope" not in sim_backend._uploaded_moves


def test_move_finish_unknown_encoding_stores_nothing(sim_backend):
    """An unrecognized encoding on finish drops the slot."""
    _feed_move(sim_backend, "enc", ["{}"], "json")
    sim_backend._upload_meta["enc"]["encoding"] = "bogus"
    sim_backend._handle_upload_finish(UploadMoveFinishCmd(upload_id="enc"))
    assert "enc" not in sim_backend._uploaded_moves


def test_move_start_too_many_active_slots_refused(sim_backend):
    """A second slot is refused when the active-slot cap is reached."""
    sim_backend._upload_max_active_slots = 1
    sim_backend._handle_upload_start(UploadMoveStartCmd(upload_id="s1", total_chunks=1))
    sim_backend._handle_upload_start(UploadMoveStartCmd(upload_id="s2", total_chunks=1))
    assert "s1" in sim_backend._upload_chunks
    assert "s2" not in sim_backend._upload_chunks


def test_move_evict_stale_uploads(sim_backend):
    """Slots older than the TTL are evicted."""
    sim_backend._handle_upload_start(UploadMoveStartCmd(upload_id="old", total_chunks=1))
    sim_backend._upload_ts["old"] = 0.0  # epoch -> ancient
    sim_backend._upload_ttl_s = 1.0
    sim_backend._evict_stale_uploads()
    assert "old" not in sim_backend._upload_chunks
    assert "old" not in sim_backend._upload_meta
    assert "old" not in sim_backend._upload_ts


# --------------------------------------------------------------------------
# Audio upload
# --------------------------------------------------------------------------


def test_audio_upload_happy(sim_backend, tmp_path):
    """Base64 audio assembled from chunks is written under the temp dir."""
    sim_backend._audio_temp_dir = str(tmp_path)
    raw = b"RIFF----WAVEfake-pcm-bytes"
    payload = base64.b64encode(raw).decode()
    chunks = _split(payload, 2)
    sim_backend._handle_audio_start(
        UploadAudioStartCmd(upload_id="a1", total_chunks=len(chunks))
    )
    for i, part in enumerate(chunks):
        sim_backend._handle_audio_chunk(
            UploadAudioChunkCmd(upload_id="a1", chunk_index=i, chunk=part)
        )
    sim_backend._handle_audio_finish(UploadAudioFinishCmd(upload_id="a1"))
    path = sim_backend._uploaded_audios["a1"]
    assert os.path.dirname(path) == str(tmp_path)
    assert path.endswith(".wav")
    with open(path, "rb") as f:
        assert f.read() == raw


def test_audio_chunk_no_slot_dropped(sim_backend):
    """Audio chunk for an unopened slot is dropped."""
    sim_backend._handle_audio_chunk(
        UploadAudioChunkCmd(upload_id="ghost", chunk_index=0, chunk="x")
    )
    assert "ghost" not in sim_backend._audio_chunks


def test_audio_chunk_out_of_order_drops_slot(sim_backend):
    """A misordered audio chunk discards the slot."""
    sim_backend._handle_audio_start(
        UploadAudioStartCmd(upload_id="ooo", total_chunks=2)
    )
    sim_backend._handle_audio_chunk(
        UploadAudioChunkCmd(upload_id="ooo", chunk_index=1, chunk="x")
    )
    assert "ooo" not in sim_backend._audio_chunks


def test_audio_chunk_index_exceeds_total_drops_slot(sim_backend):
    """An audio chunk_index past the declared total discards the slot."""
    sim_backend._handle_audio_start(
        UploadAudioStartCmd(upload_id="ovf", total_chunks=1)
    )
    sim_backend._handle_audio_chunk(
        UploadAudioChunkCmd(upload_id="ovf", chunk_index=0, chunk="a")
    )
    sim_backend._handle_audio_chunk(
        UploadAudioChunkCmd(upload_id="ovf", chunk_index=1, chunk="b")
    )
    assert "ovf" not in sim_backend._audio_chunks


def test_audio_finish_count_mismatch_writes_nothing(sim_backend, tmp_path):
    """Audio finish with the wrong chunk count writes no file."""
    sim_backend._audio_temp_dir = str(tmp_path)
    sim_backend._handle_audio_start(
        UploadAudioStartCmd(upload_id="mm", total_chunks=2)
    )
    sim_backend._handle_audio_chunk(
        UploadAudioChunkCmd(upload_id="mm", chunk_index=0, chunk="AA==")
    )
    sim_backend._handle_audio_finish(UploadAudioFinishCmd(upload_id="mm"))
    assert "mm" not in sim_backend._uploaded_audios


def test_audio_evict_stale_slots_and_orphan_file(sim_backend, tmp_path):
    """In-progress slots and orphaned finished files past TTL are evicted."""
    sim_backend._audio_temp_dir = str(tmp_path)
    sim_backend._upload_ttl_s = 1.0
    # in-progress stale slot
    sim_backend._handle_audio_start(
        UploadAudioStartCmd(upload_id="old", total_chunks=1)
    )
    sim_backend._audio_ts["old"] = 0.0
    # orphaned finished audio file, aged past TTL
    orphan = tmp_path / "orphan.wav"
    orphan.write_bytes(b"data")
    os.utime(orphan, (0, 0))  # mtime at epoch
    sim_backend._uploaded_audios["orph"] = str(orphan)

    sim_backend._evict_stale_audios()

    assert "old" not in sim_backend._audio_chunks
    assert "orph" not in sim_backend._uploaded_audios
    assert not orphan.exists()


# --------------------------------------------------------------------------
# play / cancel audio
# --------------------------------------------------------------------------


def test_play_uploaded_audio_no_such_audio(sim_backend, monkeypatch):
    """Playing an unknown audio id broadcasts an error."""
    sent: list[dict] = []
    monkeypatch.setattr(
        sim_backend, "broadcast_to_all_clients", lambda p: sent.append(json.loads(p))
    )
    monkeypatch.setattr(sim_backend, "play_sound", MagicMock())
    sim_backend._handle_play_uploaded_audio(PlayUploadedAudioCmd(upload_id="none"))
    assert sent[-1]["error"] == "no such uploaded audio"
    assert sim_backend._active_audio_upload_id is None


def test_play_uploaded_audio_happy(sim_backend, monkeypatch):
    """A known audio id broadcasts started, plays, and claims the active slot."""
    sent: list[dict] = []
    play = MagicMock()
    monkeypatch.setattr(
        sim_backend, "broadcast_to_all_clients", lambda p: sent.append(json.loads(p))
    )
    monkeypatch.setattr(sim_backend, "play_sound", play)
    sim_backend._uploaded_audios["a"] = "/tmp/a.wav"
    sim_backend._handle_play_uploaded_audio(PlayUploadedAudioCmd(upload_id="a"))
    assert sent[-1]["started"] is True
    play.assert_called_once_with("/tmp/a.wav")
    assert sim_backend._active_audio_upload_id == "a"


def test_play_uploaded_audio_play_sound_raises(sim_backend, monkeypatch):
    """A play_sound failure broadcasts an error and clears the active slot."""
    sent: list[dict] = []
    monkeypatch.setattr(
        sim_backend, "broadcast_to_all_clients", lambda p: sent.append(json.loads(p))
    )
    monkeypatch.setattr(
        sim_backend, "play_sound", MagicMock(side_effect=RuntimeError("boom"))
    )
    sim_backend._uploaded_audios["a"] = "/tmp/a.wav"
    sim_backend._handle_play_uploaded_audio(PlayUploadedAudioCmd(upload_id="a"))
    assert sent[-1]["error"] == "boom"
    assert sim_backend._active_audio_upload_id is None


def test_cancel_audio_matching(sim_backend, monkeypatch):
    """cancel_audio for the active id stops sound and clears the slot."""
    stop = MagicMock()
    monkeypatch.setattr(sim_backend, "stop_sound", stop)
    sim_backend._active_audio_upload_id = "a"
    sim_backend._handle_cancel_audio(CancelAudioCmd(upload_id="a"))
    stop.assert_called_once()
    assert sim_backend._active_audio_upload_id is None


def test_cancel_audio_non_matching_noop(sim_backend, monkeypatch):
    """cancel_audio for a different id is a no-op."""
    stop = MagicMock()
    monkeypatch.setattr(sim_backend, "stop_sound", stop)
    sim_backend._active_audio_upload_id = "a"
    sim_backend._handle_cancel_audio(CancelAudioCmd(upload_id="other"))
    stop.assert_not_called()
    assert sim_backend._active_audio_upload_id == "a"


# --------------------------------------------------------------------------
# cancel move
# --------------------------------------------------------------------------


def test_cancel_move_matching(sim_backend):
    """cancel_move flips the active token when its upload_id matches."""
    token = _PlaybackCancelToken("m1")
    sim_backend._active_move_token = token
    sim_backend._handle_cancel_move(CancelMoveCmd(upload_id="m1"))
    assert token.cancelled is True


def test_cancel_move_non_matching_noop(sim_backend):
    """cancel_move leaves a non-matching token untouched."""
    token = _PlaybackCancelToken("m1")
    sim_backend._active_move_token = token
    sim_backend._handle_cancel_move(CancelMoveCmd(upload_id="other"))
    assert token.cancelled is False


# --------------------------------------------------------------------------
# async play uploaded move
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_play_uploaded_move_no_such_move(sim_backend, monkeypatch):
    """Playing an unknown move id broadcasts an error."""
    sent: list[dict] = []
    monkeypatch.setattr(
        sim_backend, "broadcast_to_all_clients", lambda p: sent.append(json.loads(p))
    )
    await sim_backend._async_play_uploaded_move(PlayUploadedMoveCmd(upload_id="none"))
    assert sent[-1]["error"].startswith("no such uploaded move")


@pytest.mark.asyncio
async def test_async_play_uploaded_move_happy(sim_backend, monkeypatch):
    """A known move broadcasts started then finished; token installed then cleared."""
    sent: list[dict] = []
    monkeypatch.setattr(
        sim_backend, "broadcast_to_all_clients", lambda p: sent.append(json.loads(p))
    )
    play = AsyncMock()
    monkeypatch.setattr(sim_backend, "play_move", play)
    sim_backend._uploaded_moves["m1"] = RecordedMove(_minimal_move_dict())

    await sim_backend._async_play_uploaded_move(PlayUploadedMoveCmd(upload_id="m1"))

    play.assert_awaited_once()
    assert sent[0]["started"] is True
    assert sent[-1]["finished"] is True
    assert sim_backend._active_move_token is None
