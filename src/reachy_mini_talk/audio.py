from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
from .speech_tapper import SwayRollRT, HOP_MS


@dataclass
class AudioConfig:
    output_sample_rate: int = 24_000
    movement_latency_s: float = 0.08


def pcm_to_b64(array: np.ndarray) -> str:
    """array: shape (N,) int16 or (1,N) int16 -> base64 string for OpenAI input buffer."""
    a = np.asarray(array).reshape(-1).astype(np.int16, copy=False)
    return base64.b64encode(a.tobytes()).decode("utf-8")


class AudioSync:
    """
    Routes assistant audio to:
      1) a playback queue for fastrtc
      2) a sway engine that emits head-offsets aligned to audio
    """

    def __init__(
        self,
        cfg: AudioConfig,
        set_offsets: Callable[[Tuple[float, float, float, float, float, float]], None],
        sway: Optional[SwayRollRT] = None,
    ) -> None:
        """
        set_offsets: callback receiving (x,y,z,roll,pitch,yaw) at each hop, in meters/radians.
        """
        self.cfg = cfg
        self.set_offsets = set_offsets
        self.sway = sway or SwayRollRT()

        self.playback_q: asyncio.Queue = (
            asyncio.Queue()
        )  # (sr:int, pcm: np.ndarray[1,N] int16)
        self._sway_q: asyncio.Queue = (
            asyncio.Queue()
        )  # (sr:int, pcm: np.ndarray[1,N] int16)

        self._base_ts: Optional[float] = None
        self._hops_done: int = 0
        self._sway_task: Optional[asyncio.Task] = None

    # lifecycle

    def start(self) -> None:
        if self._sway_task is None:
            self._sway_task = asyncio.create_task(self._sway_consumer())

    async def stop(self) -> None:
        if self._sway_task:
            self._sway_task.cancel()
            try:
                await self._sway_task
            except asyncio.CancelledError:
                pass
            self._sway_task = None
        self._reset_all()
        self._drain(self._sway_q)
        self._drain(self.playback_q)

    # event hooks from your Realtime loop

    def on_input_speech_started(self) -> None:
        """User started speaking (server VAD). Reset sync state."""
        self._reset_all()
        self._drain(self._sway_q)

    def on_response_started(self) -> None:
        """Assistant began a new utterance."""
        self._reset_all()
        self._drain(self._sway_q)

    def on_response_completed(self) -> None:
        """Assistant finished an utterance."""
        self._reset_all()
        self._drain(self._sway_q)

    def on_response_audio_delta(self, delta_b64: str) -> None:
        """
        Called for each 'response.audio.delta' event.
        Pushes audio both to playback and to sway engine.
        """
        buf = np.frombuffer(base64.b64decode(delta_b64), dtype=np.int16).reshape(1, -1)
        # 1) to fastrtc playback
        self.playback_q.put_nowait((self.cfg.output_sample_rate, buf))
        # 2) to sway engine
        self._sway_q.put_nowait((self.cfg.output_sample_rate, buf))

    # fastrtc hook

    async def emit_playback(self):
        """Await next (sr, pcm[1,N]) frame for your Stream(...)."""
        return await self.playback_q.get()

    # internal

    async def _sway_consumer(self):
        """
        Convert streaming audio chunks into head-offset poses at precise times.
        """
        hop_dt = HOP_MS / 1000.0
        loop = asyncio.get_running_loop()

        while True:
            sr, chunk = await self._sway_q.get()  # (1,N), int16
            pcm = np.asarray(chunk).squeeze(0)
            results = self.sway.feed(pcm, sr)  # list of dicts with keys x_mm..yaw_rad

            if self._base_ts is None:
                # anchor when first audio samples of this utterance arrive
                self._base_ts = loop.time()

            i = 0
            while i < len(results):
                if self._base_ts is None:
                    self._base_ts = loop.time()
                    continue

                target = (
                    self._base_ts
                    + self.cfg.movement_latency_s
                    + self._hops_done * hop_dt
                )
                now = loop.time()

                # if late by ≥1 hop, drop poses to catch up (no drift accumulation)
                if now - target >= hop_dt:
                    lag_hops = int((now - target) / hop_dt)
                    drop = min(
                        lag_hops, len(results) - i - 1
                    )  # keep at least one to show
                    if drop > 0:
                        self._hops_done += drop
                        i += drop
                        continue

                # if early, wait
                if target > now:
                    await asyncio.sleep(target - now)

                r = results[i]
                # meters + radians
                offsets = (
                    r["x_mm"] / 1000.0,
                    r["y_mm"] / 1000.0,
                    r["z_mm"] / 1000.0,
                    r["roll_rad"],
                    r["pitch_rad"],
                    r["yaw_rad"],
                )
                self.set_offsets(offsets)

                self._hops_done += 1
                i += 1

    def _reset_all(self) -> None:
        self._base_ts = None
        self._hops_done = 0
        self.sway.reset()

    @staticmethod
    def _drain(q: asyncio.Queue) -> None:
        try:
            while True:
                q.get_nowait()
        except asyncio.QueueEmpty:
            pass
