"""Audio streaming API routes.

Exposes:
- WebSocket for streaming audio output (TTS -> robot speaker)
- WebSocket for streaming audio input (robot mic -> client)
- REST endpoints for audio info and sound playback
"""

import asyncio

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ....daemon.backend.abstract import Backend
from ..dependencies import get_backend, ws_get_backend

router = APIRouter(prefix="/audio")


class AudioInfo(BaseModel):
    """Audio device information."""

    input_sample_rate: int
    output_sample_rate: int
    input_channels: int
    output_channels: int


class PlaySoundRequest(BaseModel):
    """Request to play a sound file."""

    sound_file: str


@router.get("/info")
async def get_audio_info(
    backend: Backend = Depends(get_backend),
) -> AudioInfo:
    """Get audio device information."""
    if not backend.audio:
        raise HTTPException(status_code=503, detail="Audio not available")

    return AudioInfo(
        input_sample_rate=backend.audio.get_input_audio_samplerate(),
        output_sample_rate=backend.audio.get_output_audio_samplerate(),
        input_channels=backend.audio.get_input_channels(),
        output_channels=backend.audio.get_output_channels(),
    )


@router.post("/play-sound")
async def play_sound(
    request: PlaySoundRequest,
    backend: Backend = Depends(get_backend),
) -> dict[str, str]:
    """Play a sound file on the robot speaker."""
    if not backend.audio:
        raise HTTPException(status_code=503, detail="Audio not available")

    try:
        backend.audio.play_sound(request.sound_file)
        return {"status": "ok"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.websocket("/ws/output")
async def ws_audio_output(
    websocket: WebSocket,
    sample_rate: int = 24000,
    backend: Backend = Depends(ws_get_backend),
) -> None:
    """Stream audio to the robot speaker.

    Client sends binary PCM int16 mono audio.
    Query params:
        sample_rate: Input sample rate (default 24000 for OpenAI TTS)
    """
    if not backend.audio:
        await websocket.close(code=1011, reason="Audio not available")
        return

    await websocket.accept()
    backend.audio.start_playing()
    target_sr = backend.audio.get_output_audio_samplerate()

    try:
        while True:
            data = await websocket.receive_bytes()

            # PCM int16 -> float32
            samples_int16 = np.frombuffer(data, dtype=np.int16)
            samples = samples_int16.astype(np.float32) / 32768.0

            # Resample if needed
            if sample_rate != target_sr:
                import scipy.signal

                samples = scipy.signal.resample(
                    samples, int(len(samples) * target_sr / sample_rate)
                ).astype(np.float32)

            backend.audio.push_audio_sample(samples)

    except WebSocketDisconnect:
        pass


@router.websocket("/ws/input")
async def ws_audio_input(
    websocket: WebSocket,
    sample_rate: int = 16000,
    chunk_ms: int = 100,
    backend: Backend = Depends(ws_get_backend),
) -> None:
    """Stream audio from the robot microphone.

    Server sends binary PCM int16 mono audio chunks.
    Query params:
        sample_rate: Output sample rate (default 16000 for Whisper)
        chunk_ms: Chunk interval in ms (default 100)
    """
    if not backend.audio:
        await websocket.close(code=1011, reason="Audio not available")
        return

    await websocket.accept()
    backend.audio.start_recording()
    source_sr = backend.audio.get_input_audio_samplerate()
    interval = chunk_ms / 1000.0

    try:
        while True:
            samples = backend.audio.get_audio_sample()

            if samples is not None and len(samples) > 0:
                # Stereo -> mono
                if samples.ndim == 2:
                    samples = samples[:, 0]

                # Resample if needed
                if source_sr != sample_rate:
                    import scipy.signal

                    samples = scipy.signal.resample(
                        samples, int(len(samples) * sample_rate / source_sr)
                    ).astype(np.float32)

                # float32 -> PCM int16
                samples_int16 = (samples * 32767).astype(np.int16)
                await websocket.send_bytes(samples_int16.tobytes())

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        pass
    finally:
        backend.audio.stop_recording()

