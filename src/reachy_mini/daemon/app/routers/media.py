"""Media streaming API routes.

This module provides WebSocket endpoints for streaming video and audio
from the local camera and microphone. Used primarily for mockup-sim mode.
"""

import asyncio
import base64
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from reachy_mini.media.media_manager import MediaBackend, MediaManager

router = APIRouter(prefix="/media")
logger = logging.getLogger(__name__)

# Shared MediaManager instance for mockup-sim streaming
_media_manager: Optional[MediaManager] = None


def get_media_manager() -> Optional[MediaManager]:
    """Get the shared MediaManager instance."""
    return _media_manager


def init_media_manager(log_level: str = "INFO") -> MediaManager:
    """Initialize the shared MediaManager with camera and audio.
    
    Args:
        log_level: Logging level for the media manager.
        
    Returns:
        The initialized MediaManager instance.
    """
    global _media_manager
    if _media_manager is None:
        logger.info("Initializing MediaManager for mockup-sim streaming...")
        _media_manager = MediaManager(
            backend=MediaBackend.DEFAULT,  # OpenCV camera + SoundDevice audio
            log_level=log_level,
        )
        _media_manager.start_recording()
        logger.info("MediaManager initialized with camera and microphone.")
    return _media_manager


def close_media_manager() -> None:
    """Close and cleanup the shared MediaManager."""
    global _media_manager
    if _media_manager is not None:
        logger.info("Closing MediaManager...")
        _media_manager.stop_recording()
        _media_manager.close()
        _media_manager = None
        logger.info("MediaManager closed.")


@router.websocket("/ws/video")
async def ws_video_stream(
    websocket: WebSocket,
    fps: int = 15,
    quality: int = 80,
) -> None:
    """WebSocket endpoint to stream video frames from the camera.
    
    Frames are sent as base64-encoded JPEG images.
    
    Args:
        websocket: The WebSocket connection.
        fps: Target frames per second (default: 15).
        quality: JPEG quality 1-100 (default: 80).
    """
    await websocket.accept()
    
    media = get_media_manager()
    if media is None or media.camera is None:
        await websocket.send_json({"error": "Camera not available"})
        await websocket.close()
        return
    
    period = 1.0 / fps
    logger.info(f"Video stream started at {fps} fps, quality {quality}")
    
    try:
        import cv2
        
        while True:
            frame = media.get_frame()
            if frame is not None:
                # Encode frame as JPEG
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                _, buffer = cv2.imencode(".jpg", frame, encode_params)
                
                # Send as base64
                frame_b64 = base64.b64encode(buffer).decode("utf-8")
                await websocket.send_json({
                    "type": "frame",
                    "data": frame_b64,
                })
            
            await asyncio.sleep(period)
            
    except WebSocketDisconnect:
        logger.info("Video stream client disconnected")
    except Exception as e:
        logger.error(f"Video stream error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        logger.info("Video stream ended")


@router.websocket("/ws/audio")
async def ws_audio_stream(
    websocket: WebSocket,
    sample_rate: int = 16000,
) -> None:
    """WebSocket endpoint to stream audio samples from the microphone.
    
    Audio is sent as base64-encoded raw float32 samples.
    
    Args:
        websocket: The WebSocket connection.
        sample_rate: Expected sample rate (for client info).
    """
    await websocket.accept()
    
    media = get_media_manager()
    if media is None or media.audio is None:
        await websocket.send_json({"error": "Audio not available"})
        await websocket.close()
        return
    
    logger.info(f"Audio stream started (sample_rate={sample_rate})")
    
    try:
        while True:
            samples = media.get_audio_sample()
            if samples is not None and len(samples) > 0:
                # Send as base64-encoded bytes
                audio_b64 = base64.b64encode(samples.tobytes()).decode("utf-8")
                await websocket.send_json({
                    "type": "audio",
                    "data": audio_b64,
                    "dtype": "float32",
                    "shape": list(samples.shape),
                })
            
            # Small sleep to avoid busy loop
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        logger.info("Audio stream client disconnected")
    except Exception as e:
        logger.error(f"Audio stream error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        logger.info("Audio stream ended")


@router.get("/status")
async def get_media_status() -> dict:
    """Get the status of the media streaming system."""
    media = get_media_manager()
    
    if media is None:
        return {
            "available": False,
            "camera": None,
            "audio": None,
        }
    
    return {
        "available": True,
        "camera": media.camera is not None,
        "audio": media.audio is not None,
    }
