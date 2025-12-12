"""Camera API endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from reachy_mini.daemon.daemon import Daemon

router = APIRouter(prefix="/camera", tags=["camera"])


class CameraStatusResponse(BaseModel):
    """Response model for camera status."""
    available: bool
    backend: str
    resolution: tuple[int, int] | None = None
    framerate: int | None = None


class WebRTCOffer(BaseModel):
    """Request model for WebRTC offer."""
    sdp: str
    type: str


class WebRTCAnswer(BaseModel):
    """Response model for WebRTC answer."""
    answer: dict
    ice_candidates: list[dict]


class WebRTCCandidate(BaseModel):
    """Request model for WebRTC ICE candidate."""
    candidate: str
    sdpMid: str | None = None
    sdpMLineIndex: int | None = None


def get_daemon(request: Request) -> Daemon:
    """Dependency to get the daemon instance."""
    return request.app.state.daemon


@router.get("/status")
async def get_camera_status(daemon: Daemon = Depends(get_daemon)) -> CameraStatusResponse:
    """Get the current camera status.
    
    Returns:
        CameraStatusResponse: The camera status including availability, backend, resolution, and framerate.
    """
    try:
        # Check if media manager is available and has camera
        if not hasattr(daemon, 'media_manager') or daemon.media_manager is None:
            return CameraStatusResponse(
                available=False,
                backend="NO_MEDIA",
                resolution=None,
                framerate=None
            )
        
        if daemon.media_manager.camera is None:
            return CameraStatusResponse(
                available=False,
                backend=str(daemon.media_manager.backend),
                resolution=None,
                framerate=None
            )
        
        # Get camera resolution and framerate if available
        resolution = None
        framerate = None
        
        if hasattr(daemon.media_manager.camera, 'resolution'):
            resolution = daemon.media_manager.camera.resolution
        
        if hasattr(daemon.media_manager.camera, 'framerate'):
            framerate = daemon.media_manager.camera.framerate
        
        return CameraStatusResponse(
            available=True,
            backend=str(daemon.media_manager.backend),
            resolution=resolution,
            framerate=framerate
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting camera status: {str(e)}"
        )


@router.post("/webrtc/offer")
async def handle_webrtc_offer(offer: WebRTCOffer, daemon: Daemon = Depends(get_daemon)) -> WebRTCAnswer:
    """Handle WebRTC offer and return answer with ICE candidates.
    
    Args:
        offer: The WebRTC offer containing SDP and type.
        
    Returns:
        WebRTCAnswer: The WebRTC answer and ICE candidates.
        
    Raises:
        HTTPException: If WebRTC is not available or offer processing fails.
    """
    try:
        # Check if WebRTC backend is available
        if daemon.media_manager is None or daemon.media_manager.backend != "webrtc":
            raise HTTPException(
                status_code=400,
                detail="WebRTC backend not available"
            )
        
        # Check if camera is available
        if daemon.media_manager.camera is None:
            raise HTTPException(
                status_code=400,
                detail="Camera not available"
            )
        
        # For now, return a mock response since the actual WebRTC implementation
        # would require more complex signaling setup
        # TODO: Implement actual WebRTC signaling with the GStreamer backend
        
        mock_answer = {
            "type": "answer",
            "sdp": "v=0\r\no=- 1234567890 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\na=ice-ufrag:abcdef\r\na=ice-pwd:xyz123\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\nc=IN IP4 0.0.0.0\r\na=mid:video\r\na=sendrecv\r\na=rtpmap:96 H264/90000\r\n"
        }
        
        mock_ice_candidates = [
            {
                "candidate": "candidate:1234567890 1 udp 2122260223 192.168.1.100 5000 typ host generation 0",
                "sdpMid": "video",
                "sdpMLineIndex": 0
            }
        ]
        
        return WebRTCAnswer(
            answer=mock_answer,
            ice_candidates=mock_ice_candidates
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing WebRTC offer: {str(e)}"
        )


@router.post("/webrtc/ice")
async def handle_ice_candidate(candidate: WebRTCCandidate, daemon: Daemon = Depends(get_daemon)) -> JSONResponse:
    """Handle ICE candidate from client.
    
    Args:
        candidate: The ICE candidate to add to the connection.
        
    Returns:
        JSONResponse: Confirmation that the candidate was processed.
        
    Raises:
        HTTPException: If WebRTC is not available or candidate processing fails.
    """
    try:
        # Check if WebRTC backend is available
        if daemon.media_manager is None or daemon.media_manager.backend != "webrtc":
            raise HTTPException(
                status_code=400,
                detail="WebRTC backend not available"
            )
        
        # For now, just acknowledge the candidate
        # TODO: Implement actual ICE candidate handling with GStreamer
        
        return JSONResponse(
            status_code=200,
            content={"status": "ok", "message": "ICE candidate received"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing ICE candidate: {str(e)}"
        )