"""TCP JPEG Frame Streaming.

Simple length-prefixed JPEG over TCP for one client.
"""

import io
import socket
import struct
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image


class TCPJPEGFrameServer:
    """Non-blocking, reconnect-safe JPEG-over-TCP server."""

    def __init__(self, listen_ip="0.0.0.0", listen_port=5010):
        self.addr = (listen_ip, listen_port)

        # Listening socket
        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listen_sock.bind(self.addr)
        self.listen_sock.listen(1)
        self.listen_sock.setblocking(False)  # << non-blocking accept

        self.client_sock = None

    def _accept_if_needed(self):
        """Try to accept a new client (non-blocking)."""
        if self.client_sock is not None:
            return  # Already have client

        try:
            client, addr = self.listen_sock.accept()
            client.setblocking(False)
            print(f"[TCPJPEGFrameServer] Client connected from {addr}")
            self.client_sock = client
        except BlockingIOError:
            # No pending connection, normal
            pass

    def send_frame(self, frame: npt.NDArray[np.uint8]):
        """Send one frame to the connected client (if any)."""

        # First: accept connections if someone is trying
        self._accept_if_needed()

        if self.client_sock is None:
            return  # No client, nothing to do

        # Encode frame
        ok, jpeg_bytes = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return

        data = jpeg_bytes.tobytes()
        header = struct.pack("!I", len(data))

        try:
            self.client_sock.sendall(header)
            self.client_sock.sendall(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[TCPJPEGFrameServer] Client disconnected")
            self.client_sock.close()
            self.client_sock = None

    def close(self):
        if self.client_sock:
            self.client_sock.close()
        self.listen_sock.close()


class TCPJPEGFrameClient:
    """Receive JPEG frames over TCP."""

    def __init__(self, server_ip: str, server_port: int = 5010, timeout: float = 5.0):
        """
        Args:
            server_ip: IP or hostname of the TCPJPEGFrameServer.
            server_port: Port where the server listens.
            timeout: Socket timeout in seconds.
        """
        self.server_addr = (server_ip, server_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(self.server_addr)
        print(f"[TCPJPEGFrameClient] Connected to {self.server_addr}")

    def _recv_exact(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes or return None on error."""
        buf = b""
        while len(buf) < n:
            try:
                chunk = self.sock.recv(n - len(buf))
            except socket.timeout:
                return None
            if not chunk:
                # Connection closed
                return None
            buf += chunk
        return buf

    def recv_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Receive one JPEG frame.

        Returns:
            RGB numpy array or None on timeout / disconnect.
        """
        # Read 4-byte length
        header = self._recv_exact(4)
        if header is None:
            return None

        (length,) = struct.unpack("!I", header)
        if length == 0:
            return None

        # Read JPEG payload
        data = self._recv_exact(length)
        if data is None:
            return None

        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return np.array(img)
        except Exception:
            return None

    def close(self) -> None:
        self.sock.close()
