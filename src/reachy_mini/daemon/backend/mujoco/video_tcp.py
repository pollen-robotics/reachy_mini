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
    """Non-blocking accept, blocking client socket, reconnect-safe."""

    def __init__(self, listen_ip="0.0.0.0", listen_port=5010):
        self.addr = (listen_ip, listen_port)

        # Listening socket
        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listen_sock.bind(self.addr)
        self.listen_sock.listen(1)
        self.listen_sock.setblocking(False)  # non-blocking accept

        self.client_sock: socket.socket | None = None

    def _accept_if_needed(self):
        if self.client_sock is not None:
            return

        try:
            client, addr = self.listen_sock.accept()
            # DO NOT: client.setblocking(False)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[TCPJPEGFrameServer] Client connected from {addr}")
            self.client_sock = client
        except BlockingIOError:
            pass

    def send_frame(self, frame: npt.NDArray[np.uint8]):
        """Send one frame to the connected client (if any)."""
        self._accept_if_needed()
        if self.client_sock is None:
            return

        # Encode frame
        ok, jpeg_bytes = cv2.imencode(
            ".jpg",
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )
        if not ok:
            return

        data = jpeg_bytes.tobytes()
        header = struct.pack("!I", len(data))

        try:
            self.client_sock.sendall(header)
            self.client_sock.sendall(data)

        except BlockingIOError:
            # Kernel send buffer full (non-blocking socket). Just drop this frame.
            # Do NOT close the connection.
            # Next frame we try again.
            # print("[TCPJPEGFrameServer] Send buffer full, dropping frame")
            return

        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"[TCPJPEGFrameServer] Client disconnected: {e}")
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
        """Receive exactly n bytes or return None on error/timeout."""
        buf = b""
        while len(buf) < n:
            try:
                chunk = self.sock.recv(n - len(buf))
            except socket.timeout:
                # no data in time
                return None
            except OSError as e:
                print(f"[TCPJPEGFrameClient] OSError in recv: {e}")
                return None

            if not chunk:
                # peer closed
                print("[TCPJPEGFrameClient] Connection closed by peer")
                return None
            buf += chunk
        return buf

    def recv_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Receive one JPEG frame. Returns RGB array or None."""
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
        except Exception as e:
            print(f"[TCPJPEGFrameClient] Error decoding JPEG: {e}")
            return None

    def close(self) -> None:
        self.sock.close()