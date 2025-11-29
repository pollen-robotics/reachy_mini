"""UDP JPEG Frame Sender.

This module provides a class to send JPEG frames over UDP. It encodes the frames as JPEG images and splits them into chunks to fit within the maximum packet size for UDP transmission.
"""

import io
import socket
import struct
from typing import List, Optional

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image


class UDPJPEGFrameSender:
    """A class to send JPEG frames over UDP."""

    def __init__(
        self,
        dest_ip: str = "127.0.0.1",
        dest_port: int = 5005,
        max_packet_size: int = 1400,
    ) -> None:
        """Initialize the UDPJPEGFrameSender.

        Args:
            dest_ip (str): Destination IP address.
            dest_port (int): Destination port number.
            max_packet_size (int): Maximum size of each UDP packet.

        """
        self.addr = (dest_ip, dest_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.max_packet_size = max_packet_size

    def send_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        """Send a frame as a JPEG image over UDP.

        Args:
            frame (np.ndarray): The frame to be sent, in RGB format.

        """
        frame_cvt = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, jpeg_bytes = cv2.imencode(
            ".jpg", frame_cvt, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        data = jpeg_bytes.tobytes()
        total_size = len(data)
        n_chunks = (total_size + self.max_packet_size - 1) // self.max_packet_size
        self.sock.sendto(struct.pack("!II", n_chunks, total_size), self.addr)
        for i in range(n_chunks):
            start = i * self.max_packet_size
            end = min(start + self.max_packet_size, total_size)
            self.sock.sendto(data[start:end], self.addr)

    def close(self) -> None:
        """Close the socket."""
        self.sock.close()


class UDPJPEGFrameReceiver:
    """A class to receive JPEG frames over UDP."""

    def __init__(self, listen_ip: str = "127.0.0.1", listen_port: int = 5005, timeout: float = 1.0) -> None:
        """Initialize the UDPJPEGFrameReceiver.

        Args:
            listen_ip (str): The IP address to listen on.
            listen_port (int): The port to listen on.
            timeout (float): The timeout in seconds.

        """
        self.addr = (listen_ip, listen_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.addr)
        self.sock.settimeout(timeout)

    def recv_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Receive and reconstruct one JPEG frame."""
        try:
            # Receive header: number of chunks, total size
            header, _ = self.sock.recvfrom(8)  # 2x uint32
            n_chunks, total_size = struct.unpack("!II", header)

            chunks: List[bytes] = []
            received = 0
            while len(chunks) < n_chunks:
                packet, _ = self.sock.recvfrom(65536)
                chunks.append(packet)
                received += len(packet)

            if received != total_size:
                # corrupted / missing chunks
                return None

            jpeg_bytes = b"".join(chunks)
            img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            return np.array(img)

        except socket.timeout:
            return None
        except Exception:
            # partial/corrupted frame, skip it
            return None

    def close(self) -> None:
        """Close the socket."""
        self.sock.close()