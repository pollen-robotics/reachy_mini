"""UDP JPEG Frame Sender.

This module provides a class to send JPEG frames over UDP. It encodes the frames as JPEG images and splits them into chunks to fit within the maximum packet size for UDP transmission.
"""

import socket
import struct

import cv2
import numpy as np
import numpy.typing as npt


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
