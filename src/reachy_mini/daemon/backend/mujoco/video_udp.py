"""UDP JPEG Frame Sender.

This module provides a class to send JPEG frames over UDP. It encodes the frames as JPEG images and splits them into chunks to fit within the maximum packet size for UDP transmission.
"""

import socket

import cv2
import numpy as np
import numpy.typing as npt


class UDPJPEGFrameSender:
    """A class to send JPEG frames over UDP.

    Frames are JPEG-encoded and automatically reduced in quality to fit within
    the max packet size, ensuring cross-platform compatibility (macOS defaults
    to a 9216-byte UDP limit).
    """

    def __init__(
        self,
        dest_ip: str = "127.0.0.1",
        dest_port: int = 5005,
        max_packet_size: int = 8192,
        jpeg_quality: int = 80,
        min_jpeg_quality: int = 10,
    ) -> None:
        """Initialize the UDPJPEGFrameSender.

        Args:
            dest_ip (str): Destination IP address.
            dest_port (int): Destination port number.
            max_packet_size (int): Maximum size of each UDP packet. Default is 8192,
                which fits within macOS's default UDP datagram limit of 9216 bytes.
            jpeg_quality (int): Starting JPEG compression quality (1-100). Default is 80.
            min_jpeg_quality (int): Minimum JPEG quality before giving up. Default is 10.

        """
        self.addr = (dest_ip, dest_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.max_packet_size = max_packet_size
        self.jpeg_quality = jpeg_quality
        self.min_jpeg_quality = min_jpeg_quality

    def send_frame(self, frame: npt.NDArray[np.uint8], min_quality: int = 10) -> None:
        frame_cvt = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        quality = 80

        while quality >= min_quality:
            ret, jpeg_bytes = cv2.imencode(
                ".jpg", frame_cvt, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            data = jpeg_bytes.tobytes()
            if len(data) <= self.max_packet_size:
                break
            quality -= 10

        self.sock.sendto(data, self.addr)

    def close(self) -> None:
        """Close the socket."""
        self.sock.close()
