import numpy as np
import socket
import struct
import cv2

class UDPJPEGFrameSender:
    def __init__(self, dest_ip="127.0.0.1", dest_port=5005, max_packet_size=1400):
        self.addr = (dest_ip, dest_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.max_packet_size = max_packet_size

    def send_frame(self, frame: np.ndarray):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, jpeg_bytes = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = jpeg_bytes.tobytes()
        total_size = len(data)
        n_chunks = (total_size + self.max_packet_size - 1) // self.max_packet_size
        self.sock.sendto(struct.pack('!II', n_chunks, total_size), self.addr)
        for i in range(n_chunks):
            start = i * self.max_packet_size
            end = min(start + self.max_packet_size, total_size)
            self.sock.sendto(data[start:end], self.addr)
