import socket
from threading import Thread, Lock

from reachy_mini.io.abstract import AbstractServer
from reachy_mini.command import ReachyMiniCommand


class SocketServer(AbstractServer):
    def __init__(self, host="0.0.0.0", port=1234):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        # Store the latest command
        self._latest_command = ReachyMiniCommand.default()
        self._lock = Lock()
        self._running = False

    def start(self):
        if not self._running:
            self._running = True
            Thread(target=self._client_handler, daemon=True).start()

    def stop(self):
        pass

    def _client_handler(self):
        while self._running:
            print("SocketServer: Waiting for connection on port", self.port)
            try:
                conn, address = self.server_socket.accept()
                print(f"SocketServer: Client connected from {address}")
                with conn:
                    while True:
                        try:
                            data = conn.recv(4096)
                            if not data:
                                print("SocketServer: Client disconnected")
                                break

                            with self._lock:
                                self._latest_command.update_with(
                                    ReachyMiniCommand.from_json(data),
                                )

                        except (
                            ConnectionResetError,
                            EOFError,
                        ) as e:
                            print(f"SocketServer: Client error: {e}")
                            break
            except Exception as e:
                print(f"SocketServer: Server error: {e}")

    def get_latest_command(self) -> ReachyMiniCommand:
        with self._lock:
            # Return a copy to avoid race conditions
            return self._latest_command.copy()
