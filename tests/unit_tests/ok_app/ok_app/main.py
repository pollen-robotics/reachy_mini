import threading
import time

from reachy_mini import ReachyMini, ReachyMiniApp


class OkApp(ReachyMiniApp):
    def __init__(self):
        super().__init__()
        # Override media backend for testing
        self.media_backend = "no_media"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        while not stop_event.is_set():
            time.sleep(0.5)


if __name__ == "__main__":
    app = OkApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
