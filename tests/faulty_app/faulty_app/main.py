import threading

from reachy_mini import ReachyMini, ReachyMiniApp


class FaultyApp(ReachyMiniApp):
    def __init__(self):
        super().__init__()
        # Override media backend for testing
        self.media_backend = "no_media"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        raise RuntimeError("This is a faulty app for testing purposes.")


if __name__ == "__main__":
    app = FaultyApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()

