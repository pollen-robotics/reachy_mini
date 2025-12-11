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
    import sys

    app = FaultyApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
        sys.exit(0)
    except Exception as e:
        # Log the error and exit with non-zero code
        print(f"App crashed with error: {e}", file=sys.stderr)
        sys.exit(1)

