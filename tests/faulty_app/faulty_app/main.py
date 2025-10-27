import threading

from reachy_mini import ReachyMini, ReachyMiniApp


class FaultyApp(ReachyMiniApp):
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        raise RuntimeError("This is a faulty app for testing purposes.")

