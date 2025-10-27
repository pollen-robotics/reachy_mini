import threading
import time

from reachy_mini import ReachyMini, ReachyMiniApp


class OkApp(ReachyMiniApp):
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        while not stop_event.is_set():
            time.sleep(0.5)