from abc import ABC, abstractmethod
import threading

from reachy_mini.reachy_mini import ReachyMini


class ReachyMiniApp(ABC):
    def __init__(self):
        self.stop_event = threading.Event()

    def wrapped_run(self):
        try:
            with ReachyMini() as reachy_mini:
                self.run(reachy_mini, self.stop_event)
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    @abstractmethod
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Run the main logic of the app."""
        pass

    def stop(self):
        """Stop the app gracefully."""
        self.stop_event.set()
        print("App is stopping...")
