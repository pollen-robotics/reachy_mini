"""Monitor GPIO24 for shutdown signal."""

import time
from signal import pause
from subprocess import call

from gpiozero import Button

shutdown_button = Button(23, pull_up=False)


def released() -> None:
    """Handle shutdown button released."""
    for _ in range(200):
        time.sleep(0.001)

        if shutdown_button.is_pressed:
            # probably just a bounce, ignore
            return

    print("Shutdown button released, shutting down...")
    call(["sudo", "shutdown", "-h", "now"])


shutdown_button.when_released = released

print("Monitoring GPIO23 for shutdown signal...")
pause()
