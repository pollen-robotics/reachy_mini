"""Monitor GPIO24 for shutdown signal."""

from signal import pause
from subprocess import call

from gpiozero import Button

shutdown_button = Button(23, pull_up=False)


def released() -> None:
    """Handle shutdown button released."""
    print("Shutdown button released, shutting down...")
    call(["sudo", "shutdown", "-h", "now"])


shutdown_button.when_released = released

print("Monitoring GPIO23 for shutdown signal...")
pause()
