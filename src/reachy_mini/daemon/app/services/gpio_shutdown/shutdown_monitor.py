"""Monitor GPIO24 for shutdown signal."""

from signal import pause
from subprocess import call

from gpiozero import Button

shutdown_button = Button(24, pull_up=False)

# log_file = "/var/log/shutdown_button.log"
log_file = "/vens/shutdown_button.log"


def released() -> None:
    """Handle shutdown button released."""
    call(["sudo", "systemctl", "stop", "reachy-mini-daemon"])
    call(["sudo", "shutdown", "-h", "now"])


shutdown_button.when_released = released

print("Monitoring GPIO24 for shutdown signal...")
pause()
