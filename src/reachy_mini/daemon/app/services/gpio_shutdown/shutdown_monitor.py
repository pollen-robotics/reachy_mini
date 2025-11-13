"""Monitor GPIO24 for shutdown signal."""

from signal import pause

from gpiozero import Button

shutdown_button = Button(24)

# log_file = "/var/log/shutdown_button.log"
log_file = "/vens/shutdown_button.log"


def released() -> None:
    """Handle shutdown button released."""
    print("Shutdown button released! Shutting down...")
    with open(log_file) as f:
        f.write("Shutdown button released! Shutting down...\n")


def pressed() -> None:
    """Handle shutdown button pressed."""
    print("Shutdown button pressed!")
    with open(log_file) as f:
        f.write("Shutdown button pressed!\n")


shutdown_button.when_released = released
shutdown_button.when_pressed = pressed

print("Monitoring GPIO24 for shutdown signal...")
pause()
