"""Utility functions for audio handling, specifically for detecting the ReSpeaker sound card.

This module provides helper functions for working with the ReSpeaker microphone array
and managing audio device configuration on Linux systems. It includes functions for
detecting sound cards, checking configuration files, and managing ALSA configuration.

Example usage:
    >>> from reachy_mini.media.audio_utils import get_respeaker_card_number, has_reachymini_asoundrc
    >>>
    >>> # Get the ReSpeaker card number
    >>> card_num = get_respeaker_card_number()
    >>> print(f"ReSpeaker card number: {card_num}")
    >>>
    >>> # Check if .asoundrc is properly configured
    >>> if has_reachymini_asoundrc():
    ...     print("Reachy Mini audio configuration is properly set up")
    ... else:
    ...     print("Need to configure audio devices")
"""

import logging
import subprocess
from pathlib import Path


def _process_card_number_output(output: str) -> int:
    """Process the output of 'arecord -l' to find the ReSpeaker or Reachy Mini Audio card number.

    Args:
        output (str): The output string from the 'arecord -l' command containing
                     information about available audio devices.

    Returns:
        int: The card number of the detected Reachy Mini Audio or ReSpeaker device,
             or 0 if neither is found (default sound card).

    Note:
        This function parses the output of 'arecord -l' to identify Reachy Mini
        Audio or ReSpeaker devices. It prefers Reachy Mini Audio devices and
        warns if only a ReSpeaker device is found (indicating firmware update needed).

    Example:
        ```python
        output = "card 1: ReachyMiniAudio [reachy mini audio], device 0: USB Audio [USB Audio]"
        card_num = _process_card_number_output(output)
        print(f"Detected card: {card_num}")
        ```

    """
    lines = output.split("\n")
    for line in lines:
        if "reachy mini audio" in line.lower():
            card_number = line.split(" ")[1].split(":")[0]
            logging.debug(f"Found Reachy Mini Audio sound card: {card_number}")
            return int(card_number)
        elif "respeaker" in line.lower():
            card_number = line.split(" ")[1].split(":")[0]
            logging.warning(
                f"Found ReSpeaker sound card: {card_number}. Please update firmware!"
            )
            return int(card_number)

    logging.warning("Reachy Mini Audio sound card not found. Returning default card")
    return 0  # default sound card


def get_respeaker_card_number() -> int:
    """Return the card number of the ReSpeaker sound card, or 0 if not found.

    Returns:
        int: The card number of the detected ReSpeaker/Reachy Mini Audio device.
             Returns 0 if no specific device is found (uses default sound card),
             or -1 if there's an error running the detection command.

    Note:
        This function runs 'arecord -l' to list available audio capture devices
        and processes the output to find Reachy Mini Audio or ReSpeaker devices.
        It's primarily used on Linux systems with ALSA audio configuration.

        The function returns:
        - Positive integer: Card number of detected Reachy Mini Audio device
        - 0: No Reachy Mini Audio device found, using default sound card
        - -1: Error occurred while trying to detect audio devices

    Example:
        ```python
        card_num = get_respeaker_card_number()
        if card_num > 0:
            print(f"Using Reachy Mini Audio card {card_num}")
        elif card_num == 0:
            print("Using default sound card")
        else:
            print("Error detecting audio devices")
        ```

    """
    try:
        result = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True, check=True
        )
        output = result.stdout

        return _process_card_number_output(output)

    except subprocess.CalledProcessError as e:
        logging.error(f"Cannot find sound card: {e}")
        return -1


def has_reachymini_asoundrc() -> bool:
    """Check if ~/.asoundrc exists and contains both reachymini_audio_sink and reachymini_audio_src.

    Returns:
        bool: True if ~/.asoundrc exists and contains the required Reachy Mini
             audio configuration entries, False otherwise.

    Note:
        This function checks for the presence of the ALSA configuration file
        ~/.asoundrc and verifies that it contains the necessary configuration
        entries for Reachy Mini audio devices (reachymini_audio_sink and
        reachymini_audio_src). These entries are required for proper audio
        routing and device management.

    Example:
        ```python
        if has_reachymini_asoundrc():
            print("Reachy Mini audio configuration is properly set up")
        else:
            print("Need to configure Reachy Mini audio devices")
            write_asoundrc_to_home()  # Create the configuration
        ```

    """
    asoundrc_path = Path.home().joinpath(".asoundrc")
    if not asoundrc_path.exists():
        return False
    content = asoundrc_path.read_text(errors="ignore")
    return "reachymini_audio_sink" in content and "reachymini_audio_src" in content


def check_reachymini_asoundrc() -> bool:
    """Check if ~/.asoundrc exists and is correctly configured for Reachy Mini Audio."""
    asoundrc_path = Path.home().joinpath(".asoundrc")
    if not asoundrc_path.exists():
        return False
    content = asoundrc_path.read_text(errors="ignore")
    card_id = get_respeaker_card_number()
    # Check for both sink and src
    if not ("reachymini_audio_sink" in content and "reachymini_audio_src" in content):
        return False
    # Check that the card number in .asoundrc matches the detected card_id
    import re

    card_numbers = set(re.findall(r"card\s+(\d+)", content))
    if str(card_id) not in card_numbers:
        return False
    return True


def write_asoundrc_to_home() -> None:
    """Write the .asoundrc file with Reachy Mini audio configuration to the user's home directory.

    This function creates an ALSA configuration file (.asoundrc) in the user's home directory
    that configures the ReSpeaker sound card for proper audio routing and multi-client support.
    The configuration enables simultaneous audio input and output access, which is essential
    for the Reachy Mini Wireless version's audio functionality.

    The generated configuration includes:
        - Default audio device settings pointing to the ReSpeaker sound card
        - dmix plugin for multi-client audio output (reachymini_audio_sink)
        - dsnoop plugin for multi-client audio input (reachymini_audio_src)
        - Proper buffer and sample rate settings for optimal performance

    Note:
    This function automatically detects the ReSpeaker card number and creates a configuration
    tailored to the detected hardware. It is primarily used for the Reachy Mini Wireless version.

    The configuration file will be created at ~/.asoundrc and will overwrite any existing file
    with the same name. Existing audio configurations should be backed up before calling this function.


    """
    card_id = get_respeaker_card_number()
    asoundrc_content = f"""
pcm.!default {{
    type hw
    card {card_id}
}}

ctl.!default {{
    type hw
    card {card_id}
}}

pcm.reachymini_audio_sink {{
    type dmix
    ipc_key 4241
    slave {{
        pcm "hw:{card_id},0"
        channels 2
        period_size 256
        buffer_size 1024
        rate 16000
    }}
    bindings {{
        0 0
        1 1
    }}
}}

pcm.reachymini_audio_src {{
    type dsnoop
    ipc_key 4242
    slave {{
        pcm "hw:{card_id},0"
        channels 2
        rate 16000
        period_size 256
        buffer_size 1024
    }}
}}
"""
    asoundrc_path = Path.home().joinpath(".asoundrc")
    with open(asoundrc_path, "w") as f:
        f.write(asoundrc_content)
