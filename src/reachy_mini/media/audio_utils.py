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
import re
import subprocess
from pathlib import Path

DEFAULT_DEVICE_NAMES = ["reachy mini audio", "respeaker"]


def _process_card_number_output(
    output: str, device_names: list[str] = DEFAULT_DEVICE_NAMES
) -> int:
    """Process the output of 'arecord -l' to find a sound card matching any of the given device names.

    Args:
        output (str): The output string from the 'arecord -l' command containing
                     information about available audio devices.
        device_names (list[str]): List of device name patterns to search for (case-insensitive).
                     Defaults to DEFAULT_DEVICE_NAMES (["reachy mini audio", "respeaker"]).

    Returns:
        int: The card number of the first matching device found,
             or 0 if no match is found (default sound card).

    Note:
        This function parses the output of 'arecord -l' to identify audio devices.
        Device names are matched in order, so earlier entries in the list have priority.

    Example:
        >>> output = "card 1: ReachyMiniAudio [reachy mini audio], device 0: USB Audio [USB Audio]"
        >>> card_num = _process_card_number_output(output)
        >>> print(f"Detected card: {card_num}")
        >>>
        >>> # Search for custom devices
        >>> card_num = _process_card_number_output(output, ["my device", "other device"])

    """
    lines = output.split("\n")
    logging.warning(f"{lines}")
    for device_name in device_names:
        for line in lines:
            if device_name.lower() in line.lower():
                card_number = line.split(" ")[1].split(":")[0]
                logging.debug(f"Found '{device_name}' sound card: {card_number}")
                return int(card_number)

    logging.warning(f"No sound card matching {device_names} found. Returning default card")
    return 0  # default sound card


def get_respeaker_card_number(device_names: list[str] = DEFAULT_DEVICE_NAMES) -> int:
    """Return the card number of a sound card matching any of the given device names, or 0 if not found.

    Args:
        device_names (list[str]): List of device name patterns to search for (case-insensitive).
                     Defaults to DEFAULT_DEVICE_NAMES (["reachy mini audio", "respeaker"]).

    Returns:
        int: The card number of the detected audio device.
             Returns 0 if no specific device is found (uses default sound card),
             or -1 if there's an error running the detection command.

    Note:
        This function runs 'arecord -l' to list available audio capture devices
        and processes the output to find matching devices.
        It's primarily used on Linux systems with ALSA audio configuration.

        The function returns:
        - Positive integer: Card number of detected audio device
        - 0: No matching device found, using default sound card
        - -1: Error occurred while trying to detect audio devices

    Example:
        >>> card_num = get_respeaker_card_number()
        >>> if card_num > 0:
        ...     print(f"Using Reachy Mini Audio card {card_num}")
        ... elif card_num == 0:
        ...     print("Using default sound card")
        ... else:
        ...     print("Error detecting audio devices")
        >>>
        >>> # Search for custom devices
        >>> card_num = get_respeaker_card_number(["my device", "other device"])

    """
    try:
        result = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True, check=True
        )
        output = result.stdout

        return _process_card_number_output(output, device_names)

    except subprocess.CalledProcessError as e:
        logging.error(f"Cannot find sound card: {e}")
        return -1


def _parse_gst_node_name(output: str, device_names: list[str] = DEFAULT_DEVICE_NAMES) -> str | None:
    """Parse gst-device-monitor-1.0 output to find the node.name for a given device name.

    Args:
        output: The output string from gst-device-monitor-1.0 Audio command.
        device_names: List of device name patterns to search for (case-insensitive).

    Returns:
        The node.name if found, or None if no match.
    """
    device_blocks = output.split("Device found:")

    for device_name in device_names:
        for block in device_blocks:
            # Check if this block contains the device name
            name_match = re.search(r"^\s*name\s*:\s*(.+)$", block, re.MULTILINE)
            if name_match:
                found_name = name_match.group(1).strip()
                if device_name.lower() in found_name.lower():
                    # Found the device, now extract node.name
                    node_match = re.search(r"node\.name\s*=\s*\"?([^\"\n]+)\"?", block)
                    if node_match:
                        node_name = node_match.group(1).strip()
                        logging.debug(f"Found node.name for '{device_name}': {node_name}")
                        return node_name

    logging.warning(f"No node.name found for devices {device_names}")
    return None


def get_respeaker_node_name(device_names: list[str] = DEFAULT_DEVICE_NAMES) -> str | None:
    """Return the node.name of a device matching any of the given device names, or None if not found.

    Args:
        device_names: List of device name patterns to search for (case-insensitive).
                     Defaults to DEFAULT_DEVICE_NAMES (["reachy mini audio", "respeaker"]).

    Returns:
        The node.name if found, or None if not found or error.

    Example:
        >>> node = get_respeaker_node_name()
        >>> if node:
        ...     print(f"Node name: {node}")
    """
    try:
        result = subprocess.run(
            ["gst-device-monitor-1.0", "Audio"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return _parse_gst_node_name(result.stdout, device_names)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logging.error(f"gst-device-monitor-1.0 failed: {e}")
    return None


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
        >>> if has_reachymini_asoundrc():
        ...     print("Reachy Mini audio configuration is properly set up")
        ... else:
        ...     print("Need to configure Reachy Mini audio devices")
        ...     write_asoundrc_to_home()  # Create the configuration

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
        period_size 1024
        buffer_size 4096
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
        period_size 1024
        buffer_size 4096
    }}
}}
"""
    asoundrc_path = Path.home().joinpath(".asoundrc")
    with open(asoundrc_path, "w") as f:
        f.write(asoundrc_content)
