"""Utility functions for audio handling, specifically for detecting the ReSpeaker sound card."""

import logging
import subprocess
from pathlib import Path


def get_respeaker_card_number() -> int:
    """Return the card number of the ReSpeaker sound card, or 0 if not found."""
    try:
        result = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True, check=True
        )
        output = result.stdout

        lines = output.split("\n")
        for line in lines:
            if "reachy mini audio" in line.lower() and "card" in line:
                card_number = line.split(":")[0].split("card ")[1].strip()
                logging.debug(f"Found Reachy Mini Audio sound card: {card_number}")
                return int(card_number)
            elif "respeaker" in line.lower() and "card" in line:
                card_number = line.split(":")[0].split("card ")[1].strip()
                logging.warning(
                    f"Found ReSpeaker sound card: {card_number}. Please update firmware!"
                )
                return int(card_number)

        logging.warning(
            "Reachy Mini Audio sound card not found. Returning default card"
        )
        return 0  # default sound card

    except subprocess.CalledProcessError as e:
        logging.error(f"Cannot find sound card: {e}")
        return -1


def has_reachymini_asoundrc() -> bool:
    """Check if ~/.asoundrc exists and contains both reachymini_audio_sink and reachymini_audio_src."""
    asoundrc_path = Path.home().joinpath(".asoundrc")
    if not asoundrc_path.exists():
        return False
    content = asoundrc_path.read_text(errors="ignore")
    return "reachymini_audio_sink" in content and "reachymini_audio_src" in content
