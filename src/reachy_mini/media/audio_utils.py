"""Utility functions for audio handling, specifically for detecting the ReSpeaker sound card."""

import logging
import subprocess


def get_respeaker_card_number() -> int:
    """Return the card number of the ReSpeaker sound card, or 0 if not found."""
    try:
        result = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True, check=True
        )
        output = result.stdout

        lines = output.split("\n")
        for line in lines:
            if "respeaker" in line.lower() and "card" in line:
                card_number = line.split(":")[0].split("card ")[1].strip()
                logging.debug(f"Found ReSpeaker sound card: {card_number}")
                return int(card_number)

        logging.warning("ReSpeaker sound card not found. Returning default card")
        return 0  # default sound card

    except subprocess.CalledProcessError as e:
        logging.error(f"Cannot find sound card: {e}")
        return 0
