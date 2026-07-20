"""Direction of Arrival (DoA) estimation via the ReSpeaker microphone array.

This module wraps the ReSpeaker USB device to provide Direction of Arrival
readings.  It is used by ``GStreamerAudio`` and ``MediaManager`` to expose
DoA data without coupling it to a specific audio backend.

The spatial angle is given in radians:
    0 radians is left, π/2 radians is front/back, π radians is right.

Note:
    The microphone array requires firmware version 2.1.0 or higher.
    The firmware is located in ``src/reachy_mini/assets/firmware/*.bin``.
    Refer to https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/#update-firmware
    for the upgrade process.

"""

import logging
from threading import Lock
from typing import Optional

from reachy_mini.media.audio_control_utils import ReSpeaker, init_respeaker_usb

logger = logging.getLogger(__name__)


class AudioDoA:
    """Direction of Arrival helper backed by a ReSpeaker USB device.

    Attributes:
        _respeaker: The underlying ReSpeaker device, or ``None`` if no
            compatible hardware was detected on first use.

    Example::

        doa = AudioDoA()
        result = doa.get_DoA()
        if result is not None:
            angle, speech = result
            print(f"Sound at {angle:.2f} rad, speech={speech}")
        doa.close()

    """

    def __init__(self) -> None:
        """Initialize the DoA helper without probing USB until first use."""
        self._respeaker: Optional[ReSpeaker] = None
        self._initialization_attempted = False
        self._closed = False
        self._lock = Lock()

    def get_DoA(self) -> tuple[float, bool] | None:
        """Read the current Direction of Arrival from the ReSpeaker.

        The USB device is discovered on the first call and reused thereafter.

        Returns:
            A tuple ``(angle_radians, speech_detected)`` or ``None`` when
            the device is not available or the read fails.

        """
        with self._lock:
            if self._closed:
                return None
            if not self._initialization_attempted:
                self._respeaker = init_respeaker_usb()
                self._initialization_attempted = True
            if self._respeaker is None:
                return None

            result = self._respeaker.read("DOA_VALUE_RADIANS")
            if result is None:
                return None
            return float(result[0]), bool(result[1])

    def close(self) -> None:
        """Release the USB resource."""
        with self._lock:
            self._closed = True
            if self._respeaker:
                self._respeaker.close()
                self._respeaker = None


def main() -> None:
    """Poll Direction of Arrival at ~10 Hz and print results."""
    import math
    import time

    logging.basicConfig(level=logging.INFO)

    doa = AudioDoA()
    result = doa.get_DoA()
    if doa._respeaker is None:
        print("No ReSpeaker device found. Exiting.")
        return

    print("Reading DoA — press Ctrl+C to stop.\n")
    try:
        while True:
            if result is not None:
                angle, speech = result
                print(
                    f"angle={math.degrees(angle):6.1f}°  "
                    f"({angle:.2f} rad)  "
                    f"speech={speech}"
                )
            else:
                print("no reading")
            time.sleep(0.1)
            result = doa.get_DoA()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        doa.close()


if __name__ == "__main__":
    main()
