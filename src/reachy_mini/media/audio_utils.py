"""Utility functions for audio handling.

This module provides helper functions for working with the ReSpeaker microphone array,
managing audio device configuration on Linux systems, and saving audio data to files
using GStreamer.

Example usage:
    >>> from reachy_mini.media.audio_utils import get_respeaker_card_number, has_reachymini_asoundrc, save_audio_to_wav
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
    >>>
    >>> # Save recorded audio to a WAV file (no soundfile dependency)
    >>> import numpy as np
    >>> audio = np.zeros((16000, 2), dtype=np.float32)
    >>> save_audio_to_wav(audio, samplerate=16000, filepath="output.wav")
"""

import logging
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

# Head-shell "boxy" resonance correction; see tools/speaker_eq_calibration/.
DEFAULT_SPEAKER_EQ_GAINS = [
    0.0,
    -13.21,
    -5.55,
    -4.28,
    -4.32,
    5.80,
    4.65,
    4.90,
    3.41,
    0.0,
]


def resolve_speaker_eq_gains() -> list[float]:
    """Return the 10 speaker-EQ band gains (dB), from daemon config or the default."""
    from reachy_mini.daemon.startup_app_config import get_speaker_eq_gains

    gains = get_speaker_eq_gains()
    return gains if gains is not None else list(DEFAULT_SPEAKER_EQ_GAINS)


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


def save_audio_to_wav(
    audio_data: npt.NDArray[np.float32],
    samplerate: int,
    filepath: str,
) -> None:
    """Write a float32 audio array to a WAV file using GStreamer.

    No external dependencies (e.g. ``soundfile``) are required — the WAV
    container is encoded by the GStreamer ``wavenc`` element.

    The pipeline used internally::

        appsrc → audioconvert → audioresample → wavenc → filesink

    Args:
        audio_data: Audio samples as a float32 array.  Shape ``(N,)`` for
            mono or ``(N, C)`` for interleaved multi-channel audio.
        samplerate: Sample rate in Hz.
        filepath: Destination file path (e.g. ``"output.wav"``).

    Raises:
        ImportError: If the ``gi`` / GStreamer Python bindings are not installed.
        RuntimeError: If GStreamer pipeline elements cannot be created, or if
            the pipeline does not complete within the timeout.

    Example::

        import numpy as np
        from reachy_mini.media.audio_utils import save_audio_to_wav

        audio = np.zeros((16000, 2), dtype=np.float32)
        save_audio_to_wav(audio, samplerate=16000, filepath="output.wav")

    """
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except ImportError as e:
        raise ImportError(
            "The 'gi' module is required for save_audio_to_wav but could not be "
            "imported. Please check the GStreamer installation."
        ) from e

    Gst.init([])

    # Normalise shape and infer channel count
    data = np.ascontiguousarray(audio_data, dtype=np.float32)
    if data.ndim == 1:
        channels = 1
    elif data.ndim == 2:
        channels = data.shape[1]
    else:
        raise ValueError(f"audio_data must be 1-D or 2-D, got shape {data.shape}")

    caps = Gst.Caps.from_string(
        f"audio/x-raw,format=F32LE,rate={samplerate},"
        f"channels={channels},layout=interleaved"
    )

    appsrc = Gst.ElementFactory.make("appsrc")
    audioconvert = Gst.ElementFactory.make("audioconvert")
    audioresample = Gst.ElementFactory.make("audioresample")
    wavenc = Gst.ElementFactory.make("wavenc")
    filesink = Gst.ElementFactory.make("filesink")

    if not all([appsrc, audioconvert, audioresample, wavenc, filesink]):
        raise RuntimeError("Failed to create GStreamer elements for save_audio_to_wav")

    appsrc.set_property("caps", caps)
    filesink.set_property("location", filepath)

    pipeline = Gst.Pipeline.new("wav-writer")
    for element in [appsrc, audioconvert, audioresample, wavenc, filesink]:
        pipeline.add(element)

    appsrc.link(audioconvert)
    audioconvert.link(audioresample)
    audioresample.link(wavenc)
    wavenc.link(filesink)

    pipeline.set_state(Gst.State.PLAYING)

    buf = Gst.Buffer.new_wrapped(data.tobytes())
    appsrc.emit("push-buffer", buf)
    appsrc.emit("end-of-stream")

    # Wait for EOS or ERROR (up to 5 seconds)
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(
        5 * Gst.SECOND,
        Gst.MessageType.EOS | Gst.MessageType.ERROR,
    )

    pipeline.set_state(Gst.State.NULL)

    if msg is None:
        raise RuntimeError(
            "save_audio_to_wav: GStreamer pipeline timed out waiting for EOS"
        )
    if msg.type == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        raise RuntimeError(
            f"save_audio_to_wav: GStreamer pipeline error: {err} — {debug}"
        )


def load_audio_mono(
    filepath: str, samplerate: int | None = None
) -> tuple[npt.NDArray[np.float64], int]:
    """Decode an audio file to mono float64 using GStreamer, optionally resampled.

    The counterpart of :func:`save_audio_to_wav`: it decodes anything
    ``decodebin`` handles (PCM WAV of any bit depth, IEEE-float WAV, OGG, MP3,
    FLAC, ...), so it reads back what that function writes.  Multi-channel input
    is downmixed to mono by ``audioconvert``.

    Decoding through GStreamer rather than by hand matters when the result is
    compared against audio the robot played: ``play_sound`` uses the same
    decoders, so any decoder quirk cancels out instead of showing up as a
    spurious mismatch.

    The pipeline used internally::

        filesrc → decodebin → audioconvert → audioresample → capsfilter → appsink

    Args:
        filepath: Path to any audio file GStreamer can decode.
        samplerate: Target rate in Hz.  ``None`` keeps the file's own rate.

    Returns:
        ``(samples, rate)`` — mono samples in ``[-1, 1]`` of shape ``(N,)``, and
        the rate they are at (the file's own when ``samplerate`` is ``None``).

    Raises:
        ImportError: If the ``gi`` / GStreamer Python bindings are not installed.
        RuntimeError: If the pipeline errors out or stalls.
        ValueError: If the file decodes to no audio at all.

    Example::

        from reachy_mini.media.audio_utils import load_audio_mono

        reference, rate = load_audio_mono("wake_up.wav", samplerate=16000)

    """
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except ImportError as e:
        raise ImportError(
            "The 'gi' module is required for load_audio_mono but could not be "
            "imported. Please check the GStreamer installation."
        ) from e

    Gst.init([])

    caps = "audio/x-raw,format=F32LE,channels=1,layout=interleaved"
    if samplerate is not None:
        caps += f",rate={samplerate}"

    # parse_launch rather than hand-built elements: decodebin exposes its audio
    # pad only once it has sniffed the format, and parse_launch does that
    # delayed linking for us.
    pipeline = Gst.parse_launch(
        "filesrc name=src ! decodebin ! audioconvert ! audioresample "
        f'! capsfilter caps="{caps}" ! appsink name=sink sync=false max-buffers=0'
    )
    pipeline.get_by_name("src").set_property("location", filepath)
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)

    chunks: list[npt.NDArray[np.float32]] = []
    rate = samplerate or 0
    try:
        while True:
            sample = sink.emit("try-pull-sample", 5 * Gst.SECOND)
            if sample is None:
                if sink.get_property("eos"):
                    break
                # Not EOS, so this is a stall or a failure to decode. Surface
                # the bus error when there is one — it names the actual cause
                # (missing file, unknown format, missing plugin).
                msg = pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
                if msg is not None:
                    err, debug = msg.parse_error()
                    raise RuntimeError(
                        f"load_audio_mono: GStreamer error on {filepath}: "
                        f"{err} — {debug}"
                    )
                raise RuntimeError(f"load_audio_mono: timed out decoding {filepath}")

            if not rate:
                rate = sample.get_caps().get_structure(0).get_value("rate")
            buffer = sample.get_buffer()
            ok, info = buffer.map(Gst.MapFlags.READ)
            if ok:
                # Copy: the mapped memory is only valid until unmap.
                chunks.append(np.frombuffer(info.data, dtype=np.float32).copy())
                buffer.unmap(info)
    finally:
        pipeline.set_state(Gst.State.NULL)

    if not chunks:
        raise ValueError(f"No audio decoded from {filepath}")

    return np.concatenate(chunks).astype(np.float64), int(rate)


def correlation_peak(
    capture: npt.NDArray[np.floating],
    reference: npt.NDArray[np.floating],
    samplerate: int,
) -> tuple[float, float]:
    """Locate ``reference`` inside ``capture`` by normalized cross-correlation.

    A matched filter: the peak height says how much of the reference's
    *waveform* is present, the peak position says when it starts.  Robust to
    level and delay, and — unlike :func:`spectral_cosine` — to heavy spectral
    coloration, which makes it the right presence detector for an *acoustic*
    path (small speaker, EQ, room, mic DSP).  Measured on a real robot
    speaker→mic loopback: ~0.28 with the sound present vs ~0.02-0.04 for echo
    cancellation eating it or unrelated noise.

    Both signals must be at ``samplerate`` and ``capture`` must be at least as
    long as ``reference``.

    Args:
        capture: The recording to search in, shape ``(N,)``.
        reference: The signal to look for, shape ``(M,)``, ``M <= N``.
        samplerate: Common sample rate in Hz (used only for the lag).

    Returns:
        ``(peak, lag_s)`` — peak of the normalized cross-correlation in
        ``[0, 1]``, and the reference's start offset within the capture in
        seconds.

    """
    from scipy.signal import fftconvolve

    c = np.asarray(capture, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    c = c - c.mean()
    r = r - r.mean()

    corr = fftconvolve(c, r[::-1], mode="valid")
    # Per-position energy of the capture window, so the normalization is local:
    # a loud noise burst elsewhere in the capture can't deflate the peak.
    window_energy = fftconvolve(c**2, np.ones(len(r)), mode="valid")
    ncc = corr / (np.linalg.norm(r) * np.sqrt(np.clip(window_energy, 1e-12, None)))

    k = int(np.argmax(np.abs(ncc)))
    return float(np.abs(ncc[k])), k / samplerate


def spectral_cosine(
    a: npt.NDArray[np.floating],
    b: npt.NDArray[np.floating],
    n: int | None = None,
) -> float:
    """Cosine similarity of the Hann-windowed magnitude spectra of two signals.

    Frequency-domain so it's timing-invariant — a partial capture or a start
    offset doesn't matter, only whether the same sound is present.

    Best suited to digitally clean paths (measured 0.75-0.84 on the virtual
    audio loopback vs ~0.10 for noise).  On a real *acoustic* path the
    speaker/EQ/room coloration compresses the separation to the point of
    uselessness (~0.18 present vs ~0.12 for white noise, measured on-robot) —
    use :func:`correlation_peak` there instead.

    Both signals must be at the **same sample rate**.  The FFT size is shared,
    so a given frequency lands in a different bin at a different rate:
    identical audio compared across 16 kHz and 44.1 kHz scores near zero.
    Resample one to the other's rate first
    (``load_audio_mono(..., samplerate=)``).

    The reference also needs spectral *structure* for the score to
    discriminate.  Speech and music do; a sweep or noise does not — two
    unrelated broadband signals score ~0.5, so this is the wrong metric for
    those.

    Args:
        a: First signal, shape ``(N,)``.
        b: Second signal, shape ``(M,)``.  Need not match ``a`` in length, but
            must match in sample rate.
        n: FFT size.  Defaults to the next power of two covering the longer
            signal, so neither is truncated — an ``n`` shorter than a signal
            silently crops it to its first ``n`` samples, making the score
            depend on *when* the sound occurs, exactly what this metric is
            meant to be invariant to.

    Returns:
        Similarity in ``[0, 1]``.

    """
    if n is None:
        n = 1 << (max(len(a), len(b)) - 1).bit_length()

    def spectrum(x: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        xf = np.asarray(x, dtype=np.float64)
        mag = np.abs(np.fft.rfft(xf * np.hanning(len(xf)), n))
        return np.asarray(mag / (np.linalg.norm(mag) + 1e-9), dtype=np.float64)

    return float(np.dot(spectrum(a), spectrum(b)))
