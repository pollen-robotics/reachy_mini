"""Wake-word ("wake up") detection for the Reachy Mini daemon.

A self-contained GStreamer pipeline (mic -> appsink) feeds 16 kHz mono int16
audio to a nanowakeword model. When the phrase is detected while listening,
``on_detection`` fires (on the pull thread) to wake the robot.

The detector listens only while the robot is asleep: the daemon calls
:meth:`start` on ``goto_sleep`` and :meth:`stop` on ``wake_up``. Each start
builds a fresh pipeline and each stop tears it down (switches are rare, so we
recreate rather than pause), so it costs ~zero CPU while awake.

Both ``nanowakeword`` and ``huggingface_hub`` are imported lazily so the daemon
runs without the optional ``wake-word`` extra installed.
"""

import logging
import platform
import threading
import time
from typing import Any, Callable, Optional

import gi
import numpy as np
import numpy.typing as npt

from reachy_mini.media.audio_utils import has_reachymini_asoundrc
from reachy_mini.media.device_detection import get_audio_device

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import Gst, GstApp  # noqa: E402, F401

REPO_ID = "pollen-robotics/nanowakeword-wake-up"
LITE_MODEL = "wake_up_v1_lite.onnx"

SAMPLE_RATE = 16_000
CHUNK = 1280  # 80 ms at 16 kHz, the model's frame size


class WakeWordDetector:
    """Listen for the "wake up" phrase and fire a callback on detection."""

    def __init__(
        self,
        on_detection: Callable[[], None],
        threshold: float = 0.85,
        cooldown_s: float = 1.0,
        log_level: str = "INFO",
    ) -> None:
        """Create a detector.

        Args:
            on_detection: Called (on the pull thread) when the wake word is
                heard while listening. Must return promptly.
            threshold: Detection score threshold (0-1).
            cooldown_s: Minimum seconds between two detections.
            log_level: Logging level.

        """
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

        self._on_detection = on_detection
        self._threshold = threshold
        self._cooldown_s = cooldown_s

        self._interpreter: Optional[Any] = None
        self._pipeline: Optional[Gst.Pipeline] = None
        self._appsink: Optional[Gst.Element] = None
        self._bus: Optional[Gst.Bus] = None
        self._thread: Optional[threading.Thread] = None
        self._got_audio = False  # logged once per start when samples start flowing

        self._lock = threading.Lock()
        self._running = False  # pipeline PLAYING + pull thread alive
        self._listening = False  # gate: only fire detections while asleep
        self._last_detection = 0.0
        self._buf: npt.NDArray[np.int16] = np.empty(0, dtype=np.int16)

        Gst.init(None)

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Build a fresh pipeline and begin listening.

        Called on ``goto_sleep`` and on boot when the robot starts asleep.
        """
        with self._lock:
            if self._running:
                return
            if not self._ensure_interpreter():
                return
            if not self._start_pipeline():
                return
            self._listening = True
            self._logger.info("Wake-word detector listening.")

    def stop(self) -> None:
        """Stop listening and tear the pipeline down. Called on ``wake_up``."""
        with self._lock:
            self._listening = False
            self._stop_pipeline()

    def close(self) -> None:
        """Shut the detector down permanently."""
        self.stop()

    def preload(self) -> bool:
        """Download/load the model ahead of time (blocking); return success.

        Called off the event loop at startup so later :meth:`start` calls
        (which run on the loop thread from the goto_sleep hook) never block.
        """
        with self._lock:
            return self._ensure_interpreter()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _ensure_interpreter(self) -> bool:
        """Lazily download and load the model; return False if unavailable."""
        if self._interpreter is not None:
            return True
        try:
            from huggingface_hub import hf_hub_download
            from nanowakeword import NanoInterpreter
        except ImportError as e:
            self._logger.warning(
                f"nanowakeword not installed ({e}); wake-word disabled. "
                "Install the 'wake-word' extra to enable it."
            )
            return False
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename=LITE_MODEL)
            self._interpreter = NanoInterpreter.load_model(path)
        except Exception as e:
            self._logger.warning(f"Failed to load wake-word model: {e}")
            return False
        self._logger.info(f"Loaded wake-word model {LITE_MODEL}.")
        return True

    def score_audio(self, audio: npt.NDArray[np.int16]) -> float:
        """Return the max detection score over an int16 audio buffer.

        Testable path that bypasses GStreamer. Requires the model to be
        loaded (call :meth:`preload` first).
        """
        assert self._interpreter is not None, "interpreter not loaded"
        self._interpreter.reset()
        max_score = 0.0
        for i in range(0, len(audio) - CHUNK + 1, CHUNK):
            score = self._interpreter.predict(audio[i : i + CHUNK]).score
            max_score = max(max_score, score)
        return max_score

    # ------------------------------------------------------------------
    # GStreamer pipeline
    # ------------------------------------------------------------------
    def _build_source(self) -> Optional[Gst.Element]:
        """Build a platform-aware microphone source element.

        The device id comes from :func:`get_audio_device` (the shared
        detection entry point); only the element construction is local.
        """
        # Wireless CM4 exposes the Reachy Mini Audio card through an .asoundrc
        # alias; alsasrc on it avoids PipeWire clock issues.
        if has_reachymini_asoundrc():
            src = Gst.ElementFactory.make("alsasrc")
            src.set_property("device", "reachymini_audio_src")
            self._logger.info("Using ALSA device reachymini_audio_src for capture.")
            return src

        device_id = get_audio_device("Source")
        if device_id is None:
            self._logger.warning(
                "No Reachy Mini audio card found; using default audio source."
            )
            return Gst.ElementFactory.make("autoaudiosrc")

        system = platform.system()
        if system == "Windows":
            src = Gst.ElementFactory.make("wasapi2src")
            src.set_property("device", device_id)
        elif system == "Darwin":
            src = Gst.ElementFactory.make("osxaudiosrc")
            src.set_property("unique-id", device_id)
        else:
            src = Gst.ElementFactory.make("pulsesrc")
            src.set_property("device", device_id)
        self._logger.info(f"Using audio device {device_id} for capture.")
        return src

    def _start_pipeline(self) -> bool:
        """Build and start the capture pipeline; return False on failure."""
        src = self._build_source()
        if src is None:
            self._logger.warning("No audio source; wake-word detector disabled.")
            return False

        pipeline = Gst.Pipeline.new("wake_word")
        convert = Gst.ElementFactory.make("audioconvert")
        resample = Gst.ElementFactory.make("audioresample")
        appsink = Gst.ElementFactory.make("appsink")
        appsink.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=S16LE,rate={SAMPLE_RATE},"
                "channels=1,layout=interleaved"
            ),
        )
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 10)
        appsink.set_property("sync", False)

        for el in (src, convert, resample, appsink):
            pipeline.add(el)
        src.link(convert)
        convert.link(resample)
        resample.link(appsink)

        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._logger.warning("Failed to start wake-word pipeline.")
            pipeline.set_state(Gst.State.NULL)
            return False

        self._pipeline = pipeline
        self._appsink = appsink
        self._bus = pipeline.get_bus()
        self._buf = np.empty(0, dtype=np.int16)
        self._got_audio = False
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _stop_pipeline(self) -> None:
        """Stop the pull thread and tear the pipeline down."""
        self._running = False
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        self._bus = None
        self._appsink = None

    def _run(self) -> None:
        """Pull audio samples and run detection until stopped."""
        appsink = self._appsink
        assert appsink is not None
        while self._running:
            self._drain_bus()
            sample = appsink.try_pull_sample(int(0.1 * Gst.SECOND))
            if sample is None:
                continue
            if not self._got_audio:
                self._got_audio = True
                self._logger.info("Wake-word detector receiving audio.")
            buf = sample.get_buffer()
            data = buf.extract_dup(0, buf.get_size())
            self._process(np.frombuffer(data, dtype=np.int16))

    def _drain_bus(self) -> None:
        """Log any GStreamer ERROR/WARNING so a dead capture isn't silent.

        Without this, an ALSA source that fails to stream (e.g. the device
        was grabbed by an app) leaves the detector logging "listening" while
        no samples ever arrive.
        """
        bus = self._bus
        if bus is None:
            return
        msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.WARNING)
        while msg is not None:
            err, debug = (
                msg.parse_error()
                if msg.type == Gst.MessageType.ERROR
                else msg.parse_warning()
            )
            level = "error" if msg.type == Gst.MessageType.ERROR else "warning"
            self._logger.warning(
                "Wake-word pipeline %s: %s (%s)", level, err.message, debug
            )
            msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.WARNING)

    def _process(self, pcm: npt.NDArray[np.int16]) -> None:
        """Accumulate into CHUNK-sized frames and run the model."""
        assert self._interpreter is not None
        self._buf = np.concatenate((self._buf, pcm))
        while len(self._buf) >= CHUNK:
            chunk = self._buf[:CHUNK]
            self._buf = self._buf[CHUNK:]
            score = self._interpreter.predict(chunk).score
            if score <= self._threshold or not self._listening:
                continue
            now = time.monotonic()
            if now - self._last_detection < self._cooldown_s:
                continue
            self._last_detection = now
            self._interpreter.reset()
            self._buf = np.empty(0, dtype=np.int16)
            self._logger.info(f"Wake word detected (score={score:.3f}).")
            try:
                self._on_detection()
            except Exception:
                self._logger.exception("wake-word on_detection callback failed")
