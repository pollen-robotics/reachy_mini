import os
import tempfile
import time
import wave

import numpy as np
import pytest
from scipy.signal import resample_poly

from reachy_mini.media.audio_utils import _process_card_number_output, save_audio_to_wav
from reachy_mini.media.media_manager import MediaBackend, MediaManager
from reachy_mini.utils.constants import ASSETS_ROOT_PATH

SIGNALING_HOST = "reachy-mini.local"


def _spectral_cosine(a: np.ndarray, b: np.ndarray, n: int = 8192) -> float:
    """Cosine similarity of the Hann-windowed magnitude spectra of two signals.

    Frequency-domain so it's timing-invariant — a partial capture or a start
    offset doesn't matter, only whether the same sound is present.
    """

    def spectrum(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        mag = np.abs(np.fft.rfft(x * np.hanning(len(x)), n))
        return mag / (np.linalg.norm(mag) + 1e-9)

    return float(np.dot(spectrum(a), spectrum(b)))

# Audio-capable backends to test
AUDIO_BACKENDS = [
    pytest.param(MediaBackend.LOCAL),
    pytest.param(MediaBackend.WEBRTC, marks=pytest.mark.wireless),
]


@pytest.mark.audio
@pytest.mark.parametrize("backend", AUDIO_BACKENDS)
def test_play_sound(backend: MediaBackend) -> None:
    """Test playing a sound with the given backend."""
    media = MediaManager(
        backend=backend,
        signalling_host=SIGNALING_HOST
        if backend == MediaBackend.WEBRTC
        else "localhost",
    )
    # Use a short sound file present in your assets directory
    sound_file = "wake_up.wav"  # Change to a valid file if needed
    media.play_sound(sound_file)
    print(f"Playing sound with {backend.value} backend...")
    # Wait a bit to let the sound play (non-blocking backend)
    time.sleep(2)
    # No assertion: test passes if no exception is raised.
    # Sound should be audible if the audio device is correctly set up.
    media.close()


@pytest.mark.audio
@pytest.mark.parametrize("backend", AUDIO_BACKENDS)
def test_stop_play_sound(backend: MediaBackend) -> None:
    """Test that stop_playing() actually stops a sound started by play_sound()."""
    media = MediaManager(
        backend=backend,
        signalling_host=SIGNALING_HOST
        if backend == MediaBackend.WEBRTC
        else "localhost",
    )
    sound_file = "confused1.wav"
    media.play_sound(sound_file)
    print(f"Playing sound with {backend.value} backend...")
    time.sleep(1)
    media.stop_playing()
    print(f"Stopped playing sound with {backend.value} backend.")
    if backend == MediaBackend.LOCAL:
        # GstWebRTCClient plays sound on the daemon side — no local _playbin
        assert media.audio._playbin is None, (
            "Playbin should be None after stop_playing()"
        )
    # Give a moment to confirm no crash after stopping
    time.sleep(0.5)
    media.close()


@pytest.mark.audio
@pytest.mark.parametrize("backend", AUDIO_BACKENDS)
def test_push_audio_sample(backend: MediaBackend) -> None:
    """Test pushing an audio sample with the given backend."""
    media = MediaManager(
        backend=backend,
        signalling_host=SIGNALING_HOST
        if backend == MediaBackend.WEBRTC
        else "localhost",
    )
    media.start_playing()

    # Stereo, channels last
    data = np.random.random((media.get_output_audio_samplerate(), 2)).astype(np.float32)
    media.push_audio_sample(data)
    time.sleep(1)

    # Mono, channels last
    data = np.random.random((media.get_output_audio_samplerate(), 1)).astype(np.float32)
    media.push_audio_sample(data)
    time.sleep(1)

    # Multiple channels, channels last
    data = np.random.random((media.get_output_audio_samplerate(), 10)).astype(
        np.float32
    )
    media.push_audio_sample(data)
    time.sleep(1)

    # Stereo, channels first
    data = np.random.random((2, media.get_output_audio_samplerate())).astype(np.float32)
    media.push_audio_sample(data)
    time.sleep(1)

    # No assertion: test passes if no exception is raised.
    # Sound should be audible if the audio device is correctly set up.

    data = np.array(0).astype(np.float32)
    media.push_audio_sample(data)
    time.sleep(1)

    data = np.random.random((media.get_output_audio_samplerate(), 2, 2)).astype(
        np.float32
    )
    media.push_audio_sample(data)
    time.sleep(1)

    # No assertion: test passes if no exception is raised.
    # No sound should be audible if the audio device is correctly set up.

    media.stop_playing()
    media.close()


@pytest.mark.audio
@pytest.mark.parametrize("backend", AUDIO_BACKENDS)
def test_record_audio_and_file_exists(backend: MediaBackend) -> None:
    """Test recording audio and check that the file exists and is not empty."""
    media = MediaManager(
        backend=backend,
        signalling_host=SIGNALING_HOST
        if backend == MediaBackend.WEBRTC
        else "localhost",
    )
    DURATION = 2  # seconds
    audio_samples = []
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmpfile.close()

    t0 = time.time()
    media.start_recording()

    NB_SAMPLES = DURATION * media.get_input_audio_samplerate()
    current_nb_samples = 0
    while current_nb_samples < NB_SAMPLES and time.time() - t0 < DURATION + 5:
        sample = media.get_audio_sample()
        if sample is not None:
            audio_samples.append(sample)
            current_nb_samples += sample.shape[0]

    media.stop_recording()

    assert len(audio_samples) > 0, "No audio samples were recorded."
    audio_data = np.concatenate(audio_samples, axis=0)
    assert audio_data.ndim == 2 and audio_data.shape[1] == media.get_input_channels(), (
        f"Audio data has incorrect number of channels: {audio_data.shape[1]} != {media.get_input_channels()}"
    )
    assert audio_data.shape[0] == pytest.approx(
        DURATION * media.get_input_audio_samplerate(), rel=0.1
    ), (
        f"Audio data has incorrect number of samples: {audio_data.shape[0]} != {DURATION * media.get_input_audio_samplerate()}"
    )

    save_audio_to_wav(audio_data, media.get_input_audio_samplerate(), tmpfile.name)
    assert os.path.exists(tmpfile.name), "File does not exist."
    assert os.path.getsize(tmpfile.name) > 0, "File is empty."

    os.remove(tmpfile.name)
    media.close()


@pytest.mark.audio
@pytest.mark.parametrize("backend", AUDIO_BACKENDS)
def test_record_audio_without_start_recording(backend: MediaBackend) -> None:
    """Test recording audio without starting recording."""
    media = MediaManager(
        backend=backend,
        signalling_host=SIGNALING_HOST
        if backend == MediaBackend.WEBRTC
        else "localhost",
    )
    audio = media.get_audio_sample()
    assert audio is None, "Audio samples were recorded without starting recording."
    media.close()


@pytest.mark.audio
@pytest.mark.parametrize("backend", AUDIO_BACKENDS)
def test_push_audio_sample_without_start_playing(backend: MediaBackend) -> None:
    """Test pushing an audio sample without starting playing."""
    media = MediaManager(
        backend=backend,
        signalling_host=SIGNALING_HOST
        if backend == MediaBackend.WEBRTC
        else "localhost",
    )
    data = np.random.random((media.get_output_audio_samplerate(), 2)).astype(np.float32)
    media.push_audio_sample(data)
    time.sleep(1)
    # No assertion: test passes if no exception is raised.
    # Sound should not be audible if the audio device is correctly set up.
    media.close()


@pytest.mark.audio
@pytest.mark.respeaker
@pytest.mark.parametrize("backend", [MediaBackend.LOCAL])
def test_DoA(backend: MediaBackend) -> None:
    """Test Direction of Arrival (DoA) estimation."""
    media = MediaManager(backend=backend)
    # Test via GStreamerAudio directly
    doa = media.audio.get_DoA()
    assert doa is not None, "DoA is not defined."
    assert isinstance(doa, tuple), "DoA is not a tuple."
    assert len(doa) == 2, f"DoA has incorrect length: {len(doa)} != 2"
    assert isinstance(doa[0], float), (
        f"DoA has incorrect first type: {type(doa[0])} != float"
    )
    assert isinstance(doa[1], bool), (
        f"DoA has incorrect second type: {type(doa[1])} != bool"
    )
    # Test via MediaManager proxy
    doa_proxy = media.get_DoA()
    assert doa_proxy is not None, "DoA is not defined."
    assert doa_proxy == doa, "Proxy DoA is not equal to direct DoA"

    media.close()


@pytest.mark.audio
@pytest.mark.loopback
def test_play_sound_reaches_sink(audio_loopback: None) -> None:
    """Play a sound and confirm real, non-silent audio reaches the sink.

    Unlike test_play_sound (which only checks no-exception, and on a
    discard-only sink can't tell), this records a virtual loopback of the sink
    while playing, so silence means playback never reached the speaker path.
    """
    media = MediaManager(backend=MediaBackend.LOCAL)
    samples = []
    try:
        rate = media.get_input_audio_samplerate()
        media.start_recording()
        media.play_sound("wake_up.wav")
        t0 = time.time()
        while time.time() - t0 < 3.0:
            sample = media.get_audio_sample()
            if sample is not None:
                samples.append(sample)
        media.stop_recording()
    finally:
        media.close()

    assert samples, "No audio captured from the loopback."
    audio = np.concatenate(samples, axis=0)
    peak = float(np.abs(audio).max())
    assert peak > 1e-3, (
        f"Captured audio is silent (peak={peak}); playback did not reach the sink."
    )

    # Confirm it's actually the wake_up sound (not just any audio): compare the
    # captured spectrum to the source file resampled to the capture rate.
    captured = audio.astype(np.float64).mean(axis=1)
    with wave.open(f"{ASSETS_ROOT_PATH}/wake_up.wav") as wav:
        channels = wav.getnchannels()
        src_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    reference = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    reference = reference.reshape(-1, channels).mean(axis=1)
    reference = resample_poly(reference, rate, src_rate)

    similarity = _spectral_cosine(captured, reference)
    # Measured ~0.75-0.84 for the real sound vs ~0.10 for unrelated noise.
    assert similarity > 0.5, (
        f"Captured audio does not match wake_up.wav (spectral cosine={similarity:.2f})."
    )


def test_doa_simulated(fake_respeaker, monkeypatch) -> None:
    """DoA read path works against a fake board (no ReSpeaker hardware).

    Patches init_respeaker_usb so AudioDoA wraps the seeded fake, then checks
    get_DoA decodes DOA_VALUE_RADIANS into the (angle, speech) tuple.
    """
    monkeypatch.setattr(
        "reachy_mini.media.audio_doa.init_respeaker_usb", lambda: fake_respeaker
    )
    from reachy_mini.media.audio_doa import AudioDoA

    doa = AudioDoA()
    try:
        result = doa.get_DoA()
    finally:
        doa.close()

    assert result is not None
    angle, speech = result
    assert isinstance(angle, float) and angle == pytest.approx(1.57)
    assert speech is True


def test_no_media() -> None:
    """Test that methods handle uninitialized media gracefully."""
    media = MediaManager(backend=MediaBackend.NO_MEDIA)

    assert media.get_frame() is None
    assert media.get_audio_sample() is None
    assert media.get_input_audio_samplerate() == -1
    assert media.get_output_audio_samplerate() == -1
    assert media.get_input_channels() == -1
    assert media.get_output_channels() == -1
    assert media.get_DoA() is None

    media.close()


def test_get_respeaker_card_number() -> None:
    """Test getting the ReSpeaker card number."""
    alsa_output = "carte 5 : Audio [Reachy Mini Audio], p\u00e9riph\u00e9rique 0 : USB Audio [USB Audio]"
    card_number = _process_card_number_output(alsa_output)
    assert isinstance(card_number, int)
    assert card_number == 5
    alsa_output = "card 0: Audio [Reachy Mini Audio], device 0: USB Audio [USB Audio]"
    card_number = _process_card_number_output(alsa_output)
    assert card_number == 0
    alsa_output = "card 3: PCH [HDA Intel PCH], device 0: ALC255 Analog [ALC255 Analog]"
    card_number = _process_card_number_output(alsa_output)
    assert card_number == 0
