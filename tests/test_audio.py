import importlib.util
import os
import tempfile
import time

import numpy as np
import pytest

from reachy_mini.media.audio_utils import _process_card_number_output
from reachy_mini.media.media_manager import MediaBackend, MediaManager

SIGNALING_HOST = "reachy-mini.local"

# Check if sounddevice is installed
_sounddevice_available = importlib.util.find_spec("sounddevice") is not None

# All audio only backends to test
NO_VIDEO_BACKENDS = [
    pytest.param(backend, marks=pytest.mark.wireless) if backend == MediaBackend.WEBRTC else pytest.param(backend)
    for backend in MediaBackend
    if ("NO_VIDEO" in backend.name or backend == MediaBackend.WEBRTC)
    and (_sounddevice_available or "SOUNDDEVICE" not in backend.name)
]

@pytest.mark.audio
@pytest.mark.parametrize("backend", NO_VIDEO_BACKENDS)
def test_play_sound(backend: MediaBackend) -> None:
    """Test playing a sound with the given backend."""
    media = MediaManager(backend=backend, signalling_host=SIGNALING_HOST if backend == MediaBackend.WEBRTC else "localhost")
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
@pytest.mark.parametrize("backend", NO_VIDEO_BACKENDS)
def test_push_audio_sample(backend: MediaBackend) -> None:
    """Test pushing an audio sample with the given backend."""
    media = MediaManager(backend=backend, signalling_host=SIGNALING_HOST if backend == MediaBackend.WEBRTC else "localhost")
    media.start_playing()

    #Stereo, channels last
    data = np.random.random((media.get_output_audio_samplerate(), 2)).astype(np.float32)
    media.push_audio_sample(data)
    time.sleep(1)

    #Mono, channels last
    data = np.random.random((media.get_output_audio_samplerate(), 1)).astype(np.float32)
    media.push_audio_sample(data)
    time.sleep(1)

    #Multiple channels, channels last
    data = np.random.random((media.get_output_audio_samplerate(), 10)).astype(np.float32)
    media.push_audio_sample(data)
    time.sleep(1)

    #Stereo, channels first
    data = np.random.random((2, media.get_output_audio_samplerate())).astype(np.float32)
    media.push_audio_sample(data)
    time.sleep(1)

    # No assertion: test passes if no exception is raised.
    # Sound should be audible if the audio device is correctly set up.

    data = np.array(0).astype(np.float32)
    media.push_audio_sample(data)
    time.sleep(1)

    data = np.random.random((media.get_output_audio_samplerate(), 2, 2)).astype(np.float32)
    media.push_audio_sample(data)
    time.sleep(1)

    # No assertion: test passes if no exception is raised.
    # No sound should be audible if the audio device is correctly set up.

    media.stop_playing()
    media.close()


@pytest.mark.audio
@pytest.mark.parametrize("backend", NO_VIDEO_BACKENDS)
def test_record_audio_and_file_exists(backend: MediaBackend) -> None:
    """Test recording audio and check that the file exists and is not empty."""
    import soundfile as sf
    media = MediaManager(backend=backend, signalling_host=SIGNALING_HOST if backend == MediaBackend.WEBRTC else "localhost")
    DURATION = 2  # seconds
    audio_samples = []
    tmpfile = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmpfile.close()

    t0 = time.time()
    media.start_recording()
    while time.time() - t0 < DURATION:
        sample = media.get_audio_sample()
        if sample is not None:
            audio_samples.append(sample)

    media.stop_recording()

    assert len(audio_samples) > 0, "No audio samples were recorded."
    audio_data = np.concatenate(audio_samples, axis=0)
    assert audio_data.ndim == 2 and audio_data.shape[1] == media.get_input_channels(), f"Audio data has incorrect number of channels: {audio_data.shape[1]} != {media.get_input_channels()}"
    assert audio_data.shape[0] == pytest.approx(DURATION * media.get_input_audio_samplerate(), rel=0.1), f"Audio data has incorrect number of samples: {audio_data.shape[0]} != {DURATION * media.get_input_audio_samplerate()}"

    sf.write(tmpfile.name, audio_data, media.get_input_audio_samplerate())
    assert os.path.exists(tmpfile.name), "File does not exist."
    assert os.path.getsize(tmpfile.name) > 0, "File is empty."

    os.remove(tmpfile.name)
    media.close()


@pytest.mark.audio
@pytest.mark.parametrize("backend", NO_VIDEO_BACKENDS)
def test_record_audio_without_start_recording(backend: MediaBackend) -> None:
    """Test recording audio without starting recording."""
    media = MediaManager(backend=backend, signalling_host=SIGNALING_HOST if backend == MediaBackend.WEBRTC else "localhost")
    audio = media.get_audio_sample()
    assert audio is None, "Audio samples were recorded without starting recording."
    media.close()


@pytest.mark.audio
@pytest.mark.parametrize("backend", NO_VIDEO_BACKENDS)
def test_push_audio_sample_without_start_playing(backend: MediaBackend) -> None:
    """Test pushing an audio sample without starting playing."""
    media = MediaManager(backend=backend, signalling_host=SIGNALING_HOST if backend == MediaBackend.WEBRTC else "localhost")
    data = np.random.random((media.get_output_audio_samplerate(), 2)).astype(np.float32)
    media.push_audio_sample(data)
    time.sleep(1)
    # No assertion: test passes if no exception is raised.
    # Sound should not be audible if the audio device is correctly set up.
    media.close()


@pytest.mark.audio_sounddevice
@pytest.mark.parametrize("backend", [MediaBackend.SOUNDDEVICE_NO_VIDEO])
def test_record_audio_above_max_queue_seconds(backend: MediaBackend) -> None:
    """Test recording audio and check that the maximum queue seconds is respected."""
    media = MediaManager(backend=backend, signalling_host=SIGNALING_HOST if backend == MediaBackend.WEBRTC else "localhost")
    media.audio._input_max_queue_seconds = 1
    media.start_recording()
    time.sleep(5)
    audio = media.get_audio_sample()
    media.stop_recording()

    assert audio is not None, "No audio samples were recorded."
    assert audio.shape[0] < media.audio._input_max_queue_samples, f"Audio data has incorrect number of samples: {audio.shape[0]} >= {media.audio._input_max_queue_samples}"

    media.close()


@pytest.mark.audio
@pytest.mark.parametrize("backend", NO_VIDEO_BACKENDS)
def test_DoA(backend: MediaBackend) -> None:
    """Test Direction of Arrival (DoA) estimation."""
    media = MediaManager(backend=backend, signalling_host=SIGNALING_HOST if backend == MediaBackend.WEBRTC else "localhost")
    # Test via AudioBase directly
    doa = media.audio.get_DoA()
    assert doa is not None, "DoA is not defined."
    assert isinstance(doa, tuple), "DoA is not a tuple."
    assert len(doa) == 2, f"DoA has incorrect length: {len(doa)} != 2"
    assert isinstance(doa[0], float), f"DoA has incorrect first type: {type(doa[0])} != float"
    assert isinstance(doa[1], bool), f"DoA has incorrect second type: {type(doa[1])} != bool"
    # Test via MediaManager proxy
    doa_proxy = media.get_DoA()
    assert doa_proxy is not None, "DoA is not defined."
    assert doa_proxy == doa, f"Proxy DoA is not equal to direct DoA"

    media.close()


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
    alsa_output = "carte 5 : Audio [Reachy Mini Audio], périphérique 0 : USB Audio [USB Audio]"
    card_number = _process_card_number_output(alsa_output)
    assert isinstance(card_number, int)
    assert card_number == 5
    alsa_output = "card 0: Audio [Reachy Mini Audio], device 0: USB Audio [USB Audio]"
    card_number = _process_card_number_output(alsa_output)
    assert card_number == 0
    alsa_output = "card 3: PCH [HDA Intel PCH], device 0: ALC255 Analog [ALC255 Analog]"
    card_number = _process_card_number_output(alsa_output)
    assert card_number == 0
