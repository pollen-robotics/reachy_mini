import os
import tempfile
import time

import numpy as np
import pytest
import soundfile as sf

from reachy_mini.media.audio_utils import _process_card_number_output
from reachy_mini.media.media_manager import MediaBackend, MediaManager


@pytest.mark.audio
def test_play_sound_default_backend() -> None:
    """Test playing a sound with the default backend."""
    media = MediaManager(backend=MediaBackend.DEFAULT_NO_VIDEO)
    # Use a short sound file present in your assets directory
    sound_file = "wake_up.wav"  # Change to a valid file if needed
    media.play_sound(sound_file)
    print("Playing sound with default backend...")
    # Wait a bit to let the sound play (non-blocking backend)
    time.sleep(2)
    # No assertion: test passes if no exception is raised.
    # Sound should be audible if the audio device is correctly set up.

@pytest.mark.audio
def test_push_audio_sample_default_backend() -> None:
    """Test pushing an audio sample with the default backend."""
    media = MediaManager(backend=MediaBackend.DEFAULT_NO_VIDEO)
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

@pytest.mark.audio
def test_record_audio_and_file_exists() -> None:
    """Test recording audio and check that the file exists and is not empty."""
    media = MediaManager(backend=MediaBackend.DEFAULT_NO_VIDEO)
    DURATION = 2  # seconds
    tmpfile = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmpfile.close()
    media.start_recording()
    time.sleep(DURATION)
    media.stop_recording()
    audio = media.get_audio_sample()
    samplerate = media.get_input_audio_samplerate()
    assert audio is not None
    sf.write(tmpfile.name, audio, samplerate)
    assert os.path.exists(tmpfile.name)
    assert os.path.getsize(tmpfile.name) > 0
    # comment the following line if you want to keep the file for inspection
    os.remove(tmpfile.name)
    #print(f"Recorded audio saved to {tmpfile.name}")

@pytest.mark.audio
def test_record_audio_without_start_recording() -> None:
    """Test recording audio without starting recording."""
    media = MediaManager(backend=MediaBackend.DEFAULT_NO_VIDEO)
    audio = media.get_audio_sample()
    assert audio is None

@pytest.mark.audio
def test_record_audio_above_max_queue_seconds() -> None:
    """Test recording audio and check that the maximum queue seconds is respected."""
    media = MediaManager(backend=MediaBackend.DEFAULT_NO_VIDEO)
    media.audio._input_max_queue_seconds = 1
    media.start_recording()
    time.sleep(5)
    audio = media.get_audio_sample()
    media.stop_recording()

    assert audio is not None
    assert audio.shape[0] < media.audio._input_max_queue_samples

@pytest.mark.audio
def test_DoA() -> None:
    """Test Direction of Arrival (DoA) estimation."""
    media = MediaManager(backend=MediaBackend.DEFAULT_NO_VIDEO)
    # Test via AudioBase directly
    doa = media.audio.get_DoA()
    assert doa is not None
    assert isinstance(doa, tuple)
    assert len(doa) == 2
    assert isinstance(doa[0], float)
    assert isinstance(doa[1], bool)
    # Test via MediaManager proxy
    doa_proxy = media.get_DoA()
    assert doa_proxy is not None
    assert doa_proxy == doa


'''
@pytest.mark.audio_gstreamer
def test_play_sound_gstreamer_backend() -> None:
    """Test playing a sound with the GStreamer backend."""
    media = MediaManager(backend=MediaBackend.GSTREAMER)
    time.sleep(2)  # Give some time for the audio system to initialize
    # Use a short sound file present in your assets directory
    sound_file = "wake_up.wav"  # Change to a valid file if needed
    media.play_sound(sound_file)
    print("Playing sound with GStreamer backend...")
    # Wait a bit to let the sound play (non-blocking backend)
    time.sleep(2)
    # No assertion: test passes if no exception is raised.
    # Sound should be audible if the audio device is correctly set up.
'''

@pytest.mark.audio_gstreamer
def test_record_audio_and_file_exists_gstreamer() -> None:
    """Test recording audio and check that the file exists and is not empty."""
    media = MediaManager(backend=MediaBackend.GSTREAMER)
    DURATION = 2  # seconds
    tmpfile = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmpfile.close()
    audio_samples = []
    t0 = time.time()
    media.start_recording()

    while time.time() - t0 < DURATION:
        sample = media.get_audio_sample()

        if sample is not None:
            audio_samples.append(sample)

    media.stop_recording()
    
    assert len(audio_samples) > 0
    audio_data = np.concatenate(audio_samples, axis=0)
    assert audio_data.ndim == 2 and audio_data.shape[1] == 2
    samplerate = media.get_input_audio_samplerate()
    sf.write(tmpfile.name, audio_data, samplerate)
    assert os.path.exists(tmpfile.name)
    assert os.path.getsize(tmpfile.name) > 0
    #os.remove(tmpfile.name)
    print(f"Recorded audio saved to {tmpfile.name}")


def test_no_media() -> None:
    """Test that methods handle uninitialized media gracefully."""
    media = MediaManager(backend=MediaBackend.NO_MEDIA)

    assert media.get_frame() is None
    assert media.get_audio_sample() is None
    assert media.get_input_audio_samplerate() == -1
    assert media.get_output_audio_samplerate() == -1
    assert media.get_DoA() is None


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