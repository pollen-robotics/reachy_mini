import os
import time
import tempfile
import pytest
import soundfile as sf
from reachy_mini.media.media_manager import MediaManager, MediaBackend
import numpy as np

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
def test_DoA() -> None:
    """Test Direction of Arrival (DoA) estimation."""
    media = MediaManager(backend=MediaBackend.DEFAULT_NO_VIDEO)
    doa = media.audio.get_DoA()
    assert doa is not None
    assert isinstance(doa, tuple)
    assert len(doa) == 2
    assert isinstance(doa[0], float)
    assert isinstance(doa[1], bool)


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
