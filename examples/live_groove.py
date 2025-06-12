import time
import numpy as np
import threading
import librosa
import pyaudio

# --- SDK and Dance Library Imports ---
from reachy_mini import ReachyMini
from scipy.spatial.transform import Rotation as R
from dance_moves import (
    AVAILABLE_DANCE_MOVES,
    MOVE_SPECIFIC_PARAMS
)

# --- Configuration ---
DANCE_MOVE_TO_PERFORM = "head_bob_z" # Using head_bob_z as requested
CONTROL_TIMESTEP = 0.02  # 50 Hz control loop
NEUTRAL_HEAD_POSITION = np.array([0.0, 0.0, 0.177 - 0.0075])
NEUTRAL_HEAD_EULER_ANGLES = np.array([0.0, 0.0, 0.0])

# Audio analysis configuration
AUDIO_RATE = 44100
AUDIO_CHUNK_SIZE = 2048
AUDIO_ANALYSIS_SECONDS = 2.5
AUDIO_BUFFER_SIZE = int(AUDIO_RATE * AUDIO_ANALYSIS_SECONDS)

# --- Time Controller Configuration ---
# How aggressively the robot tries to catch up to the beat.
# Higher value = faster, more aggressive correction.
PHASE_CORRECTION_FACTOR = 0.8
# Limits how much faster/slower the robot's rhythm can be than the music's.
# 0.5 means it can speed up to 150% or slow down to 50% of the music's tempo.
MAX_RATE_ADJUSTMENT = 0.5 

# --- Shared State for Threads ---
class MusicState:
    def __init__(self):
        self.lock = threading.Lock()
        self.goal_bpm = 0.0
        self.last_beat_time = 0.0

# --- Audio Analysis Thread (Unchanged) ---
def audio_analysis_thread(music_state: MusicState):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=AUDIO_CHUNK_SIZE)
    print("Audio thread started. Listening for music...")
    audio_buffer = np.array([], dtype=np.float32)
    last_printed_bpm = 0
    while True:
        try:
            data = np.frombuffer(stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False), dtype=np.float32)
            audio_buffer = np.append(audio_buffer, data)
            if len(audio_buffer) >= AUDIO_BUFFER_SIZE:
                y = audio_buffer
                tempo, beats = librosa.beat.beat_track(y=y, sr=AUDIO_RATE, units='frames')
                new_bpm = float(tempo)
                if abs(new_bpm - last_printed_bpm) > 1 and new_bpm > 40:
                    print(f"BPM Estimate: {new_bpm:.1f}", end='\r')
                    last_printed_bpm = new_bpm
                with music_state.lock:
                    if new_bpm > 40:
                        music_state.goal_bpm = new_bpm
                        if len(beats) > 0:
                            last_beat_frame = beats[-1]
                            time_of_last_beat_in_buffer = librosa.frames_to_time(last_beat_frame, sr=AUDIO_RATE)
                            buffer_duration = len(audio_buffer) / AUDIO_RATE
                            time_ago = buffer_duration - time_of_last_beat_in_buffer
                            music_state.last_beat_time = time.time() - time_ago
                audio_buffer = audio_buffer[-int(AUDIO_RATE * 1.5):]
        except Exception as e:
            print(f"Audio thread error: {e}")
            break
    stream.stop_stream()
    stream.close()
    p.terminate()

def create_head_pose(position, euler_angles):
    pose_matrix = np.eye(4)
    pose_matrix[:3, 3] = position
    pose_matrix[:3, :3] = R.from_euler('xyz', euler_angles).as_matrix()
    return pose_matrix

# --- Main Robot Control Loop ---
def main():
    music_state = MusicState()
    analyzer_thread = threading.Thread(target=audio_analysis_thread, args=(music_state,), daemon=True)
    analyzer_thread.start()

    neutral_pose = create_head_pose(NEUTRAL_HEAD_POSITION, NEUTRAL_HEAD_EULER_ANGLES)
    
    current_musical_time = 0.0
    last_loop_time = time.time()
    
    try:
        print("Connecting to Reachy Mini...")
        with ReachyMini() as reachy_mini:
            print("Connected. Moving to neutral pose.")
            reachy_mini.set_position(head=neutral_pose, antennas=np.array([0.0, 0.0]))
            time.sleep(1)
            print("\nRobot is ready. Start playing music!")

            move_function = AVAILABLE_DANCE_MOVES[DANCE_MOVE_TO_PERFORM]
            move_params = MOVE_SPECIFIC_PARAMS.get(DANCE_MOVE_TO_PERFORM, {})
            frequency_factor = move_params.get("frequency_factor", 1.0)
            phase_offset = 1.0 / (4.0 * frequency_factor)

            while True:
                loop_start_time = time.time()
                delta_t = loop_start_time - last_loop_time
                last_loop_time = loop_start_time
                
                with music_state.lock:
                    goal_bpm = music_state.goal_bpm
                    last_beat_detected_time = music_state.last_beat_time
                
                if goal_bpm == 0.0 or last_beat_detected_time == 0.0:
                    time.sleep(0.1)
                    continue

                # --- Continuous Phase-Locked Loop Controller ---
                
                # 1. Determine the music's current time (the "goal")
                # This is how far into a beat we are according to the music.
                time_since_beat = loop_start_time - last_beat_detected_time
                goal_musical_time = time_since_beat * (goal_bpm / 60.0)

                # 2. Calculate the phase error
                # This is the difference between our internal clock and the music's clock.
                time_error = goal_musical_time - current_musical_time

                # 3. Calculate the new speed for our internal clock
                # The base speed is the music's tempo. The correction speed tries to reduce the error.
                base_speed = goal_bpm / 60.0 # Beats per second
                correction_speed = time_error * PHASE_CORRECTION_FACTOR
                
                # 4. Limit the correction to make it feel natural
                # The robot can't instantly speed up or slow down infinitely.
                adjustment_limit = base_speed * MAX_RATE_ADJUSTMENT
                correction_speed = np.clip(correction_speed, -adjustment_limit, adjustment_limit)
                
                # The final speed of our internal clock for this frame.
                total_speed = base_speed + correction_speed

                # 5. Increment our internal clock continuously. NO JUMPS.
                current_musical_time += total_speed * delta_t
                
                # --- Motion Execution ---
                phased_beat_time = current_musical_time + phase_offset
                offsets = move_function(phased_beat_time, **move_params)
                
                pos_offset = offsets.get("position_offset", np.zeros(3))
                orient_offset_euler = offsets.get("orientation_offset", np.zeros(3))
                antennas_command = offsets.get("antennas_offset", np.zeros(2))
                
                target_pose = create_head_pose(
                    NEUTRAL_HEAD_POSITION + pos_offset,
                    NEUTRAL_HEAD_EULER_ANGLES + orient_offset_euler
                )
                reachy_mini.set_position(head=target_pose, antennas=antennas_command)

                time.sleep(max(0, CONTROL_TIMESTEP - (time.time() - loop_start_time)))

    except KeyboardInterrupt:
        print("\nStopping dance...")
    except Exception as e:
        print(f"\nAn error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program ended.")

if __name__ == "__main__":
    main()