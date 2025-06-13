import time
import numpy as np
import threading
import librosa
import pyaudio
import collections

# --- SDK and Dance Library Imports ---
from reachy_mini import ReachyMini
from scipy.spatial.transform import Rotation as R
from dance_moves import (
    AVAILABLE_DANCE_MOVES,
    MOVE_SPECIFIC_PARAMS
)

# --- Configuration ---
DANCE_MOVE_TO_PERFORM = "head_bob_z"
CONTROL_TIMESTEP = 0.02

# --- TUNABLE PARAMETERS FOR RHYTHM DETECTION ---
AUDIO_ANALYSIS_SECONDS = 2.0
BPM_HISTORY_LENGTH = 4
BPM_STABILITY_THRESHOLD = 3.0

# --- Time Controller Configuration ---
PHASE_CORRECTION_FACTOR = 0.8
MAX_RATE_ADJUSTMENT = 0.5

# --- NEW: Logging Configuration ---
PRINT_INTERVAL = 0.25  # Update status display 4 times per second

# --- Static Configurations (Derived from above) ---
NEUTRAL_HEAD_POSITION = np.array([0.0, 0.0, 0.177 - 0.0075])
NEUTRAL_HEAD_EULER_ANGLES = np.array([0.0, 0.0, 0.0])
AUDIO_RATE = 44100
AUDIO_CHUNK_SIZE = 2048
AUDIO_BUFFER_SIZE = int(AUDIO_RATE * AUDIO_ANALYSIS_SECONDS)

def angle_diff(a, b):
    """Returns the smallest distance between 2 angles"""
    d = a - b
    d = ((d + np.pi) % (2 * np.pi)) - np.pi
    return d

def musical_time_diff(a, b):
    """Returns the smallest distance between 2 musical times (equivalent to a modulo 1.0)"""
    d = a - b
    d = ((d + 0.5) % (2 * 0.5)) - 0.5
    return d


# --- Shared State for Threads (MODIFIED) ---
class MusicState:
    def __init__(self):
        self.lock = threading.Lock()
        # --- Control Variables ---
        self.goal_bpm = 0.0  # The stable, filtered BPM for the robot
        self.last_beat_time = 0.0
        # --- NEW: Diagnostic Variables ---
        self.latest_observation = 0.0  # The raw, unfiltered BPM from Librosa
        self.filter_std_dev = 0.0      # The current standard deviation of the BPM history
        self.filter_status = "Initializing..." # A string indicating the filter's state

# --- Audio Analysis Thread (MODIFIED) ---
def audio_analysis_thread(music_state: MusicState):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=AUDIO_CHUNK_SIZE)
    
    bpm_history = collections.deque(maxlen=BPM_HISTORY_LENGTH)
    
    print("Audio thread started. Listening for music...")
    audio_buffer = np.array([], dtype=np.float32)
    
    while True:
        try:
            data = np.frombuffer(stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False), dtype=np.float32)
            audio_buffer = np.append(audio_buffer, data)
            
            if len(audio_buffer) >= AUDIO_BUFFER_SIZE:
                y = audio_buffer
                
                tempo, beats = librosa.beat.beat_track(y=y, sr=AUDIO_RATE, units='frames', tightness=100)
                new_bpm_observation = float(tempo)
                print(new_bpm_observation)
                
                # Update latest observation for printing, regardless of validity
                with music_state.lock:
                    music_state.latest_observation = new_bpm_observation

                if new_bpm_observation > 40:
                    bpm_history.append(new_bpm_observation)

                if len(bpm_history) < BPM_HISTORY_LENGTH:
                     with music_state.lock:
                        music_state.filter_status = f"Gathering history ({len(bpm_history)}/{BPM_HISTORY_LENGTH})"
                else:
                    std_dev = np.std(bpm_history)
                    
                    with music_state.lock:
                        music_state.filter_std_dev = std_dev

                    if std_dev < BPM_STABILITY_THRESHOLD:
                        stable_bpm = np.mean(bpm_history)
                        with music_state.lock:
                            music_state.goal_bpm = stable_bpm
                            music_state.filter_status = "Locked"
                            if len(beats) > 0:
                                last_beat_frame = beats[-1]
                                time_of_last_beat_in_buffer = librosa.frames_to_time(last_beat_frame, sr=AUDIO_RATE)
                                buffer_duration = len(audio_buffer) / AUDIO_RATE
                                time_ago = buffer_duration - time_of_last_beat_in_buffer
                                music_state.last_beat_time = time.time() - time_ago
                    else:
                        with music_state.lock:
                            music_state.filter_status = "Ignoring unstable"

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

# --- NEW: Status Printing Function ---
def print_status(status_info):
    """Formats and prints a multi-line status block to the console."""
    # ANSI escape codes to move cursor up and clear lines
    move_up = f"\033[{status_info['lines']}F"
    clear_line = "\033[K"
    
    status_string = f"""
--- Reachy Mini Live Groove ---
{clear_line}BPM Filter Status: {status_info['filter_status']}
{clear_line}Live Observation:  {status_info['latest_observation']:.1f} BPM
{clear_line}History Std Dev:   {status_info['filter_std_dev']:.2f} (Threshold: < {BPM_STABILITY_THRESHOLD:.1f})
{clear_line}
{clear_line}ROBOT TARGET BPM:  {status_info['goal_bpm']:.1f} BPM
{clear_line}
{clear_line}Phase Error:       {status_info['time_error']:+.3f} beats
"""
    # Move cursor up and print the block. The initial move_up is only for subsequent prints.
    if status_info['loop_count'] > 0:
        print(move_up, end="")
    print(status_string, end="")

# --- Main Robot Control Loop (MODIFIED) ---
def main():
    music_state = MusicState()
    analyzer_thread = threading.Thread(target=audio_analysis_thread, args=(music_state,), daemon=True)
    analyzer_thread.start()
    neutral_pose = create_head_pose(NEUTRAL_HEAD_POSITION, NEUTRAL_HEAD_EULER_ANGLES)
    current_musical_time = 0.0
    last_loop_time = time.time()
    
    # NEW: Variables for controlling the print rate
    last_print_time = 0
    loop_count = 0

    try:
        print("Connecting to Reachy Mini...")
        with ReachyMini() as reachy_mini:
            print("Connected. Moving to neutral pose.")
            reachy_mini.set_position(head=neutral_pose, antennas=np.array([0.0, 0.0]))
            time.sleep(1)
            print("\nRobot is ready. Start playing music!")
            
            # This empty print creates space for our status block
            print("\n" * 9)

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
                    # Get diagnostic info
                    latest_observation = music_state.latest_observation
                    filter_std_dev = music_state.filter_std_dev
                    filter_status = music_state.filter_status
                
                if goal_bpm == 0.0 or last_beat_detected_time == 0.0 or music_state.filter_status != "Locked":
                    time.sleep(0.1)
                    continue

                time_since_beat = loop_start_time - last_beat_detected_time
                goal_musical_time = time_since_beat * (goal_bpm / 60.0)
                time_error = musical_time_diff(goal_musical_time, current_musical_time)
                
                current_musical_time = current_musical_time + time_error
                
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

                # --- NEW: Periodic Status Printing ---
                if (loop_start_time - last_print_time) > PRINT_INTERVAL:
                    status_info = {
                        "lines": 9,
                        "loop_count": loop_count,
                        "filter_status": filter_status,
                        "latest_observation": latest_observation,
                        "filter_std_dev": filter_std_dev,
                        "goal_bpm": goal_bpm,
                        "time_error": time_error,
                        # "speed_percent": (total_speed / base_speed * 100) if base_speed > 0 else 0
                    }
                    print_status(status_info)
                    last_print_time = loop_start_time
                    loop_count += 1
                
                time.sleep(max(0, CONTROL_TIMESTEP - (time.time() - loop_start_time)))

    except KeyboardInterrupt:
        print("\n" * 10) # Move down past the status block before printing exit message
        print("Stopping dance...")
    except Exception as e:
        print(f"\nAn error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program ended.")

if __name__ == "__main__":
    main()