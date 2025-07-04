import time
import numpy as np
import threading
import aubio
import pyaudio

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
AUBIO_HOP_SIZE = 512
# --- NEW: Hysteresis Thresholds ---
# The confidence score required to START dancing.
START_DANCING_CONFIDENCE = 0.13
# If confidence drops below this value, the robot will STOP dancing.
STOP_DANCING_CONFIDENCE = 0.08

# --- Logging Configuration ---
PRINT_INTERVAL = 0.25

# --- Static Configurations ---
NEUTRAL_HEAD_POSITION = np.array([0.0, 0.0, 0.0])
NEUTRAL_HEAD_EULER_ANGLES = np.array([0.0, 0.0, 0.0])
AUDIO_RATE = 44100

# --- Shared State for Threads ---
class MusicState:
    def __init__(self):
        self.lock = threading.Lock()
        self.goal_bpm = 0.0
        self.last_beat_time = 0.0
        self.latest_observation = 0.0
        self.confidence = 0.0
        self.filter_status = "Waiting for beat..."

# --- Audio Analysis Thread (Unchanged) ---
# This thread continuously provides observations. The main loop decides what to do with them.
def audio_analysis_thread(music_state: MusicState):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=AUBIO_HOP_SIZE)
    aubio_tempo = aubio.tempo("default", AUBIO_HOP_SIZE * 4, AUBIO_HOP_SIZE, AUDIO_RATE)
    print("Audio thread started. Listening for music...")
    while True:
        try:
            data = stream.read(AUBIO_HOP_SIZE, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=aubio.float_type)
            is_beat = aubio_tempo(samples)
            if is_beat:
                current_bpm = aubio_tempo.get_bpm()
                confidence = aubio_tempo.get_confidence()
                with music_state.lock:
                    music_state.latest_observation = current_bpm
                    music_state.confidence = confidence
                    # The goal_bpm is only updated on a confident beat.
                    # The robot will continue using the last known good BPM even if confidence drops.
                    if confidence >= START_DANCING_CONFIDENCE:
                        music_state.goal_bpm = current_bpm
                        music_state.last_beat_time = time.time()
                        music_state.filter_status = "Confident Lock"
                    else:
                        music_state.filter_status = "Low Confidence"
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

# --- NEW: User-provided control function ---
def musical_time_diff(a, b):
    """Calculates the shortest circular distance between two musical times (modulo 1.0)."""
    d = a - b
    # This is a standard way to compute modulo with results in [-0.5, 0.5]
    d = (d + 0.5) % 1.0 - 0.5
    return d

# --- Status Printing Function (MODIFIED) ---
def print_status(status_info):
    move_up = f"\033[{status_info['lines']}F"
    clear_line = "\033[K"
    status_string = f"""
--- Reachy Mini Live Groove (Aubio) ---
{clear_line}Robot State:       {status_info['robot_state']}
{clear_line}Live Observation:  {status_info['latest_observation']:.1f} BPM
{clear_line}Confidence:        {status_info['confidence']:.2f} (Start: >{START_DANCING_CONFIDENCE:.2f}, Stop: <{STOP_DANCING_CONFIDENCE:.2f})
{clear_line}
{clear_line}ROBOT TARGET BPM:  {status_info['goal_bpm']:.1f} BPM
{clear_line}Tempo Error (mod 1): {status_info['time_error']:+.3f} beats
"""
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
    last_print_time = 0
    loop_count = 0
    
    # NEW: State variable for hysteresis
    is_dancing = False

    try:
        print("Connecting to Reachy Mini...")
        with ReachyMini() as reachy_mini:
            print("Connected. Moving to neutral pose.")
            reachy_mini.set_target(head=neutral_pose, antennas=np.array([0.0, 0.0]))
            time.sleep(1)
            print("\nRobot is ready. Start playing music!")
            print("\n" * 7) # Adjusted for new print block size

            move_function = AVAILABLE_DANCE_MOVES[DANCE_MOVE_TO_PERFORM]
            move_params = MOVE_SPECIFIC_PARAMS.get(DANCE_MOVE_TO_PERFORM, {})
            frequency_factor = move_params.get("frequency_factor", 1.0)
            phase_offset = 1.0 / (4.0 * frequency_factor)
            
            while True:
                loop_start_time = time.time()
                
                with music_state.lock:
                    goal_bpm = music_state.goal_bpm
                    last_beat_detected_time = music_state.last_beat_time
                    latest_observation = music_state.latest_observation
                    confidence = music_state.confidence
                
                # --- NEW: Hysteresis Logic ---
                if not is_dancing and confidence > START_DANCING_CONFIDENCE:
                    is_dancing = True
                    # When starting, jump internal time immediately to the correct phase
                    if last_beat_detected_time > 0:
                         current_musical_time = (loop_start_time - last_beat_detected_time) * (goal_bpm / 60.0)
                elif is_dancing and confidence < STOP_DANCING_CONFIDENCE:
                    is_dancing = False

                # --- Control and Motion Logic ---
                time_error = 0.0
                if is_dancing and goal_bpm > 0:
                    # --- NEW: User-provided control algorithm ---
                    time_since_beat = loop_start_time - last_beat_detected_time
                    goal_musical_time = time_since_beat * (goal_bpm / 60.0)
                    time_error = musical_time_diff(goal_musical_time, current_musical_time)
                    
                    # This is the "jump" to correct the phase
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
                    reachy_mini.set_target(head=target_pose, antennas=antennas_command)
                
                # --- Always print status for continuous feedback ---
                if (loop_start_time - last_print_time) > PRINT_INTERVAL:
                    print_status({
                        "lines": 7, "loop_count": loop_count,
                        "robot_state": "Dancing" if is_dancing else "Waiting for confidence...",
                        "latest_observation": latest_observation,
                        "confidence": confidence, "goal_bpm": goal_bpm,
                        "time_error": time_error,
                    })
                    last_print_time = loop_start_time
                    loop_count += 1
                
                time.sleep(max(0, CONTROL_TIMESTEP - (time.time() - loop_start_time)))

    except KeyboardInterrupt:
        print("\n" * 8)
        print("Stopping dance...")
    except Exception as e:
        print(f"\nAn error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program ended.")

if __name__ == "__main__":
    main()