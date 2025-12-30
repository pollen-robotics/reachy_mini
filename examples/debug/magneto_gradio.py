import gradio as gr
import numpy as np
import threading
import time
import argparse
import sys
import os
import soundfile as sf # Required for reading playback files

# --- CONFIGURATION ---
FS = 16000  # Standard sample rate (16kHz)

# --- RECORDER CLASSES ---

class BaseRecorder:
    """Generic interface for recording."""
    def __init__(self):
        self.is_recording = False
        self.audio_buffer = []
        self.thread = None
        self.playback_path = None # Store file path

    def set_playback_file(self, filepath):
        self.playback_path = filepath

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

class RawRecorder(BaseRecorder):
    """Records directly from the hardware (Requires Daemon STOPPED)."""
    def __init__(self):
        super().__init__()
        import sounddevice as sd
        self.sd = sd
        self.stream = None

    def callback(self, indata, frames, time, status):
        if status:
            print(f"⚠️ Audio Status: {status}")
        self.audio_buffer.append(indata.copy())

    def start(self):
        if self.is_recording: return "⚠️ Already in progress"
        
        print("🎙️ (RAW) Opening audio stream...")
        self.audio_buffer = []
        
        try:
            # --- PLAYBACK LOGIC ---
            if self.playback_path:
                print(f"🔊 (RAW) Playing: {self.playback_path}")
                data, file_fs = sf.read(self.playback_path)
                
                # Resample if file is not 16k (otherwise it plays in slow motion)
                if file_fs != FS:
                    print(f"   ↳ Resampling {file_fs}Hz -> {FS}Hz")
                    # Calculate new length
                    new_len = int(len(data) * FS / file_fs)
                    # Interpolate (Resample)
                    if len(data.shape) == 1: # Mono
                        data = np.interp(np.linspace(0, len(data), new_len), np.arange(len(data)), data)
                    else: # Stereo
                        left = np.interp(np.linspace(0, len(data), new_len), np.arange(len(data)), data[:, 0])
                        right = np.interp(np.linspace(0, len(data), new_len), np.arange(len(data)), data[:, 1])
                        data = np.column_stack((left, right))

                # Play non-blocking
                self.sd.play(data, FS)
            # ----------------------

            self.stream = self.sd.InputStream(
                samplerate=FS, channels=1, callback=self.callback
            )
            self.stream.start()
            self.is_recording = True
            return "🔴 RAW Recording + Playback..."
        except Exception as e:
            return f"❌ RAW Error: {e}\n(Did you stop the daemon?)"

    def stop(self):
        if not self.is_recording: return "⚠️ Nothing to stop", None
        
        print("⏹️ (RAW) Stopping...")
        self.stream.stop()
        self.stream.close()
        self.sd.stop() # Stop playback if still running
        self.is_recording = False
        
        if not self.audio_buffer: return "⚠️ Buffer empty", None
        
        full_audio = np.concatenate(self.audio_buffer)
        return "✅ RAW Audio captured", (FS, full_audio)


class DaemonRecorder(BaseRecorder):
    """Records via Reachy SDK (Requires Daemon RUNNING)."""

    def start(self):
        if self.is_recording: return "⚠️ Already in progress"
        try:
            import reachy_mini
        except ImportError:
            return "❌ Error: SDK 'reachy_mini' not found.", None

        self.audio_buffer = []
        self.is_recording = True
        
        self.thread = threading.Thread(target=self.record_loop)
        self.thread.start()
        return "🔴 DAEMON Recording + Playback..."

    def record_loop(self):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        print("🤖 (SDK) Connecting to Daemon...")
        
        try:
            from reachy_mini import ReachyMini 
            with ReachyMini(media_backend="gstreamer") as mini:
                print("✅ (SDK) Connected.")

                # 1. Start Recording
                if hasattr(mini.media, 'start_recording'):
                    mini.media.start_recording()
                
                time.sleep(0.5) 

                # 2. Play Sound
                if self.playback_path:
                    abs_path = os.path.abspath(self.playback_path)
                    print(f"🔊 (SDK) Playing: {abs_path}")
                    if hasattr(mini.media, 'play_sound'):
                        mini.media.play_sound(abs_path)
                
                # 3. Capture Loop
                while self.is_recording:
                    chunk = mini.media.get_audio_sample()
                    if chunk is not None and len(chunk) > 0:
                        self.audio_buffer.append(chunk)
                    time.sleep(0.005)

                # --- CLEANUP ---
                print("⏹️ (SDK) Cleaning up...")
                
                if hasattr(mini.media, 'stop_recording'):
                    mini.media.stop_recording()
                else:
                    print("⚠️ Warning: 'stop_recording' method not found.")

                if hasattr(mini.media, 'stop_playing'):
                    print("🔇 Stopping playback (stop_playing)...")
                    mini.media.stop_playing()
                else:
                    print("⚠️ Warning: 'stop_playing' method not found.")

        except Exception as e:
            print(f"❌ SDK Thread Error: {e}")
            self.is_recording = False
        finally:
            loop.close()

    def stop(self):
        if not self.is_recording: return "⚠️ Nothing to stop", None
        
        print("⏹️ (SDK) Stop requested...")
        self.is_recording = False 
        
        if self.thread:
            self.thread.join(timeout=2.0)
            
        if not self.audio_buffer:
            return "❌ Buffer empty.", None
            
        full_audio = np.concatenate(self.audio_buffer)
        return "✅ DAEMON Audio captured", (FS, full_audio)


# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Reachy Mini Audio Test Tool")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--raw", action="store_true", help="Use sounddevice directly")
group.add_argument("--daemon", action="store_true", help="Use SDK via Daemon")

args = parser.parse_args()

# Select Engine
recorder = None
title_str = ""

if args.raw:
    print("\n⚠️  RAW MODE: Ensure 'Reachy Mini' service is STOPPED.")
    recorder = RawRecorder()
    title_str = "Reachy Audio Test (RAW Mode)"
elif args.daemon:
    print("\n⚠️  DAEMON MODE: Ensure 'Reachy Mini' service is RUNNING.")
    recorder = DaemonRecorder()
    title_str = "Reachy Audio Test (DAEMON Mode)"

# --- HELPER FOR UI ---
def start_wrapper(file_path):
    recorder.set_playback_file(file_path)
    return recorder.start()

# --- GRADIO INTERFACE ---
with gr.Blocks(title=title_str) as demo:
    gr.Markdown(f"# 🎙️ {title_str}")
    
    with gr.Row():
        audio_in = gr.Audio(label="Optional Playback File (MP3/WAV)", type="filepath")

    status_box = gr.Textbox(label="Status", value="Ready.")
    
    with gr.Row():
        btn_start = gr.Button("🔴 RECORD & PLAY", variant="primary")
        btn_stop = gr.Button("⏹️ STOP", variant="stop")
    
    audio_out = gr.Audio(label="Recorded Result")

    btn_start.click(fn=start_wrapper, inputs=audio_in, outputs=status_box)
    btn_stop.click(fn=recorder.stop, outputs=[status_box, audio_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7880)