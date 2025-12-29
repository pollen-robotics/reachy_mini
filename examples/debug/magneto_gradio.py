import gradio as gr
import numpy as np
import threading
import time
import argparse
import sys

# --- CONFIGURATION ---
FS = 16000  # Standard sample rate (16kHz)

# --- RECORDER CLASSES ---

class BaseRecorder:
    """Generic interface for recording."""
    def __init__(self):
        self.is_recording = False
        self.audio_buffer = []
        self.thread = None

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
        
        print("🎙️ (RAW) Opening direct audio stream...")
        self.audio_buffer = []
        try:
            # Tries to use the 'default' device which handles conversion via .asoundrc
            self.stream = self.sd.InputStream(
                samplerate=FS, channels=1, callback=self.callback
            )
            self.stream.start()
            self.is_recording = True
            return "🔴 RAW Recording in progress... (Daemon must be STOPPED)"
        except Exception as e:
            return f"❌ RAW Error: {e}\n(Did you stop the daemon? 'sudo systemctl stop reachy_mini_kinesist')"

    def stop(self):
        if not self.is_recording: return "⚠️ Nothing to stop", None
        
        print("⏹️ (RAW) Stopping...")
        self.stream.stop()
        self.stream.close()
        self.is_recording = False
        
        if not self.audio_buffer: return "⚠️ Buffer empty", None
        
        full_audio = np.concatenate(self.audio_buffer)
        return "✅ RAW Audio captured", (FS, full_audio)


class DaemonRecorder(BaseRecorder):
    """Records via Reachy SDK (Requires Daemon RUNNING)."""
    def start(self):
        if self.is_recording: return "⚠️ Already in progress"
        
        # Local import to avoid errors if running in RAW mode without SDK installed
        from reachy_mini import ReachyMini
        self.ReachyMini = ReachyMini

        self.audio_buffer = []
        self.is_recording = True
        
        def record_loop():
            print("🤖 (SDK) Connecting to Daemon...")
            try:
                with self.ReachyMini() as mini:
                    print("✅ (SDK) Connected. Capturing...")
                    while self.is_recording:
                        # Official SDK method
                        if hasattr(mini, 'media'):
                            chunk = mini.media.get_audio_sample()
                            if chunk is not None and len(chunk) > 0:
                                self.audio_buffer.append(chunk)
                        time.sleep(0.005)
            except Exception as e:
                print(f"❌ SDK Thread Error: {e}")
                self.is_recording = False

        self.thread = threading.Thread(target=record_loop)
        self.thread.start()
        return "🔴 DAEMON Recording in progress... (Daemon must be RUNNING)"

    def stop(self):
        if not self.is_recording: return "⚠️ Nothing to stop", None
        
        print("⏹️ (SDK) Stop requested...")
        self.is_recording = False # Stops the while loop
        
        if self.thread:
            self.thread.join(timeout=2.0)
            
        if not self.audio_buffer:
            return "❌ Buffer empty (Daemon sent nothing). Check the service.", None
            
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
    print("\n⚠️  RAW MODE ENABLED: Ensure 'Reachy Mini' service is STOPPED.")
    recorder = RawRecorder()
    title_str = "Reachy Audio Test (RAW Mode - Direct Hardware)"
elif args.daemon:
    print("\n⚠️  DAEMON MODE ENABLED: Ensure 'Reachy Mini' service is RUNNING.")
    recorder = DaemonRecorder()
    title_str = "Reachy Audio Test (DAEMON Mode - Via SDK)"

# --- GRADIO INTERFACE ---
with gr.Blocks(title=title_str) as demo:
    gr.Markdown(f"# 🎙️ {title_str}")
    
    status_box = gr.Textbox(label="Status", value="Ready.")
    
    with gr.Row():
        btn_start = gr.Button("🔴 RECORD", variant="primary")
        btn_stop = gr.Button("⏹️ STOP", variant="stop")
    
    audio_out = gr.Audio(label="Playback")

    btn_start.click(fn=recorder.start, outputs=status_box)
    btn_stop.click(fn=recorder.stop, outputs=[status_box, audio_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)