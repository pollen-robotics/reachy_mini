import gradio as gr
import sounddevice as sd
import numpy as np

# --- CONFIGURATION ---
FS = 16000          # 16kHz (Standard for speech recognition)
CHANNELS = 1        # Mono
DEVICE_ID = 9       # Put the ID that worked for you here (e.g., 0, 7, or 9 for default)

# Global variables for recording state
current_stream = None
audio_buffer = []

def start_recording():
    """Starts the background recording."""
    global current_stream, audio_buffer
    
    if current_stream is not None:
        return "⚠️ Already recording!", None

    print("🔴 Starting recording...")
    audio_buffer = []  # Clear memory
    
    # Callback: called continuously by the audio hardware
    def callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        # Append new audio chunk to our list
        audio_buffer.append(indata.copy())

    try:
        # Open the non-blocking InputStream
        current_stream = sd.InputStream(
            samplerate=FS, 
            channels=CHANNELS, 
            device=DEVICE_ID, 
            callback=callback
        )
        current_stream.start()
        return "🔴 RECORDING IN PROGRESS... (Press Stop to finish)", None
    except Exception as e:
        print(f"❌ Error: {e}")
        return f"❌ Error starting stream: {e}", None

def stop_recording():
    """Stops the recording and compiles the audio file."""
    global current_stream, audio_buffer
    
    if current_stream is None:
        return "⚠️ Nothing was being recorded.", None

    print("⏹️ Stopping...")
    current_stream.stop()
    current_stream.close()
    current_stream = None
    
    if not audio_buffer:
        return "⚠️ Recording buffer is empty!", None

    # Concatenate all chunks into one numpy array
    full_recording = np.concatenate(audio_buffer, axis=0)
    duration = len(full_recording) / FS
    
    print(f"✅ Done: {duration:.2f} seconds captured.")
    return "✅ Recording saved. Ready to play.", (FS, full_recording)

# --- GRADIO INTERFACE ---
with gr.Blocks(title="Reachy Recorder") as demo:
    gr.Markdown(f"# 🎙️ Reachy Recorder ({FS} Hz)")
    gr.Markdown("Manual Control: Press **Start**, speak, then press **Stop**.")
    
    status_box = gr.Textbox(label="System Status", value="Ready.", interactive=False)
    
    with gr.Row():
        start_btn = gr.Button("🔴 START Recording", variant="primary")
        stop_btn = gr.Button("⏹️ STOP Recording", variant="stop")
    
    # Audio player
    audio_output = gr.Audio(label="Playback", interactive=False)

    # Button actions
    start_btn.click(fn=start_recording, inputs=None, outputs=[status_box, audio_output])
    stop_btn.click(fn=stop_recording, inputs=None, outputs=[status_box, audio_output])

if __name__ == "__main__":
    print(f"🚀 Launching Reachy Recorder on Device {DEVICE_ID}...")
    demo.launch(server_name="0.0.0.0", server_port=7860)