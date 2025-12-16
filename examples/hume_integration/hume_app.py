import asyncio
import base64
import logging
import threading
from typing import Any, Dict, List, Optional
import numpy as np
from hume import AsyncHumeClient
from hume.empathic_voice import (
    AudioInput,
    AudioOutput,
    UserInterruption,
    AssistantProsody,
    SessionSettings,
    AudioConfiguration
)

from reachy_mini.apps.app import ReachyMiniApp
from reachy_mini.reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

class RawWrapper:
    def __init__(self, data): self.data = data
    def dict(self, **kwargs): return self.data
    def model_dump(self, **kwargs): return self.data
    def json(self, **kwargs): import json; return json.dumps(self.data)

class HumeApp(ReachyMiniApp):
    """
    Reachy Mini Application for Hume EVI 3 Integration using the Official SDK.
    """

    def __init__(self, api_key: str, config_id: Optional[str] = None):
        super().__init__()
        self.api_key = api_key
        self.config_id = config_id
        
        self.logger = logging.getLogger("hume_app")
        self.logger.setLevel(logging.INFO)
        
        # Audio state
        self._audio_buffer = []
        self._audio_buffer_primed = False
        self.sample_rate = 48000 # Default fallback

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Main application loop."""
        asyncio.run(self._async_run(reachy_mini, stop_event))

    async def _async_run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        self.logger.info("Initializing Hume Client...")
        
        client = AsyncHumeClient(api_key=self.api_key)
        
        # Detect Microphone Rate
        input_sample_rate = reachy_mini.media.audio.get_input_audio_samplerate()
        input_channels = getattr(reachy_mini.media.audio, "get_input_channels", lambda: 1)()
        self.sample_rate = input_sample_rate
        self.logger.info(f"Microphone detected: {input_sample_rate}Hz, {input_channels} channels")
        
        # Setup Session Settings using RawWrapper to force 'models'
        settings_dict = {
            "type": "session_settings",
            "audio": {
                "encoding": "linear16",
                "sample_rate": input_sample_rate,
                "channels": input_channels,
            },
            "system_prompt": "You are Reachy, a helpful and expressive robot companion.",
            "models": {
                "prosody": {} # Explicitly enable prosody inference
            }
        }
        session_settings = RawWrapper(settings_dict)
        
        reachy_mini.media.audio.start_playing()
        reachy_mini.media.audio.start_recording()

        try:
            self.logger.info("Connecting to EVI Chat...")
            async with client.empathic_voice.chat.connect(
                config_id=self.config_id,
            ) as socket:
                self.logger.info("Connected to EVI.")
                self.logger.info(f"Socket methods: {[m for m in dir(socket) if not m.startswith('_')]}")
                
                # Send Session Settings
                await socket.send_publish(session_settings)
                self.logger.info("Session settings updated.")
                
                # Create concurrent tasks
                audio_task = asyncio.create_task(self._audio_stream_task(reachy_mini, socket, stop_event))
                receive_task = asyncio.create_task(self._receive_task(reachy_mini, socket, stop_event))
                
                while not stop_event.is_set():
                    await asyncio.sleep(0.1)

                audio_task.cancel()
                receive_task.cancel()
                
        except Exception as e:
            self.logger.error(f"Error in Hume App: {e}")
        finally:
            reachy_mini.media.audio.stop_recording()
            reachy_mini.media.audio.stop_playing()
            self.logger.info("Hume App stopped.")

    async def _audio_stream_task(self, reachy_mini: ReachyMini, socket: Any, stop_event: threading.Event):
        """Streaming audio input."""
        self.logger.info("Starting audio streaming task...")
        try:
            while not stop_event.is_set():
                audio_chunk = reachy_mini.media.audio.get_audio_sample()
                if audio_chunk is not None and len(audio_chunk) > 0:
                     # Convert to PCM, Base64
                    pcm_data = (audio_chunk * 32767).astype(np.int16)
                    base64_audio = base64.b64encode(pcm_data.tobytes()).decode("utf-8")
                    
                    # SDK Method: Use send_publish with AudioInput
                    # We can use RawWrapper or AudioInput if we trust it
                    # But send_publish works with AudioInput object too?
                    # Let's use AudioInput object since it worked before
                    await socket.send_publish(AudioInput(data=base64_audio))
                
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in audio stream: {e}")

    async def _receive_task(self, reachy_mini: ReachyMini, socket: Any, stop_event: threading.Event):
        """Handling events from SDK."""
        self.logger.info("Starting message receiver task...")
        try:
            async for event in socket:
                if stop_event.is_set():
                    break
                
                self.logger.info(f"Received event: {type(event)}") # Debug log
                
                if isinstance(event, AudioOutput):
                    # Handle Audio
                    b64_data = event.data
                    if b64_data:
                        audio_bytes = base64.b64decode(b64_data)
                        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float32 = audio_int16.astype(np.float32) / 32768.0
                        # JITTER BUFFER + COMPENSATOR LOGIC
                        PRIMING_THRESHOLD = 4096
                        
                        if not self._audio_buffer_primed:
                            self._audio_buffer.append(audio_float32)
                            total_samples = sum(len(c) for c in self._audio_buffer)
                            
                            if total_samples >= PRIMING_THRESHOLD:
                                combined_audio = np.concatenate(self._audio_buffer)
                                await self._push_audio_safe(reachy_mini, combined_audio)
                                self._audio_buffer = []
                                self._audio_buffer_primed = True
                                self.logger.info("Buffer primed.")
                        else:
                            await self._push_audio_safe(reachy_mini, audio_float32)
                            
                elif isinstance(event, UserInterruption):
                    self.logger.info("Interruption detected.")
                    self._audio_buffer = []
                    self._audio_buffer_primed = False
                    reachy_mini.media.audio.stop_playing()
                    reachy_mini.media.audio.start_playing()
                    
                elif isinstance(event, AssistantProsody):
                    # Handle Prosody
                    scores = event.models.prosody.scores
                    if scores:
                        # SDK returns object? Or dict?
                        # Usually custom object. converted to dict for mapping?
                        # Assuming it behaves dict-like or we iterate
                        # If it is an object, we access attributes?
                        # Let's try converting to dict if possible or accessing assuming Dict[str, float]
                        # Actually most likely: event.models.prosody.scores is { 'Joy': 0.5, ... } 
                        self._handle_emotions(reachy_mini, scores)
                    else:
                        self.logger.warning("AssistantProsody received but no scores.")
                        
                # Handle other types if needed (UserMessage etc)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in receiver: {e}")

    async def _push_audio_safe(self, reachy_mini: ReachyMini, audio_float32: np.ndarray):
        """Push audio with optional resampling."""
        output_sample_rate = getattr(reachy_mini.media.audio, "get_output_audio_samplerate", lambda: 48000)()
        hume_sample_rate = getattr(self, "sample_rate", 24000)
        
        if output_sample_rate != hume_sample_rate:
             num_samples_in = len(audio_float32)
             num_samples_out = int(num_samples_in * output_sample_rate / hume_sample_rate)
             x_in = np.arange(num_samples_in)
             x_out = np.linspace(0, num_samples_in - 1, num_samples_out)
             audio_float32 = np.interp(x_out, x_in, audio_float32).astype(np.float32)
             
        reachy_mini.media.audio.push_audio_sample(audio_float32)

    def _handle_emotions(self, reachy_mini: ReachyMini, scores: Any): # Types might vary
        # ... Reuse the Emotion mapping logic ...
        # Assuming scores is a dict or iterable of (name, value)
        if hasattr(scores, "items"):
            items = scores.items()
        elif isinstance(scores, dict):
            items = scores.items()
        else:
            # Fallback if SDK returns object list?
            # User will debug if crash.
            return
            
        top_emotion = max(items, key=lambda x: x[1])
        emotion_name, score = top_emotion
        
        if score < 0.3: return
        self.logger.info(f"Emotion: {emotion_name} ({score:.2f})")
        
        if emotion_name in ["Amusement", "Excitement", "Joy", "Ecstasy"]:
             reachy_mini.set_target(head=create_head_pose(pitch=-5, z=20, degrees=True, mm=True), antennas=[0.5, -0.5])
        elif emotion_name in ["Sadness", "Distress", "Pain", "Disappointment"]:
             reachy_mini.set_target(head=create_head_pose(pitch=10, z=-10, degrees=True, mm=True), antennas=[-0.8, 0.8])
        elif emotion_name in ["Confusion", "Doubt", "Awkwardness"]:
             reachy_mini.set_target(head=create_head_pose(roll=10, z=10, degrees=True, mm=True), antennas=[0.2, 0.8])
        elif emotion_name == "Neutral":
             reachy_mini.set_target(head=create_head_pose(z=15, degrees=True, mm=True), antennas=[0.0, 0.0])

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hume EVI Integration SDK")
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--config-id", type=str)
    args = parser.parse_args()
    
    app = HumeApp(api_key=args.api_key, config_id=args.config_id)
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
