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
        
        # Audio state - using queue for proper buffering
        self._audio_queue = None  # Will be initialized in async context
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
        
        # Setup Session Settings - prosody is automatically enabled in EVI 3
        settings_dict = {
            "type": "session_settings",
            "audio": {
                "encoding": "linear16",
                "sample_rate": input_sample_rate,
                "channels": input_channels,
            },
            "system_prompt": "You are Reachy, a helpful and expressive robot companion.",
        }
        session_settings = RawWrapper(settings_dict)

        # Initialize audio queue for proper buffering
        self._audio_queue = asyncio.Queue(maxsize=50)
        
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
                playback_task = asyncio.create_task(self._audio_playback_task(reachy_mini, stop_event))

                while not stop_event.is_set():
                    await asyncio.sleep(0.1)

                audio_task.cancel()
                receive_task.cancel()
                playback_task.cancel()
                
        except Exception as e:
            self.logger.error(f"Error in Hume App: {e}")
        finally:
            reachy_mini.media.audio.stop_recording()
            reachy_mini.media.audio.stop_playing()
            self.logger.info("Hume App stopped.")

    async def _audio_stream_task(self, reachy_mini: ReachyMini, socket: Any, stop_event: threading.Event):
        """Streaming audio input - using 20ms buffer window as recommended by Hume."""
        self.logger.info("Starting audio streaming task...")
        try:
            while not stop_event.is_set():
                audio_chunk = reachy_mini.media.audio.get_audio_sample()
                if audio_chunk is not None and len(audio_chunk) > 0:
                     # Convert to PCM, Base64
                    pcm_data = (audio_chunk * 32767).astype(np.int16)
                    base64_audio = base64.b64encode(pcm_data.tobytes()).decode("utf-8")

                    # Send audio input to Hume
                    await socket.send_publish(AudioInput(data=base64_audio))

                # Hume recommends 20ms buffer window for native apps
                await asyncio.sleep(0.02)
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

                event_type = type(event).__name__
                self.logger.debug(f"Received event: {event_type}")

                if isinstance(event, AudioOutput):
                    # Queue audio instead of playing directly to avoid clicking
                    b64_data = event.data
                    if b64_data:
                        audio_bytes = base64.b64decode(b64_data)
                        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float32 = audio_int16.astype(np.float32) / 32768.0

                        # Add to queue for playback task to handle
                        try:
                            await asyncio.wait_for(self._audio_queue.put(audio_float32), timeout=1.0)
                        except asyncio.TimeoutError:
                            self.logger.warning("Audio queue full, dropping chunk")

                elif isinstance(event, UserInterruption):
                    self.logger.info("Interruption detected - clearing audio queue")
                    # Clear the queue
                    while not self._audio_queue.empty():
                        try:
                            self._audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    reachy_mini.media.audio.stop_playing()
                    reachy_mini.media.audio.start_playing()

                elif isinstance(event, AssistantProsody):
                    # Handle Prosody with enhanced logging
                    self.logger.info(f"✓ AssistantProsody event received!")
                    try:
                        if hasattr(event, 'models') and hasattr(event.models, 'prosody'):
                            scores = event.models.prosody.scores
                            if scores:
                                self.logger.info(f"Prosody scores structure: {type(scores)}")
                                self._handle_emotions(reachy_mini, scores)
                            else:
                                self.logger.warning("AssistantProsody received but scores are None/empty")
                        else:
                            self.logger.warning(f"AssistantProsody structure unexpected: {dir(event)}")
                    except Exception as e:
                        self.logger.error(f"Error processing AssistantProsody: {e}", exc_info=True)

                # Log other event types for debugging
                else:
                    self.logger.debug(f"Other event type: {event_type}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in receiver: {e}", exc_info=True)

    async def _audio_playback_task(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Dedicated task for smooth audio playback from queue.

        This prevents clicking by ensuring audio is queued and played at the correct rate,
        rather than playing chunks immediately as they arrive from Hume.
        """
        self.logger.info("Starting audio playback task...")
        PRIMING_THRESHOLD = 4096  # Prime buffer before starting playback
        primed = False
        buffer = []

        try:
            while not stop_event.is_set():
                try:
                    # Get audio from queue
                    audio_chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)

                    if not primed:
                        # Build up initial buffer to prevent underruns
                        buffer.append(audio_chunk)
                        total_samples = sum(len(c) for c in buffer)

                        if total_samples >= PRIMING_THRESHOLD:
                            combined_audio = np.concatenate(buffer)
                            await self._push_audio_safe(reachy_mini, combined_audio)
                            buffer = []
                            primed = True
                            self.logger.info("Audio buffer primed - starting playback")
                    else:
                        # Normal playback - push audio immediately
                        await self._push_audio_safe(reachy_mini, audio_chunk)

                except asyncio.TimeoutError:
                    # No audio available, continue waiting
                    continue

        except asyncio.CancelledError:
            self.logger.info("Audio playback task cancelled")
        except Exception as e:
            self.logger.error(f"Error in audio playback: {e}", exc_info=True)

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

    def _handle_emotions(self, reachy_mini: ReachyMini, scores: Any):
        """Process prosody scores and animate robot based on detected emotions."""
        try:
            # Handle different score formats from Hume SDK
            if hasattr(scores, "items"):
                items = scores.items()
            elif isinstance(scores, dict):
                items = scores.items()
            else:
                self.logger.warning(f"Unexpected scores format: {type(scores)}")
                return

            if not items:
                self.logger.warning("No emotion items found in scores")
                return

            # Find top emotion
            top_emotion = max(items, key=lambda x: x[1])
            emotion_name, score = top_emotion

            self.logger.info(f"Top emotion: {emotion_name} (score: {score:.2f})")

            # Only animate if score is significant
            if score < 0.3:
                self.logger.debug(f"Emotion score too low ({score:.2f}), skipping animation")
                return

            # Animate robot based on emotion
            if emotion_name in ["Amusement", "Excitement", "Joy", "Ecstasy"]:
                self.logger.info(f"🎉 Animating happy emotion: {emotion_name}")
                reachy_mini.set_target(
                    head=create_head_pose(pitch=-5, z=20, degrees=True, mm=True),
                    antennas=[0.5, -0.5]
                )
            elif emotion_name in ["Sadness", "Distress", "Pain", "Disappointment"]:
                self.logger.info(f"😢 Animating sad emotion: {emotion_name}")
                reachy_mini.set_target(
                    head=create_head_pose(pitch=10, z=-10, degrees=True, mm=True),
                    antennas=[-0.8, 0.8]
                )
            elif emotion_name in ["Confusion", "Doubt", "Awkwardness"]:
                self.logger.info(f"🤔 Animating confused emotion: {emotion_name}")
                reachy_mini.set_target(
                    head=create_head_pose(roll=10, z=10, degrees=True, mm=True),
                    antennas=[0.2, 0.8]
                )
            elif emotion_name == "Neutral":
                self.logger.info(f"😐 Animating neutral emotion")
                reachy_mini.set_target(
                    head=create_head_pose(z=15, degrees=True, mm=True),
                    antennas=[0.0, 0.0]
                )
            else:
                self.logger.debug(f"No animation defined for emotion: {emotion_name}")

        except Exception as e:
            self.logger.error(f"Error handling emotions: {e}", exc_info=True)

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
