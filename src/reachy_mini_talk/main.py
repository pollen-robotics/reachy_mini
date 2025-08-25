from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import time
import warnings
from threading import Thread

import cv2
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from websockets import ConnectionClosedError, ConnectionClosedOK

from reachy_mini.reachy_mini import IMAGE_SIZE
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.camera import find_camera
from scipy.spatial.transform import Rotation

from reachy_mini_talk.head_tracker import HeadTracker
from reachy_mini_talk.prompts import SESSION_INSTRUCTIONS
from reachy_mini_talk.tools import (
    Deps,
    OpenAIImageHandler,
    TOOL_SPECS,
    dispatch_tool_call,
)
from reachy_mini_talk.audio import AudioSync, AudioConfig, pcm_to_b64
from reachy_mini_talk.movement import MovementManager
from reachy_mini_talk.gstreamer import GstPlayer, GstRecorder

# env + logging
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress WebRTC warnings
warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

# Reduce logging noise
logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("fastrtc").setLevel(logging.ERROR)
logging.getLogger("aioice").setLevel(logging.WARNING)


# Read from .env
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))
SIM = os.getenv("SIM", "false").lower() in ("true", "1", "yes", "on")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-realtime-preview")

HEAD_TRACKING = os.getenv("HEAD_TRACKING", "false").lower() in (
    "true",
    "1",
    "yes",
    "on",
)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logger.error("OPENAI_API_KEY not set! Please add it to your .env file.")
    raise RuntimeError("OPENAI_API_KEY missing")
masked = (API_KEY[:6] + "..." + API_KEY[-4:]) if len(API_KEY) >= 12 else "<short>"
logger.info("OPENAI_API_KEY loaded (prefix): %s", masked)

# hardware / IO
current_robot = ReachyMini()
movement_manager = MovementManager(current_robot=current_robot)
movement_manager.is_head_tracking = HEAD_TRACKING
logger.info("Head tracking %s", "ENABLED" if HEAD_TRACKING else "DISABLED")
robot_is_speaking = asyncio.Event()
speaking_queue = asyncio.Queue()


# init camera
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

if SIM:
    # Default build-in camera in SIM
    # TODO: please, test on Linux and Windows
    camera = cv2.VideoCapture(
        0, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else 0
    )
else:
    if sys.platform == "darwin":
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
        if not camera or not camera.isOpened():
            logger.warning(
                "Camera %d failed with AVFoundation; trying default backend",
                CAMERA_INDEX,
            )
            camera = cv2.VideoCapture(CAMERA_INDEX)
    else:
        camera = find_camera()

if not camera or not camera.isOpened():
    logger.error("Camera failed to open (index=%s)", 0 if SIM else CAMERA_INDEX)
else:
    logger.info(
        "Camera ready (index=%s)%s", 0 if SIM else CAMERA_INDEX, " [SIM]" if SIM else ""
    )


# Constants
BACKOFF_START_S = 1.0
BACKOFF_MAX_S = 30.0
LOOP_SLEEP_S = 0.05


# tool deps
deps = Deps(
    reachy_mini=current_robot,
    create_head_pose=create_head_pose,
    camera=camera,
    image_handler=OpenAIImageHandler(),
)

# audio sync
audio_sync = AudioSync(
    AudioConfig(output_sample_rate=SAMPLE_RATE),
    set_offsets=movement_manager.set_offsets,
)


class OpenAIRealtimeHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=SAMPLE_RATE,
            input_sample_rate=SAMPLE_RATE,
        )
        self.client: AsyncOpenAI | None = None
        self.connection = None
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self._pending_calls: dict[str, dict] = {}  # call_id -> {name, args_buf}
        self._stop = False
        self._started_audio = False
        self._connection_ready = False
        self._speech_start_time = 0.0

    def copy(self):
        return OpenAIRealtimeHandler()

    async def start_up(self):
        if not self._started_audio:
            audio_sync.start()
            self._started_audio = True

        if self.client is None:
            logger.info("Realtime start_up: creating AsyncOpenAI client...")
            self.client = AsyncOpenAI(api_key=API_KEY)

        backoff = BACKOFF_START_S
        while not self._stop:
            try:
                async with self.client.beta.realtime.connect(
                    model=MODEL_NAME
                ) as rt_connection:
                    self.connection = rt_connection
                    self._connection_ready = False
                    self._pending_calls.clear()

                    # configure session
                    await rt_connection.session.update(
                        session={
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.6,  # Higher threshold = less sensitive
                                "prefix_padding_ms": 300,  # More padding before speech
                                "silence_duration_ms": 800,  # Longer silence before detecting end
                            },
                            "voice": "ballad",
                            "instructions": SESSION_INSTRUCTIONS,
                            "input_audio_transcription": {
                                "model": "whisper-1",
                                "language": "en",
                            },
                            "tools": TOOL_SPECS,
                            "tool_choice": "auto",
                            "temperature": 0.7,
                        }
                    )

                    # Wait for session to be configured
                    await asyncio.sleep(0.2)

                    # Add system message with even stronger brevity emphasis
                    await rt_connection.conversation.item.create(
                        item={
                            "type": "message",
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"{SESSION_INSTRUCTIONS}\n\nIMPORTANT: Always keep responses under 25 words. Be extremely concise.",
                                }
                            ],
                        }
                    )

                    self._connection_ready = True

                    logger.info(
                        "Session updated: tools=%d, voice=%s, vad=improved",
                        len(TOOL_SPECS),
                        "ballad",
                    )

                    logger.info("Realtime event loop started with improved VAD")
                    backoff = BACKOFF_START_S

                    async for event in rt_connection:
                        event_type = getattr(event, "type", None)
                        logger.debug("RT event: %s", event_type)

                        # Enhanced speech state tracking
                        if event_type == "input_audio_buffer.speech_started":
                            # Only process user speech if robot isn't currently speaking
                            if not robot_is_speaking.is_set():
                                audio_sync.on_input_speech_started()
                                logger.info("User speech detected (robot not speaking)")
                            else:
                                logger.info(
                                    "Ignoring speech detection - robot is speaking"
                                )

                        elif event_type == "response.started":
                            self._speech_start_time = time.time()
                            audio_sync.on_response_started()
                            logger.info("Robot started speaking")

                        elif event_type in (
                            "response.audio.completed",
                            "response.completed",
                            "response.audio.done",
                        ):
                            logger.info(f"Robot finished speaking {event_type}")

                        elif (
                            event_type
                            == "conversation.item.input_audio_transcription.completed"
                        ):
                            await self.output_queue.put(
                                AdditionalOutputs(
                                    {"role": "user", "content": event.transcript}
                                )
                            )

                        elif event_type == "response.audio_transcript.done":
                            await self.output_queue.put(
                                AdditionalOutputs(
                                    {"role": "assistant", "content": event.transcript}
                                )
                            )

                        # audio streaming
                        if event_type == "response.audio.delta":
                            robot_is_speaking.set()
                            # block mic from recording for given time, for each audio delta
                            speaking_queue.put_nowait(0.25) 
                            audio_sync.on_response_audio_delta(
                                getattr(event, "delta", b"")
                            )

                        # tool-call handling
                        if event_type == "response.output_item.added":
                            output_item = getattr(event, "item", None)
                            if (
                                output_item
                                and getattr(output_item, "type", "") == "function_call"
                            ):
                                call_id = getattr(output_item, "call_id", None)
                                name = getattr(output_item, "name", None)
                                if call_id and name:
                                    logger.info(
                                        "Tool call: %s (call_id=%s)", name, call_id
                                    )
                                    self._pending_calls[call_id] = {
                                        "name": name,
                                        "args_buf": "",
                                    }

                        elif event_type == "response.function_call_arguments.delta":
                            call_id = getattr(event, "call_id", None)
                            chunk = getattr(event, "delta", "")
                            if call_id in self._pending_calls and chunk:
                                self._pending_calls[call_id]["args_buf"] += chunk

                        elif event_type == "response.function_call_arguments.done":
                            call_id = getattr(event, "call_id", None)
                            call_info = self._pending_calls.get(call_id)
                            if not call_info:
                                continue
                            tool_name = call_info["name"]
                            args_json_str = call_info["args_buf"] or "{}"

                            try:
                                tool_result = await dispatch_tool_call(
                                    tool_name, args_json_str, deps
                                )
                            except Exception as e:
                                logger.exception("Tool %s failed", tool_name)
                                tool_result = {"error": str(e)}

                            await rt_connection.conversation.item.create(
                                item={
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": json.dumps(tool_result),
                                }
                            )
                            self._pending_calls.pop(call_id, None)

                        # server errors
                        if event_type == "error":
                            err = getattr(event, "error", None)
                            msg = getattr(
                                err, "message", str(err) if err else "unknown error"
                            )
                            logger.error("Realtime error: %s", msg)
                            await self.output_queue.put(
                                AdditionalOutputs(
                                    {"role": "assistant", "content": f"[error] {msg}"}
                                )
                            )

            except (ConnectionClosedOK, ConnectionClosedError) as e:
                if self._stop:
                    break
                logger.warning(
                    "Connection closed (%s). Reconnecting…",
                    getattr(e, "code", "no-code"),
                )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Realtime loop error; will reconnect")
            finally:
                self.connection = None
                self._connection_ready = False
                self._pending_calls.clear()

            # Exponential backoff
            delay = min(backoff, BACKOFF_MAX_S) + random.uniform(0, 0.5)
            logger.info("Reconnect in %.1fs…", delay)
            await asyncio.sleep(delay)
            backoff = min(backoff * 2.0, BACKOFF_MAX_S)

    async def receive(self, frame: bytes) -> None:
        """Mic frames from fastrtc."""
        # Don't send mic audio while robot is speaking (simple echo cancellation)
        if robot_is_speaking.is_set() or not self._connection_ready:
            return

        mic_samples = np.frombuffer(frame, dtype=np.int16).squeeze()
        audio_b64 = pcm_to_b64(mic_samples)

        try:
            await self.connection.input_audio_buffer.append(audio=audio_b64)
        except (ConnectionClosedOK, ConnectionClosedError):
            pass

    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        """Return audio for playback or chat outputs."""
        try:
            sample_rate, pcm_frame = audio_sync.playback_q.get_nowait()
            logger.debug(
                "Emitting playback frame (sr=%d, n=%d)", sample_rate, pcm_frame.size
            )
            return (sample_rate, pcm_frame)
        except asyncio.QueueEmpty:
            pass
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        logger.info("Shutdown: closing connections and audio")
        self._stop = True
        if self.connection:
            try:
                await self.connection.close()
            except Exception:
                logger.exception("Error closing realtime connection")
            finally:
                self.connection = None
                self._connection_ready = False
        await audio_sync.stop()


def update_chatbot(chatbot: list[dict], response: dict) -> list[dict]:
    chatbot.append(response)
    return chatbot


async def receive_loop(recorder: GstRecorder, openai: OpenAIRealtimeHandler) -> None:
    logger.info("Starting receive loop")
    while True:
        data = recorder.get_sample()
        if data is not None:
            await openai.receive(data)
        await asyncio.sleep(0)  # Prevent busy waiting


async def emit_loop(player: GstPlayer, openai: OpenAIRealtimeHandler) -> None:
    while True:
        data = await openai.emit()
        if isinstance(data, AdditionalOutputs):
            for msg in data.args:
                logger.info(f"role : {msg['role']}: content: {msg['content']}")

        elif isinstance(data, tuple):
            _, frame = data
            player.push_sample(frame.tobytes())

        else:
            pass
        await asyncio.sleep(0)  # Prevent busy waiting


async def control_mic_loop():
    # Control mic to prevent echo, blocks mic for given time
    while True:
        try:
            time = speaking_queue.get_nowait()
        except asyncio.QueueEmpty:
            robot_is_speaking.clear()
            audio_sync.on_response_completed()
            await asyncio.sleep(0)
            continue
        
        await asyncio.sleep(time)


chatbot = gr.Chatbot(type="messages")

async def audio_stream():
    openai = OpenAIRealtimeHandler()
    recorder = GstRecorder()
    recorder.record()
    player = GstPlayer()
    player.play()
    logger.info("Starting main audio loop. You can start to speak")
    
    start_up_task = asyncio.create_task(openai.start_up())
    emit_task = asyncio.create_task(emit_loop(player, openai))
    receive_task = asyncio.create_task(receive_loop(recorder, openai))
    mic_task = asyncio.create_task(control_mic_loop())

    await asyncio.gather(start_up_task, emit_task, receive_task, mic_task)
    await openai.shutdown()
    recorder.stop()
    player.stop()


def main():
    try:
        audio_thread = Thread(target=lambda: asyncio.run(audio_stream()), daemon=True)
        audio_thread.start()

        time.sleep(1.0)  # Let UI start

        # Initialize neutral pose
        movement_manager.set_neutral()
        logger.info("Initialized with neutral pose")

        # Head tracker init (once)
        head_tracker: HeadTracker | None = None
        if movement_manager.is_head_tracking and not SIM:
            head_tracker = HeadTracker()
            logger.info("HeadTracker loaded from: %s", HeadTracker.__module__)
        elif movement_manager.is_head_tracking and SIM:
            logger.warning("SIM mode active -> head tracking disabled")

        logger.info("Starting main control loop...")
        last_log_ts = 0.0
        debug_frame_count = 0

        while True:
            debug_frame_count += 1
            current_time = time.time()

            # Head tracking
            if movement_manager.is_head_tracking and head_tracker is not None:
                success, im = camera.read()
                if not success:
                    if current_time - last_log_ts > 1.5:
                        logger.warning("Camera read failed")
                        last_log_ts = current_time
                else:
                    eye_center, _ = head_tracker.get_head_position(im) # as [-1, 1]

                    if eye_center is not None:
                        # Rescale target position into IMAGE_SIZE coordinates 
                        w, h = IMAGE_SIZE
                        eye_center = (eye_center + 1) / 2
                        eye_center[0] *= w
                        eye_center[1] *= h

                        current_head_pose = (
                            movement_manager.current_robot.look_at_image(
                                *eye_center, duration=0.0, apply=False
                            )
                        )
                        movement_manager.current_head_pose = current_head_pose
            # Pose calculation
            try:
                current_x, current_y, current_z = movement_manager.current_head_pose[
                    :3, 3
                ]
                current_roll, current_pitch, current_yaw = Rotation.from_matrix(
                    movement_manager.current_head_pose[:3, :3]
                ).as_euler("xyz", degrees=False)

                if debug_frame_count % 50 == 0:
                    logger.debug(
                        f"Current pose XYZ: {current_x:.3f}, {current_y:.3f}, {current_z:.3f}"
                    )
                    logger.debug(
                        f"Current angles: roll={current_roll:.3f}, pitch={current_pitch:.3f}, yaw={current_yaw:.3f}"
                    )

            except Exception:
                logger.exception("Invalid pose; resetting")
                movement_manager.reset_head_pose()
                current_x, current_y, current_z = movement_manager.current_head_pose[
                    :3, 3
                ]
                current_roll = current_pitch = current_yaw = 0.0

            # Movement check
            moving = (
                time.monotonic() - movement_manager.moving_start
                < movement_manager.moving_for
            )

            if debug_frame_count % 50 == 0:
                logger.debug(f"Robot moving: {moving}")

            # Apply speech offsets when not moving
            if not moving:
                try:
                    head_pose = create_head_pose(
                        x=current_x + movement_manager.speech_head_offsets[0],
                        y=current_y + movement_manager.speech_head_offsets[1],
                        z=current_z + movement_manager.speech_head_offsets[2],
                        roll=current_roll + movement_manager.speech_head_offsets[3],
                        pitch=current_pitch + movement_manager.speech_head_offsets[4],
                        yaw=current_yaw + movement_manager.speech_head_offsets[5],
                        degrees=False,
                        mm=False,
                    )

                    if debug_frame_count % 50 == 0:
                        logger.debug(
                            f"Final head pose with offsets: {head_pose[:3, 3]}"
                        )
                        logger.debug(
                            f"Speech offsets: {movement_manager.speech_head_offsets}"
                        )

                    current_robot.set_target(head=head_pose, antennas=(0.0, 0.0))

                    if debug_frame_count % 50 == 0:
                        logger.debug(f"Sent pose to robot successfully")

                except Exception as e:
                    logger.debug(f"Failed to set robot target: {e}")

            time.sleep(LOOP_SLEEP_S)

    except KeyboardInterrupt:
        logger.info("Shutting down on interrupt...")
        exit(0)
    except Exception:
        logger.exception("Error in main loop")

if __name__ == "__main__":
    main()
