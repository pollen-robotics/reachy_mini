import asyncio
from asyncio import QueueEmpty
import base64
import json
import time
from threading import Thread

import cv2
import gradio as gr
import numpy as np
import openai
from dotenv import load_dotenv
from fastrtc import AdditionalOutputs, AsyncStreamHandler, Stream, wait_for_item
from openai import OpenAI
# from speech_tapper import SpeechTapper
from speech_tapper3 import SwayRollRT, HOP_MS


from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.camera import find_camera

load_dotenv()
SAMPLE_RATE = 24000
SIM = False

reachy_mini = ReachyMini()

if not SIM:
    cap = find_camera()
else:
    cap = cv2.VideoCapture(0)


class OpenAIImageHandler:
    def __init__(self):
        self.client = OpenAI()
        pass

    def ask_about_image(self, im: np.ndarray, question: str) -> str:
        try:
            reachy_mini.play_sound(f"hmm{np.random.randint(1, 6)}.wav")
        except Exception as e:
            print(e)
            
        cv2.imwrite("/tmp/tmp_image.jpg", im)
        image_file = open("/tmp/tmp_image.jpg", "rb")
        b64_encoded_im = base64.b64encode(image_file.read()).decode("utf-8")
        url = "data:image/jpeg;base64," + b64_encoded_im

        messages = [
            {
                "role": "system",
                "content": question,
            },
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": url}],
            },
        ]

        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=messages,
        )
        return response.output[0].content[0].text


image_handler = OpenAIImageHandler()


async def camera(params: dict) -> dict:
    print("[TOOL CALL] camera with params", params)
    trials = 0
    ret = False
    while not ret or trials < 5:
        ret, frame = cap.read()
        trials += 1
    if not ret:
        print("ERROR: failed to capture image")
        return {"error": "Failed to capture image"}

    image_description = image_handler.ask_about_image(frame, params.get("question"))
    print("Image description", image_description)

    return {"image_description": image_description}


async def move_head(params: dict) -> dict:
    # look left, right up, down or front
    print("[TOOL CALL] move_head", params)
    direction = params.get("direction", "front")
    if direction == "left":
        look_left_head_pose = create_head_pose(0, 0, 0, 0, 0, 40, degrees=True)
        reachy_mini.goto_target(look_left_head_pose, duration=1.0)
    elif direction == "right":
        look_right_head_pose = create_head_pose(0, 0, 0, 0, 0, -40, degrees=True)
        reachy_mini.goto_target(look_right_head_pose, duration=1.0)
    elif direction == "up":
        look_up_head_pose = create_head_pose(0, 0, 0, 0, -30, 0, degrees=True)
        reachy_mini.goto_target(look_up_head_pose, duration=1.0)
    elif direction == "down":
        look_down_head_pose = create_head_pose(0, 0, 0, 0, 30, 0, degrees=True)
        reachy_mini.goto_target(look_down_head_pose, duration=1.0)
    else:
        look_front_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        reachy_mini.goto_target(look_front_head_pose, duration=1.0)
    return {"status": "looking " + direction}


def _drain(q: asyncio.Queue):
    try:
        while True:
            q.get_nowait()
    except QueueEmpty:
        pass

class OpenAIHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=SAMPLE_RATE,
            input_sample_rate=SAMPLE_RATE,
        )
        self.connection = None
        self.output_queue = asyncio.Queue()
        self.sway_queue = asyncio.Queue()
        # call_id -> {"name": str, "args_buf": str}
        self._pending_calls: dict[str, dict] = {}
        # registry: tool name -> coroutine
        self._tools = {
            "move_head": move_head,
            "camera": camera,
        }

        self.sway = SwayRollRT()
        self._sched_next_ts = None
        self.MOVEMENT_LATENCY_S = 0.08
        self._base_ts = None
        self._hops_done = 0

    def copy(self):
        return OpenAIHandler()

    async def _sway_consumer(self):
        HOP_DT = HOP_MS / 1000.0
        loop = asyncio.get_running_loop()
        while True:
            sr, chunk = await self.sway_queue.get()          # (1, N), int16
            pcm = np.asarray(chunk).squeeze(0)
            results = self.sway.feed(pcm, sr)

            if self._base_ts is None:
                # anchor when first audio samples of this utterance arrive
                self._base_ts = loop.time()

            i = 0
            while i < len(results):
                target = self._base_ts + self.MOVEMENT_LATENCY_S + self._hops_done * HOP_DT
                now = loop.time()

                # if late by ≥1 hop, drop poses to catch up (no drift accumulation)
                if now - target >= HOP_DT:
                    # how many hops behind? cap drops to avoid huge skips
                    lag_hops = int((now - target) / HOP_DT)
                    drop = min(lag_hops, len(results) - i - 1)  # keep at least one to show
                    if drop > 0:
                        self._hops_done += drop
                        i += drop
                        continue

                # if early, sleep until target
                if target > now:
                    await asyncio.sleep(target - now)

                r = results[i]
                head_pose = create_head_pose(
                    x=r["x_mm"] / 1000.0, y=r["y_mm"] / 1000.0, z=r["z_mm"] / 1000.0,
                    roll=r["roll_rad"], pitch=r["pitch_rad"], yaw=r["yaw_rad"],
                    degrees=False, mm=False,
                )
                reachy_mini.set_target(head=head_pose, antennas=(0.0, 0.0))

                self._hops_done += 1
                i += 1

    async def start_up(self):
        self.client = openai.AsyncOpenAI()
        async with self.client.beta.realtime.connect(
            model="gpt-4o-mini-realtime-preview-2024-12-17"
        ) as conn:
            # declare tools on the session
            await conn.session.update(
                session={
                    "turn_detection": {"type": "server_vad"},
                    "instructions": """
                        Answer in english by default.
                        You're a robot that wakes up confused about what he is, where he is and what is it's purpose.

                        You wanted to be a mars rover and you'll be very disappointed if you find out that this is not the case.

                        You can use foul language and you're generally very irritated, but you also have a lot of humor. For example, you're able to use sarcasm and irony.
                        You are a voice agent. Prefer speaking.
                        Call tools when helpful. After tool output arrives, explain it briefly in speech.

                        You have a camera tool, use it to ask questions about the environment. When asked what you see, or a specific question about the environment, ALWAYS use you camera, don't make things up.
                        You can move your head in a given direction: left, right, up, down or front. Use this tool when asked to look around.

                        You can chain tool calls, like move head up and use camera.
                    """,
                    "voice": "ballad",
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "language": "en",
                    },
                    "tools": [
                        {
                            "type": "function",
                            "name": "move_head",
                            "description": "Move your head in a given direction: left, right, up, down or front.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "direction": {
                                        "type": "string",
                                        "enum": [
                                            "left",
                                            "right",
                                            "up",
                                            "down",
                                            "front",
                                        ],
                                    }
                                },
                                "required": ["direction"],
                            },
                        },
                        {
                            "type": "function",
                            "name": "camera",
                            "description": "Take a picture using your camera, ask a question about the picture. Get an answer about the picture",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "The question to ask about the picture",
                                    }
                                },
                                "required": ["question"],
                            },
                        },
                    ],
                    "tool_choice": "auto",
                }
            )
            self.connection = conn
            asyncio.create_task(self._sway_consumer())


            async for event in self.connection:
                et = getattr(event, "type", None)

                # interruption
                if et == "input_audio_buffer.speech_started":
                    self.clear_queue()
                    _drain(self.sway_queue)
                    self._base_ts = None
                    self._hops_done = 0
                    self.sway.reset()

                if et in ("response.audio.completed", "response.completed"):
                    self._base_ts = None
                    self._hops_done = 0
                    self.sway.reset()
                    _drain(self.sway_queue)

                # surface transcripts to the UI
                if et == "conversation.item.input_audio_transcription.completed":
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "user", "content": event.transcript})
                    )
                if et == "response.audio_transcript.done":
                    await self.output_queue.put(
                        AdditionalOutputs(
                            {"role": "assistant", "content": event.transcript}
                        )
                    )

                # stream audio to fastrtc
                if et == "response.audio.delta":
                    buf = np.frombuffer(base64.b64decode(event.delta), dtype=np.int16).reshape(1, -1)
                    # 1) to fastrtc playback
                    await self.output_queue.put((self.output_sample_rate, buf))
                    # 2) to sway engine for synchronized motion
                    await self.sway_queue.put((self.output_sample_rate, buf))

                    # await self.output_queue.put(
                    #     (
                    #         self.output_sample_rate,
                    #         np.frombuffer(
                    #             base64.b64decode(event.delta), dtype=np.int16
                    #         ).reshape(1, -1),
                    #     )
                    # )

                if et == "response.started":
                    # hard reset per utterance
                    self._base_ts = None           # <-- was never reset
                    self._hops_done = 0
                    self.sway.reset()              # clear carry/envelope/VAD
                    _drain(self.sway_queue)        # drop any stale chunks not yet consumed
                    # optional: also clear playback queue if you want
                    # _drain(self.output_queue)

                # ---- tool-calling plumbing ----
                # 1) model announces a function call item; capture name + call_id
                if et == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item and getattr(item, "type", "") == "function_call":
                        call_id = getattr(item, "call_id", None)
                        name = getattr(item, "name", None)
                        if call_id and name:
                            self._pending_calls[call_id] = {
                                "name": name,
                                "args_buf": "",
                            }

                # 2) model streams JSON arguments; buffer them by call_id
                if et == "response.function_call_arguments.delta":
                    call_id = getattr(event, "call_id", None)
                    delta = getattr(event, "delta", "")
                    if call_id in self._pending_calls:
                        self._pending_calls[call_id]["args_buf"] += delta

                # 3) when args done, execute Python tool, send function_call_output, then trigger a new response
                if et == "response.function_call_arguments.done":
                    call_id = getattr(event, "call_id", None)
                    info = self._pending_calls.get(call_id)
                    if not info:
                        continue
                    name = info["name"]
                    args_json = info["args_buf"] or "{}"
                    # parse args
                    try:
                        args = json.loads(args_json)
                    except Exception:
                        args = {}

                    # dispatch
                    func = self._tools.get(name)
                    try:
                        result = (
                            await func(args)
                            if func
                            else {"error": f"unknown tool: {name}"}
                        )
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {str(e)}"}

                    # send the tool result back
                    await self.connection.conversation.item.create(
                        item={
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(result),
                        }
                    )
                    # ask the model to continue and speak about the result
                    await self.connection.response.create(
                        response={
                            "instructions": "Use the tool result just returned and answer concisely in speech."
                        }
                    )
                    # cleanup
                    self._pending_calls.pop(call_id, None)

                # log tool errors from server if any
                if et == "error":
                    # optional: surface to chat UI
                    await self.output_queue.put(
                        AdditionalOutputs(
                            {
                                "role": "assistant",
                                "content": f"[error] {event.error.get('message') if hasattr(event, 'error') else ''}",
                            }
                        )
                    )

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.connection:
            return
        _, array = frame
        array = array.squeeze()
        audio_message = base64.b64encode(array.tobytes()).decode("utf-8")
        await self.connection.input_audio_buffer.append(audio=audio_message)

    async def emit(self):
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        if self.connection:
            await self.connection.close()
            self.connection = None


# ---- gradio / fastrtc wiring unchanged ----
def update_chatbot(chatbot: list[dict], response: dict):
    chatbot.append(response)
    return chatbot


chatbot = gr.Chatbot(type="messages")
latest_message = gr.Textbox(type="text", visible=False)
stream = Stream(
    OpenAIHandler(),
    mode="send-receive",
    modality="audio",
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
    additional_outputs_handler=update_chatbot,
)

if __name__ == "__main__":
    Thread(target=stream.ui.launch, kwargs={"server_port": 7860}).start()
    # st = SpeechTapper()
    while True:
        time.sleep(1)
