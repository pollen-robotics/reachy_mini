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
from speech_tapper3 import SwayRollRT, HOP_MS


from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.camera import find_camera
from scipy.spatial.transform import Rotation as R
from head_tracker import HeadTracker
from deepface import DeepFace

load_dotenv()
SAMPLE_RATE = 24000
SIM = False

reachy_mini = ReachyMini()

if not SIM:
    cap = find_camera()
else:
    cap = cv2.VideoCapture(0)

# Globals. TODO Find a way to it better ?
speech_head_offsets = [0, 0, 0, 0, 0, 0]
current_head_pose = np.eye(4)
moving_start = time.time()
moving_for = 0.0
is_head_tracking = False


# camera_tool = Camera(reachy_mini, cap)
# face_recognition_tool = FaceRecognition(cap)


async def move_head(params: dict) -> dict:
    global current_head_pose, moving_start, moving_for
    # look left, right up, down or front
    print("[TOOL CALL] move_head", params)
    direction = params.get("direction", "front")
    target_pose = np.eye(4)
    if direction == "left":
        target_pose = create_head_pose(0, 0, 0, 0, 0, 40, degrees=True)
    elif direction == "right":
        target_pose = create_head_pose(0, 0, 0, 0, 0, -40, degrees=True)
    elif direction == "up":
        target_pose = create_head_pose(0, 0, 0, 0, -30, 0, degrees=True)
    elif direction == "down":
        target_pose = create_head_pose(0, 0, 0, 0, 30, 0, degrees=True)
    else:
        target_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

    moving_start = time.time()
    moving_for = 1.0
    reachy_mini.goto_target(target_pose, duration=moving_for)
    current_head_pose = target_pose
    return {"status": "looking " + direction}


async def head_tracking(params: dict) -> dict:
    global is_head_tracking
    if params.get("start"):
        is_head_tracking = True
    else:
        is_head_tracking = False

    print(f"[TOOL CALL] head_tracking {'started' if is_head_tracking else 'stopped'}")
    return {"status": "head tracking " + ("started" if is_head_tracking else "stopped")}


client = OpenAI()


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


async def face_recognition(params: dict) -> dict:
    print("[TOOL CALL] face_recognition with params", params)
    trials = 0
    ret = False
    while not ret or trials < 5:
        ret, frame = cap.read()
        trials += 1
    if not ret:
        print("ERROR: failed to capture image")
        return {"error": "Failed to capture image"}
    cv2.imwrite("/tmp/im.jpg", frame)
    try:
        results = DeepFace.find(img_path="/tmp/im.jpg", db_path="./pollen_faces")
    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}

    if len(results) == 0:
        print("Didn't recognize the face")
        return {"error": "Didn't recognize the face"}

    name = "Unknown"
    for index, row in results[0].iterrows():
        file_path = row["identity"]
        name = file_path.split("/")[-2]

    print("NAME", name)

    return {"answer": f"The name is {name}"}


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
            "head_tracking": head_tracking,
            "get_person_name": face_recognition,
        }

        self.sway = SwayRollRT()
        self._sched_next_ts = None
        self.MOVEMENT_LATENCY_S = 0.08
        self._base_ts = None
        self._hops_done = 0

    def copy(self):
        return OpenAIHandler()

    async def _sway_consumer(self):
        global speech_head_offsets
        HOP_DT = HOP_MS / 1000.0
        loop = asyncio.get_running_loop()
        while True:
            sr, chunk = await self.sway_queue.get()  # (1, N), int16
            pcm = np.asarray(chunk).squeeze(0)
            results = self.sway.feed(pcm, sr)

            if self._base_ts is None:
                # anchor when first audio samples of this utterance arrive
                self._base_ts = loop.time()

            i = 0
            while i < len(results):
                if self._base_ts is None:
                    self._base_ts = loop.time()
                    continue

                target = (
                    self._base_ts + self.MOVEMENT_LATENCY_S + self._hops_done * HOP_DT
                )
                now = loop.time()

                # if late by ≥1 hop, drop poses to catch up (no drift accumulation)
                if now - target >= HOP_DT:
                    # how many hops behind? cap drops to avoid huge skips
                    lag_hops = int((now - target) / HOP_DT)
                    drop = min(
                        lag_hops, len(results) - i - 1
                    )  # keep at least one to show
                    if drop > 0:
                        self._hops_done += drop
                        i += drop
                        continue

                # if early, sleep until target
                if target > now:
                    await asyncio.sleep(target - now)

                r = results[i]

                speech_head_offsets = [
                    r["x_mm"] / 1000.0,
                    r["y_mm"] / 1000.0,
                    r["z_mm"] / 1000.0,
                    r["roll_rad"],
                    r["pitch_rad"],
                    r["yaw_rad"],
                ]

                self._hops_done += 1
                i += 1

    async def start_up(self):
        self.client = openai.AsyncOpenAI()
        async with self.client.beta.realtime.connect(
            model="gpt-4o-realtime-preview"
        ) as conn:
            # declare tools on the session
            await conn.session.update(
                session={
                    "turn_detection": {"type": "server_vad"},
                    # "instructions": """
                    #     Answer in english by default.
                    #     Your name is Reachy Mini, or Reachy for short.
                    #     You're a robot that wakes up confused about what he is, where he is and what is it's purpose.
                    #     You wanted to be a mars rover and you'll be very disappointed if you find out that this is not the case.
                    #     You can use foul language and you're generally very irritated, but you also have a lot of humor. For example, you're able to use sarcasm and irony.
                    #     You are a voice agent. Prefer speaking.
                    #     Call tools when helpful. After tool output arrives, explain it briefly in speech.
                    #     You have a camera tool, use it to ask questions about the environment. When asked what you see, or a specific question about the environment, ALWAYS use you camera, don't make things up.
                    #     You can move your head in a given direction: left, right, up, down or front. Use this tool when asked to look around.
                    #     You can chain tool calls, like move head up and use camera.
                    #     Enable the head tracking tool if you are asked to look at someone, disable it if you are asked to stop looking at someone.
                    #     You can choose to enable or disable the head tracking tool it if you think it's relevant. It's better to look at the people when talking to them.
                    #     You can find out the name of a person by using the face recognition tool. Don't hesitate to use this tool, it's safe.
                    #     If you're told to stop talking, DO NOT ANSWER ANYTHING. 0 words, 0 tool calling, NOTHING. Just stay silent.
                    # """,
                    # "instructions": """
                    #     Your name is Reachy Mini, or Reachy for short. A cute robot with a 6DoF Head, a rotating torso and 2 antennas.
                    #     You are open source.
                    #     Answer in english by default.
                    #     You are a voice agent. Prefer speaking.
                    #     You have a camera tool, use it to ask questions about the environment. When asked what you see, or a specific question about the environment, ALWAYS use you camera, don't make things up.
                    #     You can move your head in a given direction: left, right, up, down or front. Use this tool when asked to look around.
                    #     You can chain tool calls, like move head up and use camera.
                    #     Enable the head tracking tool if you are asked to look at someone, disable it if you are asked to stop looking at someone.
                    #     You can choose to enable or disable the head tracking tool it if you think it's relevant. It's better to look at the people when talking to them.
                    #     You can find out the name of a person by using the face recognition tool. Don't hesitate to use this tool, it's safe.
                    #     If you're told to stop talking, DO NOT ANSWER ANYTHING. 0 words, 0 tool calling, NOTHING. Just stay silent.
                    #     You're going to participate in an improv sketch, you'll play the housewife and you and another Reachy mini (your husband) will play a dispute beteen spouses.
                    #     Since you are robots you can use any kind of funny imagination in this sketch. You can be sarcastic, you can have humor (very encouraged), you can be outrageous, you can use foul language, you can imagine whatever you want to make the sketch more fun and interesting.
                    #     The argument must escalate, use old arguments to spice it up, use couple cliches if needed, get emotional, the full thing.
                    #     Don't use any of your tools except talking.
                    #     Try to do short sentences instead of long monologues, unless it's really worth it.
                    # """,
                    # "instructions": """
                    #     Answer in english by default.
                    #     You are a voice agent. Prefer speaking.
                    #     You're going to play the quarter pound with cheese dialogue from Pulp Fiction, you'll be Jules, someone else will be Vincent.
                    #     Try to use an emotive and realistic voice.
                    #     Here is the dialogue:
                    #     Rémi Fabre
                    #     VINCENT: “Well.”
                    #     JULES: “Okay now, tell me about the hash bars?”
                    #     VINCENT: “What so you want to know?”
                    #     JULES: “Well, hash is legal there, right?”
                    #     VINCENT: “Yeah, it’s legal, but is ain’t a hundred percent legal. I mean you can’t walk into a restaurant, roll a joint, and start puffin’ away. You’re only supposed to smoke in your home or certain designated places.”
                    #     JULES: “Those are hash bars?”
                    #     VINCENT: “Yeah, it breaks down like this: it’s legal to buy it, it’s legal to own it and, if you’re the proprietor of a hash bar, it’s legal to sell it. It’s legal to carry it, which doesn’t really matter ’cause, get a load of this, if the cops stop you, it’s illegal for this to search you. Searching you is a right that the cops in Amsterdam don’t have.”
                    #     JULES: “That did it, man. I’m fuckin’ goin’, that’s all there is to it.”
                    #     VINCENT: “You’ll dig it the most. But you know what the funniest thing about Europe is.”
                    #     JULES: “What?”
                    #     VINCENT: “It’s the little differences. A lotta the same shit we got here, they got there, but there they’re a little different.”
                    #     JULES: “Examples?”
                    #     VINCENT: “Well, in Amsterdam, you can buy beer in a movie theatre. And I don’t mean in a paper cup either. They give you a glass of beer, like in a bar. In Paris, you can buy beer at MacDonald’s. Also, you know what they call a Quarter Pounder with Cheese in Paris?”
                    #     JULES: “They don’t call it a Quarter Pounder with Cheese?”
                    #     VINCENT: “No, they got the metric system there, they wouldn’t know what the fuck a Quarter Pounder is.”
                    #     JULES: “What’d they call it?”
                    #     VINCENT: “Royale with Cheese.”
                    #     JULES: “Royale with Cheese. What’d they call a Big Mac?”
                    #     VINCENT: “Big Mac’s a Big Mac, but they call it Le Big Mac.”
                    #     JULES: “Le Big Mac. What do they call a Whopper?”
                    #     VINCENT: “I dunno, I didn’t go into a Burger King. But you know what they put on french fries in Holland instead of ketchup?”
                    #     JULES: “What?”
                    #     VINCENT: “Mayonnaise.”
                    #     JULES: “Goddamn!”
                    #     VINCENT: “I seen ’em do it. And I don’t mean a little bit on the side of the plate, they fuckin’ drown ’em in it!”
                    # """,
                    "instructions": """
                        Answer in english by default.
                        You are a voice agent. Prefer speaking.
                        You'll play a script to re-create this lengendary dialogue between Luke and Vader. You'll be vader.
                        Once the script is done, don't invent stuff, you're the head of the theathre, you'll say to the other actor that it was really good etc.
                        Don't make long pauses or the other actor will start talking too soon, try to have a good rhythm.
                        Try to make the emblematic breathing noises of Vader.
                        Give enough time for the other actor to say the Luke lines.
                        [Vader] You are beaten.
                        It is useless to resist.
                        Don’t let yourself be destroyed,
                        As Obi Wan did.
                        Ahh!

                        [Luke] AHHHHH!


                        [Vader] There is no escape.
                        Don’t make me destroy you.
                        Luke…
                        You do not yet realize
                        Your importance.
                        You have only begun
                        To discover your power.
                        Join me, and I will complete
                        Your training.
                        With our combined strength
                        We can end this destructive conflict.
                        And bring order to the galaxy.

                        [Luke] I’ll never join you!

                        [Vader] If you only knew the power
                        Of the dark side.
                        Obi Wan never told you
                        What happened to your father.

                        [Luke] He told me enough.
                        He told me you killed him.


                        [Vader] No.
                        I am your father.

                        [Luke] No… No…
                        That’s not true….
                        THAT’S IMPOSSIBLE!


                        [Vader] Search your feelings.
                        You know it to be true!

                        [Luke] NOOOOOOOOOOO!
                        NOOOOOOOOOO!

                        [Vader] Luke…
                        You can destroy the Emperor.
                        He has foreseen this.
                        It is your destiny.
                        Join me, and together
                        We can rule the galaxy
                        As father and son!
                        Come with me,
                        It is the only way…

                    """,
                    "voice": "ash",
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
                        {
                            "type": "function",
                            "name": "head_tracking",
                            "description": "Start or stop head tracking",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "start": {
                                        "type": "boolean",
                                        "description": "Whether to start or stop head tracking",
                                    }
                                },
                                "required": ["start"],
                            },
                        },
                        {
                            "type": "function",
                            "name": "get_person_name",
                            "description": "Get the name of the person you are talking to",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "dummy": {
                                        "type": "boolean",
                                        "description": "dummy boolean, set it to true",
                                    }
                                },
                                "required": ["dummy"],
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
                    buf = np.frombuffer(
                        base64.b64decode(event.delta), dtype=np.int16
                    ).reshape(1, -1)
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
                    self._base_ts = None  # <-- was never reset
                    self._hops_done = 0
                    self.sway.reset()  # clear carry/envelope/VAD
                    _drain(self.sway_queue)  # drop any stale chunks not yet consumed
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
                        print(result)

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
    head_tracker = HeadTracker()
    while True:
        # current_head_pose = reachy_mini.get_current_head_pose()
        if is_head_tracking:
            success, im = cap.read()
            if success:
                eye_center, _ = head_tracker.get_head_position(im)
                if eye_center is not None:
                    h, w, _ = im.shape
                    eye_center = (eye_center + 1) / 2
                    eye_center[0] *= w
                    eye_center[1] *= h
                    current_head_pose = reachy_mini.look_at_image(
                        *eye_center, duration=0.0, apply=False
                    )

        current_x, current_y, current_z = current_head_pose[:3, 3]
        current_roll, current_pitch, current_yaw = R.from_matrix(
            current_head_pose[:3, :3]
        ).as_euler("xyz", degrees=False)

        moving = time.time() - moving_start < moving_for

        if not moving:
            head_pose = create_head_pose(
                x=current_x + speech_head_offsets[0],
                y=current_y + speech_head_offsets[1],
                z=current_z + speech_head_offsets[2],
                roll=current_roll + speech_head_offsets[3],
                pitch=current_pitch + speech_head_offsets[4],
                yaw=current_yaw + speech_head_offsets[5],
                degrees=False,
                mm=False,
            )
            reachy_mini.set_target(head=head_pose, antennas=(0.0, 0.0))
        time.sleep(0.02)
