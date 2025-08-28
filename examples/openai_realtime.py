"""Show cases fastrtc and OpenAI realtime.

Requirements:
pip install openai[realtime]
set your OPENAI_API_KEY
"""

import argparse
import asyncio
import base64
import logging

import numpy as np
import openai
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from gst_signalling.utils import async_find_producer_peer_id_by_name

from reachy_mini.gstreamer.gstplayer import GstPlayer
from reachy_mini.gstreamer.gstrecorder import GstRecorder
from reachy_mini.gstreamer.utils import PlayerMode

SAMPLE_RATE = 24000

emit_logger = logging.getLogger("emit")
receive_logger = logging.getLogger("receive")
main_logger = logging.getLogger("main")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(process)d %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class OpenAIHandler(AsyncStreamHandler):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=SAMPLE_RATE,
            input_sample_rate=SAMPLE_RATE,
        )
        self.connection = None
        self.output_queue = asyncio.Queue()

    def copy(self):
        return OpenAIHandler()

    async def start_up(
        self,
    ):
        """Connect to realtime API. Run forever in separate thread to keep connection open."""
        self.client = openai.AsyncOpenAI()
        async with self.client.beta.realtime.connect(
            model="gpt-4o-mini-realtime-preview-2024-12-17"
        ) as conn:
            await conn.session.update(
                session={
                    "turn_detection": {"type": "server_vad"},
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "language": "en",
                    },
                }
            )
            self.connection = conn
            async for event in self.connection:
                # Handle interruptions
                if event.type == "input_audio_buffer.speech_started":
                    # self.clear_queue()
                    pass
                if (
                    event.type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "user", "content": event.transcript})
                    )
                if event.type == "response.audio_transcript.done":
                    await self.output_queue.put(
                        AdditionalOutputs(
                            {"role": "assistant", "content": event.transcript}
                        )
                    )
                if event.type == "response.audio.delta":
                    await self.output_queue.put(
                        (
                            self.output_sample_rate,
                            np.frombuffer(
                                base64.b64decode(event.delta), dtype=np.int16
                            ).reshape(1, -1),
                        ),
                    )

    async def receive(self, frame: bytes) -> None:
        if not self.connection:
            return
        audio_message = base64.b64encode(frame).decode("utf-8")
        await self.connection.input_audio_buffer.append(audio=audio_message)  # type: ignore

    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        if self.connection:
            await self.connection.close()
            self.connection = None


async def receive_loop(recorder: GstRecorder, openai: OpenAIHandler) -> None:
    receive_logger.info("Starting receive loop")
    while True:
        data = recorder.get_audio_sample()
        if data is not None:
            await openai.receive(data)
        await asyncio.sleep(0)  # Prevent busy waiting


async def emit_loop(player: GstPlayer, openai: OpenAIHandler) -> None:
    emit_logger.info("Starting emit loop")
    while True:
        data = await openai.emit()
        if isinstance(data, AdditionalOutputs):
            for msg in data.args:
                emit_logger.info(f"role : {msg['role']}: content: {msg['content']}")

        elif isinstance(data, tuple):
            _, frame = data
            player.push_sample(frame.tobytes())
        else:
            pass
        await asyncio.sleep(0)  # Prevent busy waiting


async def main(mode: PlayerMode, signaling_host: str, signaling_port: int):
    main_logger.info(f"Starting in {mode} mode")

    peer_id = ""
    if mode == PlayerMode.WEBRTC:
        main_logger.info("Running in WebRTC mode")
        main_logger.info("Searching for reachymini peer id")
        peer_id = await async_find_producer_peer_id_by_name(
            signaling_host, signaling_port, "reachymini"
        )

        if peer_id == "":
            main_logger.error("No peer id found")
            return
        else:
            main_logger.info(f"found peer id: {peer_id}")

    openai = OpenAIHandler()

    recorder = GstRecorder(
        mode=mode,
        signaling_host=signaling_host,
        signaling_port=signaling_port,
        peer_id=peer_id,
    )
    recorder.record()

    player = GstPlayer(
        mode=mode, signaling_host=signaling_host, signaling_port=signaling_port
    )
    player.play()

    main_logger.info("Starting main loop. You can start to speak")
    start_up_task = asyncio.create_task(openai.start_up())
    emit_task = asyncio.create_task(emit_loop(player, openai))
    receive_task = asyncio.create_task(receive_loop(recorder, openai))
    await asyncio.gather(start_up_task, emit_task, receive_task)
    await openai.shutdown()
    recorder.stop()
    player.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Realtime + fastrtc demo")
    parser.add_argument(
        "--mode",
        choices=["webrtc", "local"],
        default="webrtc",
        help="Run in 'webrtc' (default) or 'local' mode.",
    )
    parser.add_argument(
        "--signaling-host",
        type=str,
        default="10.0.1.38",
        help="Signaling host for WebRTC mode.",
    )
    parser.add_argument(
        "--signaling-port",
        type=int,
        default=8443,
        help="Signaling port for WebRTC mode.",
    )

    args = parser.parse_args()

    mode = PlayerMode.WEBRTC if args.mode == "webrtc" else PlayerMode.LOCAL

    try:
        asyncio.run(main(mode, args.signaling_host, args.signaling_port))
    except KeyboardInterrupt:
        main_logger.info("Process interrupted")
