import argparse
import base64
import logging
import time

import torch
from gst_signalling.utils import find_producer_peer_id_by_name
from transformers import AutoModelForImageTextToText, AutoProcessor

from reachy_mini.gstreamer.gstrecorder import GstRecorder
from reachy_mini.gstreamer.utils import PlayerMode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GStreamer Recorder")
    parser.add_argument(
        "--mode",
        choices=["local", "webrtc"],
        default="local",
        help="Select player mode: local or webrtc",
    )
    parser.add_argument(
        "--peer-id", type=str, default="reachymini", help="Peer ID for WebRTC mode"
    )
    parser.add_argument(
        "--signaling-host",
        type=str,
        default="localhost",
        help="Signaling host for WebRTC",
    )
    parser.add_argument(
        "--signaling-port", type=int, default=8443, help="Signaling port for WebRTC"
    )
    args = parser.parse_args()

    peer_id = ""
    if args.mode == "local":
        mode = PlayerMode.LOCAL
    else:
        mode = PlayerMode.WEBRTC
        peer_id = find_producer_peer_id_by_name(
            args.signaling_host, args.signaling_port, "reachymini"
        )
    logging.basicConfig(level=logging.INFO)
    recorder = GstRecorder(
        mode=mode,
        peer_id=peer_id,
        signaling_host=args.signaling_host,
        signaling_port=args.signaling_port,
    )

    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2"
    ).to("cuda")

    recorder.record()
    # Wait for the pipeline to start and capture a frame
    try:
        while True:
            jpeg_data = recorder.get_video_sample()

            if jpeg_data:
                image_base64 = base64.b64encode(jpeg_data).decode("utf-8")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                            {"type": "text", "text": "Can you describe this image?"},
                        ],
                    },
                ]

                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device, dtype=torch.bfloat16)

                generated_ids = model.generate(
                    **inputs, do_sample=False, max_new_tokens=64
                )
                generated_texts = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

                logging.info(
                    f"Generated text: {generated_texts[0].replace(chr(10), ' ')}"
                )

                # time.sleep(1)
            else:
                logging.warning("No image captured yet. waiting a little bit...")
                time.sleep(1)

            # time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping recorder...")
    recorder.stop()
