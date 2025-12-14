# Integrations & Apps

Reachy Mini is designed for AI Builders. Here is how to integrate LLMs and share your work.

## Building an App
We provide a CLI tool to generate a standard App structure (compatible with Hugging Face Spaces).

1.  **Generate Template:**
    ```bash
    reachy-mini-make-app my_awesome_app
    ```
2.  **Edit `app.py`:** Add your logic (e.g., "If face detected -> Wave").
3.  **Publish:** Upload to a Hugging Face Space with the tag `reachy_mini` in the README YAML metadata.

*See our blog post: [Make and Publish Reachy Mini Apps](https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps)*

## HTTP & WebSocket API
Building a dashboard or a non-Python controller? The Daemon exposes full control via REST.

* **Docs:** `http://localhost:8000/docs`
* **Get State:** `GET /api/state/full`
* **WebSocket:** `ws://localhost:8000/api/state/ws/full`

## AI Experimentation Tips

* **Low Latency Video:** Use the `--backend gstreamer` flag when starting the daemon for faster video pipelines suitable for real-time vision models.
* **Conversation Demo:** Check out our reference implementation combining VAD (Voice Activity Detection), LLMs, and TTS: [reachy_mini_conversation_demo](https://github.com/pollen-robotics/reachy_mini_conversation_demo).