# Integrations & Apps

Reachy Mini is designed for AI Builders. Here is how to integrate LLMs and share your work.

## Building an App

We provide a CLI tool to generate, check and publish a standard App structure (compatible with Hugging Face Spaces). See the full guide: **[Building & Publishing Apps](apps.md)**.

## JavaScript Web Apps
Want a zero-install, cross-platform app that runs in the browser? Check out the [JavaScript SDK & Web Apps](javascript-sdk) guide — build static Hugging Face Spaces that control your robot over WebRTC from any device, including your phone.

## HTTP & WebSocket API
Building a dashboard or a non-Python controller? The Daemon exposes full control via REST.

* **Docs:** `http://localhost:8000/docs`
* **Get State:** `GET /api/state/full`
* **WebSocket:** `ws://localhost:8000/api/state/ws/full`

## AI Experimentation Tips

* **Conversation Demo:** Check out our reference implementation combining VAD (Voice Activity Detection), LLMs, and TTS: [reachy_mini_conversation_demo](https://github.com/pollen-robotics/reachy_mini_conversation_demo).