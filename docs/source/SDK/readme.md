# Build, play, and innovate with Reachy Mini 🤖


## ⚡️ Quick Look
Control your robot in just **a few lines of code**:

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    # Look up and tilt head
    mini.goto_target(
        head=create_head_pose(z=10, roll=15, degrees=True, mm=True),
        duration=1.0
    )
```

## 🚀 Get Started
* **[Installation](installation.md)**: 5 minutes to set up your computer
* **[Quickstart Guide](quickstart.md)**: Run your first behavior on Reachy Mini
* **[Python SDK](python-sdk.md)**: Learn to move, see, speak, and hear.
* **[AI Integrations](integration.md)**: Connect LLMs, build Apps, and publish to Hugging Face.
* **[Core Concepts](core-concept.md)**: Architecture, coordinate systems, and safety limits.

## 📂 Code Examples

We provide a collection of ready-to-run scripts to help you understand how to use specific features of the robot.

[**👉 Browse the Examples Folder**](https://github.com/pollen-robotics/reachy_mini/tree/main/examples)

## 🤖 AI-Assisted Development

Using an AI coding assistant (Claude, GPT, Copilot, etc.)? Check out [**agents.md**](../../agents.md) in the repo root - it helps AI agents understand the SDK and guide you through app development.

## ❓ Troubleshooting

Encountering an issue? 👉 **[Check the Troubleshooting & FAQ Guide](../troubleshooting.md)**

## 💬 Community
* [Discord](https://discord.gg/Y7FgMqHsub) - Get help and share projects.
* [Hugging Face Spaces](https://huggingface.co/spaces?q=reachy_mini) - Discover community apps.
* [GitHub Discussions](https://github.com/pollen-robotics/reachy_mini/discussions) - Feature requests and bugs.
