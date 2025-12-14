# Build, play, and innovate with Reachy Mini 🤖


## ⚡️ Quick Look
Control your robot in just **3 lines of code**:

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
* **[Quickstart Guide](quickstart.md)**: Zero to "Hello World" in 5 minutes.
* **[Python SDK](python-sdk.md)**: Learn to move, see, speak, and hear.
* **[AI Integrations](integration.md)**: Connect LLMs, build Apps, and publish to Hugging Face.
* **[Core Concepts](core-concept.md)**: Architecture, coordinate systems, and safety limits.

## 📂 Code Examples

We provide a collection of ready-to-run scripts to help you understand how to use specific features of the robot.

[**👉 Browse the Examples Folder**](../../examples)

## 📦 Installation
Prerequisites: Python 3.10 - 3.13.

```bash
pip install reachy-mini
```

*For simulation users:*
```bash
pip install reachy-mini[mujoco]
```

## 💬 Community
* [Discord](https://discord.gg/pollen-robotics) - Get help and share projects.
* [Hugging Face Spaces](https://huggingface.co/spaces?q=reachy_mini) - Discover community apps.
* [GitHub Discussions](https://github.com/pollen-robotics/reachy_mini/discussions) - Feature requests and bugs.


