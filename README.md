# Reachy Mini 🤖

[![Ask on HuggingChat](https://img.shields.io/badge/Read_the-Documentation-yellow?logo=huggingface&logoColor=yellow)](https://huggingface.co/docs/reachy_mini/)
[![Discord](https://img.shields.io/badge/Discord-Join_the_Community-7289DA?logo=discord&logoColor=white)](https://discord.gg/Y7FgMqHsub)

**Reachy Mini is an open-source, expressive robot made for hackers and AI builders.**

🛒 [**Buy Reachy Mini**](https://www.hf.co/reachy-mini/)

[![Reachy Mini Hello](/docs/assets/reachy_mini_hello.gif)](https://www.pollen-robotics.com/reachy-mini/)

## ⚡️ Build and start your own robot

**Choose your platform to access the specific guide:**

| **🤖 Reachy Mini (Wireless)** | **🔌 Reachy Mini Lite** | **💻 Simulation** |
| :---: | :---: | :---: |
| The full autonomous experience.<br>Raspberry Pi 4 + Battery + WiFi. | The developer version.<br>USB connection to your computer. | No hardware required.<br>Prototype in MuJoCo. |
| 👉 [**Go to Wireless Guide**](docs/platforms/reachy_mini/get_started.md) | 👉 [**Go to Lite Guide**](docs/platforms/reachy_mini_lite/get_started.md) | 👉 [**Go to Simulation**](docs/platforms/simulation/get_started.md) |



> ⚡ **Pro tip:** Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for 10-100x faster app installations (auto-detected, falls back to `pip`).

<br>

## 📱 Apps & Ecosystem

Reachy Mini comes with an app store powered by Hugging Face Spaces. You can install these apps directly from your robot's dashboard with one click!

* **🗣️ [Conversation App](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app):** Talk naturally with Reachy Mini (powered by LLMs).
* **📻 [Radio](https://huggingface.co/spaces/pollen-robotics/reachy_mini_radio):** Listen to the radio with Reachy Mini!
* **👋 [Hand Tracker](https://huggingface.co/spaces/pollen-robotics/hand_tracker_v2):** The robot follows your hand movements in real-time.

👉 [**Browse all apps on Hugging Face**](https://hf.co/reachy-mini/#/apps)

<br>

## 🚀 Getting Started with Reachy Mini SDK

### User guides
* **[Installation](docs/SDK/installation.md)**: 5 minutes to set up your computer
* **[Quickstart Guide](docs/SDK/quickstart.md)**: Run your first behavior on Reachy Mini
* **[Python SDK](docs/SDK/python-sdk.md)**: Learn to move, see, speak, and hear.
* **[AI Integrations](docs/SDK/integration.md)**: Connect LLMs, build Apps, and publish to Hugging Face.
* **[Core Concepts](docs/SDK/core-concept.md)**: Architecture, coordinate systems, and safety limits.
* 🤗[**Share your app with the community**](https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps)
* 📂 [**Browse the Examples Folder**](examples)

### Quick Look
After [installing the SDK](docs/SDK/installation.md), once your robot is awake, you can control it in just **a few lines of code**:

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

<br>

## 🛠 Hardware Overview

Reachy Mini robots are sold as kits and generally take **2 to 3 hours** to assemble. Detailed step-by-step guides are available in the platform-specific folders linked above.

* **Reachy Mini (Wireless):** Runs onboard (RPi 4), autonomous, includes IMU. [See specs](docs/platforms/reachy_mini/hardware.md).
* **Reachy Mini Lite:** Runs on your PC, powered via wall outlet. [See specs](docs/platforms/reachy_mini_lite/hardware.md).

<br>

## ❓ Troubleshooting

Encountering an issue? 👉 **[Check the Troubleshooting & FAQ Guide](/docs/troubleshooting.md)**

<br>

## 🤝 Community & Contributing

* **Join the Community:** Join [Discord](https://discord.gg/2bAhWfXme9) to share your moments with Reachy, build apps together, and get help.
* **Found a bug?** Open an issue on this repository.


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
Hardware design files are licensed under Creative Commons BY-SA-NC.
