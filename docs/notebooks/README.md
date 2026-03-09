# 📚 Reachy Mini Tutorial Notebooks

Welcome to the Reachy Mini tutorial notebooks! These interactive Jupyter notebooks are designed for **progressive learning**, from your first connection to creating interactive behaviors. You'll learn how to use Reachy's SDK and understand its capabilities through practical, hands-on examples.

Each notebook includes:

* ✅ **Executable code** — Works in simulation or on real hardware
* 🎯 **Clear learning goals** — Know what you'll achieve
* 🛠️ **Hands-on exercises** — Practice what you learn
* 💡 **Self-contained explanations** — No need to jump between docs
* ⚠️ **Safety reminders** — Proper usage guidelines
---
## Requirements
To run the notebooks, make sure that you have a python environment with Reachy Mini's SDK and Jupyter installed.
- **Reachy Mini SDK** — Install the SDK by following the [installation guide](https://huggingface.co/docs/reachy_mini/SDK/installation).
- **Jupyter** — A Jupyter environment is required to run the notebooks. Install it with:
```bash
pip install notebook
```

Also, you'll need to have Reachy Mini daemon's up and running by using Reachy Mini Control. Please refer to the [installation guide](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/usage#2-installation).


<details>
<summary><strong>In case <code>ipykernel</code> is asked to be installed</strong></summary>

If you see an error or prompt about installing `ipykernel` when launching a notebook, it means your Python environment is missing the Jupyter kernel package. You can install it with:

```bash
pip install ipykernel
python -m ipykernel install --user --name mini --display-name "Python (mini)"
```

After installation, restart your Jupyter server and try opening the notebook again.

If you use multiple Python environments, make sure Jupyter is running in the same environment as your Reachy Mini SDK.

</details>

## 📘 Available Notebooks

### **Notebook 0 — First Connection and Movement**
**Duration:** ~20 minutes | **Difficulty:** Beginner

Learn the fundamentals of connecting to Reachy Mini and controlling its movements.

**What you'll learn:**
* 🔌 Connecting to Reachy Mini (both connection modes)
* 🤖 Understanding Reachy's parts (head, antennas)
* 🎯 Making your first movements with `goto_target()`
* 📐 Creating head poses and controlling antennas
* ⏱️ Using duration for smooth motion

**Topics covered:** Connection modes, head poses, antennas, `goto_target()`, `set_target()`

---

### **Notebook 1 — Basic Media: Camera & Audio**
**Duration:** ~20 minutes | **Difficulty:** Beginner

Make Reachy see and hear! Learn to capture images, record audio, and play sounds.

**What you'll learn:**
* 📸 Capturing images from the camera
* 🎬 Displaying video frames
* 🎤 Recording audio from the microphone array
* 🔊 Playing sounds through the speaker
* 💾 Saving and loading media files
* 🤖 Combining media with motion for interactive behaviors

**Topics covered:** Camera access, image capture, audio recording/playback, real-time audio processing, media + motion

---

### ❓ Troubleshooting

If you encounter any issues during your exploration of the notebooks, check the **[Troubleshooting & FAQ Guide](https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/troubleshooting.md)**