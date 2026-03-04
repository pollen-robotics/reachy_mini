# 📦 Installation Guide

> **Welcome to Reachy Mini!** This guide will help you install the Python SDK and daemon to start controlling your robot.

<div align="center">

| 🐧 **Linux** | 🍎 **macOS** | 🪟 **Windows** |
|:---:|:---:|:---:|
| ✅ Supported | ✅ Supported | ✅ Supported |

</div>

**Need help?** Feel free to open an [issue](https://github.com/pollen-robotics/reachy_mini/issues) if you encounter any problem.

## First time using the command line? 🖥️
<details>
<summary>Click here if you're new to using a terminal/command line</summary>

A **command line** (also called terminal or command prompt) is a text-based interface where you can type commands to interact with your computer. Don't worry—it's simpler than it looks!

**How to open the command line:**
* **Windows:** Press `Win + R`, type `cmd` or `powershell`, and press Enter
* **macOS:** Press `Cmd + Space`, type `Terminal`, and press Enter  
* **Linux:** Press `Ctrl + Alt + T` or search for "Terminal" in your applications

**Basic tips:**
* Type commands exactly as shown in the instructions
* Press `Enter` after typing each command to run it
* You can copy and paste commands (right-click to paste in most command line interfaces)

> [!TIP]
> _Don't be intimidated!_ The command line is just another way to give instructions to your computer. Follow the commands step by step, and you'll be controlling your Reachy Mini in no time!

</details>

## 1. 📋 Prerequisites

<div align="center">

| Tool | Version | Purpose |
|------|---------|---------|
| 🐍 **Python** | 3.10 - 3.12 | Run Reachy Mini SDK |
| 📂 **Git** | Latest | Download source code and apps |
| 📦 **Git LFS** | Latest | Download model assets |

</div>

### 🐍 Install Python

We'll use `uv` - a fast Python package manager that makes installation simple!

#### Step 1: Install uv

<hfoptions id="install-uv">
<hfoption id="Linux / macOS">

In your terminal, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</hfoption>
<hfoption id="Windows">

In your terminal, run:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

</hfoption>
</hfoptions>

**✅ Verify installation:**

Once the installation is completed, close your terminal and open a new one. You can check if everything went well with :
```bash
uv --version
```

#### Step 2: Install Python

In your terminal, run:
```bash
uv python install 3.12 --default
```

> [!TIP]
> We recommend Python 3.12 as it's the latest supported version for Reachy Mini.


### 📂 Install Git and Git LFS

<hfoptions id="install-git">
<hfoption id="Linux">

#### Install Git and Git LFS

In your terminal, run:
```bash
sudo apt install git git-lfs
```

</hfoption>
<hfoption id="macOS">

#### 1. Install Homebrew (if not already installed)

In your terminal, run:
```zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

For Apple Silicon (M1, M2, etc.), you will also be prompted to run:

```zsh
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

✅ Verify Homebrew:

Once the installation is completed you can check if it went fine with:
```zsh
brew --version
```

#### 2. Install Git and Git LFS 

In your terminal, run:
```zsh
brew install git git-lfs
```

</hfoption>
<hfoption id="Windows">

#### Download and install Git for Windows

Download and install Git for Windows:  
https://git-scm.com/install/windows

</hfoption>
</hfoptions>

**✅ Finalize installation:**

Finally, Git LFS then needs to be initialized with the command :

```bash
git lfs install
```

## 2. 🏠 Set up a Virtual Environment

> **Why use a virtual environment?** It keeps your Reachy Mini installation isolated and prevents conflicts with other Python projects. Modern Python development requires this!

### Create the environment

In your terminal, run:
```bash
uv venv reachy_mini_env --python 3.12
```

### Activate the environment

<hfoptions id="activate-venv">
<hfoption id="Linux / macOS">

In your terminal, run:
```bash
source reachy_mini_env/bin/activate
```

</hfoption>
<hfoption id="Windows">

> [!WARNING]
> _First-time setup:_ Before you can activate your virtual environment, Windows needs permission to run scripts. You only need to do this once!

**Step 1:** Open terminal as Administrator
- Press `Win + R`, type `powershell`
- Right-click on "Windows PowerShell" and select "Run as administrator"

**Step 2:** Enable script execution

In the administrator terminal, run:
```powershell
powershell Set-ExecutionPolicy RemoteSigned
```

**Step 3:** Close the administrator terminal and open a regular terminal

Now you can activate your virtual environment by running:
```powershell
reachy_mini_env\Scripts\activate
```

</hfoption>
</hfoptions>

> **✅ Success indicator:** You should see `(reachy_mini_env)` at the start of your command line prompt!

## 3. 🚀 Install Reachy Mini

Choose your installation method:

<div align="center">

| 📦 **PyPI Installation** | 🔧 **Source Installation** |
|:---:|:---:|
| **For Everyone** | **For Developers** |
| Ready to use | Modify the code |

</div>

### 📦 Option A: Install from PyPI
> **Recommended for most users** - Just want to control your robot? This is for you!

In your terminal, run:
```bash
uv pip install "reachy-mini"
```

If you want to use the simulation mode, you need to add the `mujoco` extra:
```bash
uv pip install "reachy-mini[mujoco]"
```

> [!TIP]
> The post installation of gstreamer is due to an [issue](https://github.com/pypi/support/issues/8847#issuecomment-3899714506) with PyPi and should be solved in the future.


### 🔧 Option B: Install from Source  
> **For developers** - Want to modify the SDK or contribute? Choose this option!

In your terminal, run:
```bash
git clone https://github.com/pollen-robotics/reachy_mini && cd reachy_mini
uv sync
```

If you want to use the simulation mode, you need to add the `mujoco` extra:
```bash
uv sync --extra mujoco
```
### 🐧 Linux Users

> **Linux + USB connection?** You need to grant access to Reachy Mini's serial port.

<details>
<summary>🔧 <strong>Click here to set up USB permissions</strong></summary>

Run these commands in your terminal:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="usb", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"' \
| sudo tee /etc/udev/rules.d/99-reachy-mini.rules

sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout $USER
```

> [!WARNING]
> Log out and log back in for the changes to take effect!

</details>

<br />

> **PortAudio** Make sure that portaudio is installed on your system to enable audio features with the default backend.

<details>
<summary>🔧 <strong>Installing PortAudio</strong></summary>

Run this command in your terminal:

```bash
sudo apt-get install libportaudio2
```

</details>

#### Gstreamer

Media management is performed by the GStreamer library. It is installed with all the dependencies for Mac and Windows users.
Linux users have extra steps to follow:

<div align="center">

[![GStreamer Installation Guide](https://img.shields.io/badge/📖-GStreamer%20Installation%20Guide-blue?style=for-the-badge)](gstreamer-installation)

</div>

## 🎉 Congratulations!

You've successfully installed Reachy Mini! Your robot is ready to come to life.

## ❓ Troubleshooting
Encountering an issue? 👉 **[Check the Troubleshooting & FAQ Guide](../troubleshooting.md)**

## Next Steps
* **[Quickstart Guide](quickstart.md)**: Run your first behavior on Reachy Mini
* **[Python SDK](python-sdk.md)**: Learn to move, see, speak, and hear.
* **[AI Integrations](integration.md)**: Connect LLMs, build Apps, and publish to Hugging Face.
* **[Core Concepts](core-concept.md)**: Architecture, coordinate systems, and safety limits.
