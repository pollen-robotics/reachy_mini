# 📦 Installation of the Daemon and Python SDK

**Supported OS:** We support and test on **Linux**, **MacOS** and **Windows**, but feel free to open an [issue](https://github.com/pollen-robotics/reachy_mini/issues) if you encounter any problem.

## 1. Prerequisites

* **Python:** You need Python installed on your computer (versions from _3.10_ to _3.12_ are supported).
* **Git** and **Git LFS:** You must have `git` and `git-lfs` installed to correctly download model assets.

### Install Python

#### 1. Install uv

Open the *Terminal* application and run the following command (copy-paste it and press enter):

**Linux and MacOs**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Follow the prompts of the program and let it handle the installation. Once completed, you can verify the installation with :

```bash
uv --version
```

#### 2. Install Python

In the *Terminal* application, run the following command :

```bash
uv python install 3.XX
```

We suggest replacing `3.XX` by `3.12` to install Python 3.12 which is the latest supported Python version on Reachy Mini.


### Install Git and Git LFS

**Linux**

In the _Terminal_ application, run :

```bash
sudo apt install git git-lfs
```

And initialize Git LFS by running :

```bash
git lfs install
```

**MacOS**

1. Install Homebrew

In the _Terminal_ application, run :

```zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the prompts of the program and let it handle the installation. If you have an Apple Silicon processor (M1, M2,...), Homebrew will prompt you to run this command :

```zsh
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

You can finally verify the installation with :

```zsh
brew --version
```

2. Install Git and Git LFS

In the _Terminal_ application, run :

```zsh
brew install git git-lfs
```

And initialize Git LFS by running :

```zsh
git lfs install
```

**Windows**

The easiest way to install Git and Git LFS on Windows is to install [Git for Windows](https://gitforwindows.org).

Once the installation is completed, initialize Git LFS by running the following command in the _Terminal_ application :

```Powershell
git lfs install
```

## 2. Set up a Virtual Environment (Highly Recommended)

Modern version of Python require a _Virtual Environment_ to work correctly. This is also a good development practice, as it isolates the Reachy Mini installation and prevents dependency conflicts with your other Python projects.

### Create and activate the environment

To create a virtual environment using, open the _Terminal_ application  and run :

```bash
uv venv my_environment --python 3.XX
```

Where `3.XX` is the installed Pyhton version.

The virtual environment is then activated by running this command :

**Linux and MacOS**
```bash
source .venv/bin/activate
```

**Windows**
```Powershell
.venv\Scripts\activate
```

> *Once activated, you should see `(.venv)` appear at the start of your command line.*

## 3. Install the Package

You can install Reachy Mini from PyPI (standard) or from the source code (for development).

### Option A: Install from PyPI (Standard)
Best for most users who just want to use the robot.

```bash
uv add reachy-mini
```

### Option B: Install from Source (For Developers)

Best if you want to modify the SDK code.

```bash
git clone https://github.com/pollen-robotics/reachy_mini
uv sync ./reachy_mini
```

### 🐧 Linux Users: USB Permission Setup
If you are on Linux and using the robot via USB, you must set up udev rules.
<details>
<summary>Click to see udev instructions</summary>

```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="tty", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"' \
| sudo tee /etc/udev/rules.d/99-reachy-mini.rules

sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout $USER
```
⚠️ Important: You may need to log out and log back in for the group changes to take effect.

</details>

## ❓ Troubleshooting
Encountering an issue? 👉 **[Check the Troubleshooting & FAQ Guide](/docs/troubleshooting.md)**

## Next Steps
* **[Quickstart Guide](quickstart.md)**: Run your first behavior on Reachy Mini
* **[Python SDK](python-sdk.md)**: Learn to move, see, speak, and hear.
* **[AI Integrations](integration.md)**: Connect LLMs, build Apps, and publish to Hugging Face.
* **[Core Concepts](core-concept.md)**: Architecture, coordinate systems, and safety limits.

