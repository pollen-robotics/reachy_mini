# 📡 GStreamer Installation for Wireless Reachy Mini

> This guide will help you install [GStreamer](https://gstreamer.freedesktop.org) to receive video and audio streams from your wireless Reachy Mini.

<div align="center">

| 🐧 **Linux** | 🍎 **macOS** | 🪟 **Windows** |
|:---:|:---:|:---:|
| ✅ Supported | ✅ Supported | ⚠️ Partial Support |

</div>

## 🔧 Install GStreamer

<details>
<summary>🐧 <strong>Linux</strong></summary>

### Step 1: Install GStreamer

**For Ubuntu/Debian-based systems:**

In you terminal, run:

```bash
sudo apt-get update
sudo apt-get install -y \
    libgstreamer-plugins-bad1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libglib2.0-dev \
    libssl-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    libportaudio2 \
    gstreamer1.0-libcamera \
    librpicam-app1 \
    libnice10 \
    gstreamer1.0-plugins-good \
    gstreamer1.0-alsa \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-nice \
    python3-gi \
    python3-gi-cairo
```

### Step 2: Install Rust

On Linux, the WebRTC plugin is not activated by default and needs to be compiled manually from the Rust source code. Install Rust from the commmand line using `rustup`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Step 3: Build and install WebRTC plugin

The build and install the WebRTC plugin, run the following commands :

```bash
# Clone the GStreamer Rust plugins repository
git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git
cd gst-plugins-rs
git checkout 0.14.1

# Install the cargo-c build tool
cargo install cargo-c

# Create installation directory
sudo mkdir -p /opt/gst-plugins-rs
sudo chown $USER /opt/gst-plugins-rs

# Build and install the WebRTC plugin (this may take several minutes)
cargo cinstall -p gst-plugin-webrtc --prefix=/opt/gst-plugins-rs --release

# Add plugin path to your environment
echo 'export GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/x86_64-linux-gnu:$GST_PLUGIN_PATH' >> ~/.bashrc
source ~/.bashrc
```

> **💡 Note:** For ARM64 systems (like Raspberry Pi), replace `x86_64-linux-gnu` with `aarch64-linux-gnu` in the export command.

</details>

<details>
<summary>🍎 <strong>macOS</strong></summary>

### Using Homebrew

```bash
brew install gstreamer libnice-gstreamer
```

The WebRTC plugin is activated by default in the Homebrew package.

</details>

<details>
<summary>🪟 <strong>Windows</strong></summary>

> ⚠️ **Note:** Windows support is currently partial. Some features may not work as expected.

### Step 1: Install GStreamer using the official installer

<div align="center">

[![Download GStreamer for Windows](https://img.shields.io/badge/Download-GStreamer%20for%20Windows-blue?style=for-the-badge&logo=windows&logoColor=white)](https://gstreamer.freedesktop.org/download/)

</div>

1. Download both **runtime** and **development** installers (MSVC version)
2. Install both with **Complete** installation option
3. Add to system PATH: `C:\gstreamer\1.0\msvc_x86_64\bin`
4. Add to PKG_CONFIG_PATH: `C:\gstreamer\1.0\msvc_x86_64\lib\pkgconfig`

> **💡 Important:** Replace `C:\gstreamer` with your actual GStreamer installation folder if you installed it in a different location.

### Step 2: Install Rust

On Windows, the WebRTC plugin is not activated by default and needs to be compiled manually from the Rust source code. Install Rust using the Windows installer:

1. Download and install Rust from [https://rustup.rs/](https://rustup.rs/)
2. Restart your terminal.

### Step 3: Build and install WebRTC plugin

The build and install the WebRTC plugin, run the following commands:

```powershell
# Clone the GStreamer Rust plugins repository
git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git
cd gst-plugins-rs
git checkout 0.14.1

# Install the cargo-c build tool
cargo install cargo-c

# Build the WebRTC plugin (this may take several minutes)
cargo cinstall -p gst-plugin-webrtc --prefix=C:\gst-plugins-rs --release

# Copy the plugin to GStreamer plugins directory
copy C:\gst-plugins-rs\lib\gstreamer-1.0\gstrswebrtc.dll C:\gstreamer\1.0\msvc_x86_64\lib\gstreamer-1.0\

# Add plugin path to environment 
set GST_PLUGIN_PATH="C:\gst-plugins-rs\lib\gstreamer-1.0;%GST_PLUGIN_PATH%"
```

> **💡 Note:** Replace `C:\gstreamer` with your actual GStreamer installation path if different. The last command requires Administrator privileges to set system-wide environment variables.

</details>

## ✅ Verify Installation

Finally, you can test your GStreamer installation as follows:

```bash
# Check version
gst-launch-1.0 --version

# Test basic functionalities
gst-launch-1.0 videotestsrc ! autovideosink

# Verify WebRTC plugin
gst-inspect-1.0 webrtcsrc
```

## 🔧 Python Dependencies

When installing Reachy Mini Python package, you will also need to add the `gstreamer` extra :

### Install from PyPI

```bash
uv add reachy-mini --extra gstreamer
```

### Install from source

```bash
uv sync --extra gstreamer
```
