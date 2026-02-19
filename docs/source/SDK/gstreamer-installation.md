# 📡 GStreamer Installation

> This guide will help you install [GStreamer](https://gstreamer.freedesktop.org) for receiving video and audio streams from your Reachy Mini. See the [media architecture](/docs/SDK/media-architecture.md) to understand how streams are accessed locally or remotely depending on the configuration.

Python wheels are available for the Windows and macOS platforms and are included in the project dependencies. Everything should work out of the box. For Linux users, a manual installation is required.

<div align="center">

| 🐧 **Linux** | 🍎 **macOS** | 🪟 **Windows** |
|:---:|:---:|:---:|
| ⚠️ Manual installation | ✅ Python wheels | ✅ Python wheels |

</div>

## 🔧 Install GStreamer

<hfoptions id="gstreamer-install">
<hfoption id="Linux">

### Step 1: Install GStreamer

**For Ubuntu/Debian-based systems:**

In your terminal, run:

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
    libnice10 \
    gstreamer1.0-plugins-good \
    gstreamer1.0-alsa \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-nice \
    python3-gi \
    python3-gi-cairo
```

**For Ubuntu 22.04 only:** The default GStreamer version is too old. Gstreamer >=1.22 is required. You need to add a PPA to get GStreamer 1.24.x:

```bash
sudo add-apt-repository ppa:savoury1/multimedia
sudo apt update
sudo apt install \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev
```

Verify you have the correct version:
```bash
pkg-config --modversion gstreamer-1.0
# Should output 1.24.x or higher
```

### Step 2: Install Rust

On Linux, the WebRTC plugin is not enabled by default and needs to be compiled manually from the Rust source code. Install Rust from the command line using `rustup`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Step 3: Build and install WebRTC plugin

To build and install the WebRTC plugin, run the following commands:

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


</hfoption>
<hfoption id="macOS">

It is not necessary to install GStreamer manually since the wheels are provided. However, it is still possible to avoid using the wheels and rely on the system installation.

### Using Homebrew

```bash
brew install gstreamer libnice-gstreamer
```

The WebRTC plugin is enabled by default in the Homebrew package.


</hfoption>
<hfoption id="Windows">

It is not necessary to install GStreamer manually since the wheels are provided. However, it is still possible to avoid using the wheels and rely on the system installation.

### Step 1: Install GStreamer using the official installer

<div align="center">

[![Download GStreamer for Windows](https://img.shields.io/badge/Download-GStreamer%20for%20Windows-blue?style=for-the-badge&logo=windows&logoColor=white)](https://gstreamer.freedesktop.org/download/)

</div>

1. Download the **runtime** installer (MSVC version)
2. Install with the **Complete** installation option
3. Edit the environment variables and add to system PATH: `C:\Program Files\gstreamer\1.0\msvc_x86_64\bin`
4. Add to PYTHONPATH: `C:\Program Files\gstreamer\1.0\msvc_x86_64\lib\site-packages`

> **💡 Important:** Replace `C:\Program Files\gstreamer` with your actual GStreamer installation folder if you installed it in a different location.

</hfoption>
</hfoptions>

## ✅ Verify Installation

Finally, you can test your GStreamer installation as follows:

```bash
# Check version
gst-launch-1.0(.exe) --version

# Test basic functionalities
gst-launch-1.0 videotestsrc ! autovideosink

# Verify WebRTC plugin
gst-inspect-1.0 webrtcsrc
```

You should also be able to import GStreamer libraries in a Python environment:
```bash
python -c "import gi"
```


## Troubleshooting & Unit Tests

If you encounter issues with the stream, you can test the components individually as follows.

**Test 1: Manually create the WebRTC Server**
Run this GStreamer pipeline on the robot to verify the camera and encoder stack:

```bash
gst-launch-1.0 webrtcsink run-signalling-server=true meta="meta,name=reachymini" name=ws libcamerasrc ! capsfilter caps=video/x-raw,width=1280,height=720,framerate=60/1,format=YUY2,colorimetry=bt709,interlace-mode=progressive ! queue !  v4l2h264enc extra-controls="controls,repeat_sequence_header=1" ! 'video/x-h264,level=(string)4' ! ws. alsasrc device=hw:4 ! queue ! audioconvert ! audioresample ! opusenc ! audio/x-opus, rate=48000, channels=2 ! ws.
```
