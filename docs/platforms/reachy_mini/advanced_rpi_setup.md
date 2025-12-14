# Advanced: Manual Raspberry Pi Installation

> **⚠️ Expert Guide Only**
>
> This guide explains how to install the Reachy Mini software stack from scratch on a fresh Raspberry Pi.
> **Most users do not need this.** Reachy Mini comes pre-installed. Only follow these steps if you are creating a new SD card image or building a custom setup.

## 1. Install OS

1.  Download and install **Raspberry Pi OS Lite (64-bit)** using the [official documentation](https://www.raspberrypi.com/documentation/computers/getting-started.html#installing-the-operating-system).
2.  **Important:** During the configuration, make sure to:
    * Set a hostname (e.g., `reachy-mini`).
    * Setup Wi-Fi credentials.
    * Enable **SSH** for remote access.

## 2. Install System Dependencies

You need to install GStreamer libraries and other system tools required for audio/video streaming.

Run the following command on the Raspberry Pi:

```bash
sudo apt-get update
sudo apt-get install libgstreamer-plugins-bad1.0-dev libgstreamer-plugins-base1.0-dev \
libgstreamer1.0-dev libglib2.0-dev libssl-dev git libgirepository1.0-dev \
libcairo2-dev libportaudio2 gstreamer1.0-libcamera librpicam-app1 libssl-dev \
libnice10 gstreamer1.0-plugins-good gstreamer1.0-alsa gstreamer1.0-plugins-bad \
gstreamer1.0-nice
```

## 3. Install Rust

The WebRTC plugin requires Rust. Install it via `rustup`:

```bash
curl --proto '=https' --tlsv1.2 -sSf [https://sh.rustup.rs](https://sh.rustup.rs) | sh
source $HOME/.cargo/env
```

## 4. Compile WebRTC Plugin

We use a specific version of the GStreamer WebRTC plugin written in Rust.

```bash
# Clone the repository
git clone [https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git](https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git)
cd gst-plugins-rs
git checkout 0.14.1

# Install cargo-c builder
cargo install cargo-c

# Create installation directory
sudo mkdir /opt/gst-plugins-rs
sudo chown $USER /opt/gst-plugins-rs

# Compile and install (this may take a while)
cargo cinstall -p gst-plugin-webrtc --prefix=/opt/gst-plugins-rs --release

# Register the plugin path
echo 'export GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/aarch64-linux-gnu/' >> ~/.bashrc
source ~/.bashrc
```

## 5. Install Reachy Mini Daemon

Install the python package with the GStreamer extras:

```bash
pip install -e .[gstreamer]
```

## 6. Usage

### Start the Daemon (Server)
To start the daemon with WebRTC streaming enabled (Wireless mode):

```bash
reachy-mini-daemon --wireless-version
```

### Start a Client (Viewer)
On your computer (not the robot), you can run the debug client to view the camera feed and hear audio.

```bash
# Replace <Reachy Mini IP> with the actual IP address
python examples/debug/gstreamer_client.py --signaling-host <Reachy Mini IP>
```

> **Note:** GStreamer must also be installed on your client machine (via `brew` on macOS or standard packages on Linux).

## 7. Troubleshooting & Unit Tests

If you encounter issues with the stream, you can test the components individually.

**Test 1: Manually create the WebRTC Server**
Run this GStreamer pipeline on the robot to verify the camera and encoder stack:

```bash
gst-launch-1.0 webrtcsink run-signalling-server=true meta="meta,name=reachymini" name=ws libcamerasrc ! capsfilter caps=video/x-raw,width=1280,height=720,framerate=60/1,format=YUY2,colorimetry=bt709,interlace-mode=progressive ! queue !  v4l2h264enc extra-controls="controls,repeat_sequence_header=1" ! 'video/x-h264,level=(string)4' ! ws. alsasrc device=hw:4 ! queue ! audioconvert ! audioresample ! opusenc ! audio/x-opus, rate=48000, channels=2 ! ws.
```

**Test 2: Send Audio to Reachy**
Send an audio RTP stream to port 5000 to test the speakers:

```bash
gst-launch-1.0 audiotestsrc ! audioconvert ! audioresample ! opusenc ! audio/x-opus, rate=48000, channels=2 ! rtpopuspay pt=96 ! udpsink host=<ROBOT_IP> port=5000
```