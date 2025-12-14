# Raspberry Pi Installation

## RPI Lite OS

Follow the [official documentation](https://www.raspberrypi.com/documentation/computers/getting-started.html#installing-the-operating-system) to install the Raspberry Pi OS Lite (64 bits).

It is recommended to setup a wifi password and a ssh connection.

## Gstreamer

```bash
sudo apt-get install libgstreamer-plugins-bad1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libglib2.0-dev libssl-dev git libgirepository1.0-dev libcairo2-dev libportaudio2  gstreamer1.0-libcamera librpicam-app1 libssl-dev libnice10 gstreamer1.0-plugins-good gstreamer1.0-alsa gstreamer1.0-plugins-bad gstreamer1.0-nice
```

## Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
## Webrtc plugin

```bash
git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git

cd gst-plugins-rs

git checkout 0.14.1

cargo install cargo-c

sudo mkdir /opt/gst-plugins-rs

sudo chown reachy /opt/gst-plugins-rs

cargo cinstall -p gst-plugin-webrtc --prefix=/opt/gst-plugins-rs --release

echo 'export GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/aarch64-linux-gnu/' >> ~/.bashrc
```

## Install Daemon

Install with gstreamer extra dependencies

pip install -e .[gstreamer]

## Usage

### Daemon

The webrtc streaming will start automatically with the wireless option:

```bash
reachy-mini-daemon --wireless-version
```

### Client

This should open view of the camera, and play back the sound.

```bash
python examples/debug/gstreamer_client.py --signaling-host <Reachy Mini ip>
```

It is assumed that gstreamer is installed in your machine. For Linux users you may want to follow the above procedure. For MacOS, please install via [brew](https://gstreamer.freedesktop.org/download/#macos). *ToDo* For Windows please make a conda environement.


## Unit tests

### Manually create the webrtcsrc server

```bash
gst-launch-1.0 webrtcsink run-signalling-server=true meta="meta,name=reachymini" name=ws libcamerasrc ! capsfilter caps=video/x-raw,width=1280,height=720,framerate=60/1,format=YUY2,colorimetry=bt709,interlace-mode=progressive ! queue !  v4l2h264enc extra-controls="controls,repeat_sequence_header=1" ! 'video/x-h264,level=(string)4' ! ws. alsasrc device=hw:4 ! queue ! audioconvert ! audioresample ! opusenc ! audio/x-opus, rate=48000, channels=2 ! ws.
```

### Send sound to Reachy Mini

Send an audio RTP stream to the port 5000

```bash
gst-launch-1.0     audiotestsrc !     audioconvert !     audioresample !     opusenc !  audio/x-opus, rate=48000, channels=2 !   rtpopuspay pt=96 !          udpsink host=10.0.1.38 port=5000
```



