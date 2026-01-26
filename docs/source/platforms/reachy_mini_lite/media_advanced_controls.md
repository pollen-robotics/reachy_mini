# Advanced Media Controls

This page describes advanced settings to fine-tune the camera and sound system of Reachy Mini Lite. Only the camera access differs slightly from the standard version.

## Camera

The Raspberry Pi camera is mounted on a CSI-to-USB adapter and is detected by the system as a UVC camera. It can be accessed directly by any program capable of opening a camera device.

### Windows and macOS

Currently, the default backend for these platforms is OpenCV. The parameters can be set in the [code](https://github.com/pollen-robotics/reachy_mini/tree/main/src/reachy_mini/media/camera_opencv.py):

```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution.value[0])
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution.value[1])

# Example camera control settings:
# self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
# self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
# self.cap.set(cv2.CAP_PROP_SATURATION, 64)
```

Please refer to the Linux section below for a list of available parameters.

### Linux

Linux users can check the camera using:

```bash
v4l2-ctl --list-devices
# Reachy Mini Camera: Reachy Mini (usb-0000:00:14.0-3.4):
#	/dev/video2
#	/dev/video3
#	/dev/media1
``` 

```bash
v4l2-ctl --device=/dev/video2 --list-formats-ext
# ioctl: VIDIOC_ENUM_FMT
# 	Type: Video Capture
# 
# 	[0]: 'MJPG' (Motion-JPEG, compressed)
# 		Size: Discrete 3840x2592
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 1920x1080
# 			Interval: Discrete 0.017s (60.000 fps)
# 		Size: Discrete 3840x2160
# 			Interval: Discrete 0.033s (30.000 fps)
# 		Size: Discrete 3264x2448
# 			Interval: Discrete 0.033s (30.000 fps)
# 	[1]: 'YUYV' (YUYV 4:2:2)
# 		Size: Discrete 3840x2592
# 			Interval: Discrete 1.000s (1.000 fps)
# 		Size: Discrete 1920x1080
# 			Interval: Discrete 0.200s (5.000 fps)
# 		Size: Discrete 3840x2160
# 			Interval: Discrete 1.000s (1.000 fps)
# 		Size: Discrete 3264x2448
# 			Interval: Discrete 1.000s (1.000 fps)
``` 

```bash
v4l2-ctl --device=/dev/video2 --list-ctrls

# User Controls
# 
# brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
# contrast 0x00980901 (int)    : min=0 max=95 step=1 default=1 value=1
# saturation 0x00980902 (int)    : min=0 max=100 step=1 default=48 value=48
# hue 0x00980903 (int)    : min=-2000 max=2000 step=1 default=0 value=0
# white_balance_automatic 0x0098090c (bool)   : default=1 value=1
# gamma 0x00980910 (int)    : min=80 max=160 step=1 default=100 value=100
# gain 0x00980913 (int)    : min=0 max=255 step=1 default=32 value=32
# power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=1 value=1 (50 Hz)
# white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=1 default=4600 value=4600 flags=inactive
# sharpness 0x0098091b (int)    : min=0 max=7 step=1 default=2 value=2
# backlight_compensation 0x0098091c (int)    : min=0 max=10 step=1 default=2 value=2

# Camera Controls
# 
# auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
# exposure_time_absolute 0x009a0902 (int)    : min=3 max=2047 step=1 default=166 value=166 flags=inactive
# exposure_dynamic_framerate 0x009a0903 (bool)   : default=0 value=1
``` 

Using GStreamer allows you to directly visualize the effect of parameter changes:

```bash
gst-launch-1.0 v4l2src device=/dev/video2 extra-controls=s,exposure_auto=1,exposure_absolute=100,saturation=0 ! videoconvert ! autovideosink
```

At the SDK level, the camera is controlled by GStreamer using the [v4l2src](https://github.com/pollen-robotics/reachy_mini/tree/main/src/reachy_mini/media/camera_gstreamer.py) component. You can view available parameters with the following command:

The parameters can be set in [the code](https://github.com/pollen-robotics/reachy_mini/tree/main/src/reachy_mini/media/camera_gstreamer.py) as follows:

```python
camsrc = Gst.ElementFactory.make("v4l2src")
camsrc.set_property("device", cam_path)
# Example camera control settings:
# extra_controls_structure = Gst.Structure.new_empty("extra-controls")
# extra_controls_structure.set_value("saturation", 64)
# extra_controls_structure.set_value("brightness", 50)
# camsrc.set_property("extra-controls", extra_controls_structure)
self.pipeline.add(camsrc)
```

The default OpenCV backend is also available for Linux.

## Microphones and Speakers

Refer to the [Reachy Mini documentation](../reachy_mini/media_advanced_controls.md#microphones-and-speakers). The hardware is exactly the same for the Lite version.
