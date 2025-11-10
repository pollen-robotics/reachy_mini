# Troubleshooting

## No microphone input
*For beta only*

There is a known issue where the microphone can be badly initialized. The best way to solve this is to reboot the microphone array using [xvf_host](https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/tree/master/host_control):

```bash
xvf_host(.exe) REBOOT 1
```

Then run [examples/debug/sound_record.py](../examples/debug/sound_record.py) to check that eveything is fine.

If the problem persist, you may want to check the connection of the flex cables ([slides 45 to 47](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_Assembly_Guide)).

## Sound Direction of Arrival is not working

*For beta only*

The microphone array requires firmware version 2.1.0 or higher to support this feature. The firmware is located in src/reachy_mini/assets/firmware/*.bin.

Refer to [Seeed documentation](https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/#update-firmware) for the upgrade process.

## Volume is too low


# Troubleshooting

## No Microphone Input
*For beta only*

There is a known issue where the microphone may not initialize correctly. The best way to resolve this is to reboot the microphone array using [xvf_host](https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/tree/master/host_control):

```bash
xvf_host(.exe) REBOOT 1
```

Then run [examples/debug/sound_record.py](../examples/debug/sound_record.py) to check that everything is working properly.

If the problem persists, check the connection of the flex cables ([see slides 45 to 47](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_Assembly_Guide)).

## Sound Direction of Arrival Not Working
*For beta only*

The microphone array requires firmware version 2.1.0 or higher to support this feature. The firmware is located in `src/reachy_mini/assets/firmware/*.bin`.

Refer to the [Seeed documentation](https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/#update-firmware) for the upgrade process.

## Volume Is Too Low
*Linux only*

Check with `alsamixer` that PCM1 is set to 100%. Then use PCM,0 to adjust the volume.

To make this change permanent:
```bash
CARD=$(aplay -l | grep -i "reSpeaker XVF3800 4-Mic Array" | head -n1 | sed -n 's/^card \([0-9]*\):.*/\1/p')
amixer -c "$CARD" set PCM,1 100%
sudo alsactl store "$CARD"
```

## Circular Buffer Overrun Warning

When starting a client with `with ReachyMini() as mini:` in Mujoco (--sim mode), you may see the following warning:

```bash
Circular buffer overrun. To avoid, increase fifo_size URL option. To survive in such case, use overrun_nonfatal option
```

This message comes from FFmpeg (embedded in OpenCV) while consuming the UDP video stream. It appears because the frames are not being used, causing the buffer to fill up. If you do not intend to use the frames, set `ReachyMini(media_backend="no_media")` or `ReachyMini(media_backend="default_no_video")`.