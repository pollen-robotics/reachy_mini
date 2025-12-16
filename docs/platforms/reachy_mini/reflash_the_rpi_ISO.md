# Advanced: Reflash the Raspberry Pi's ISO

> **⚠️ Expert Guide Only**
>
> This guide explains how to reflash the ReachyMiniOS ISO to Reachy Mini's CM4. Doing that will do a factory reset to your Reachy Mini.
> 
> **Most users do not need this.** Reachy Mini comes pre-installed. Only follow these steps if you have a broken installation that you couldn't debug


## Download the ISO and bmap

First, download the latest ISO and bmap from here : https://github.com/pollen-robotics/reachy-mini-os/releases

## Install rpiboot 

Follow the instructions here https://github.com/raspberrypi/usbboot?tab=readme-ov-file#building-1

## Install bmap tools

```bash
sudo apt install bmap-tools
```

## Setup 

```bash
sudo ./rpiboot -d mass-storage-gadget64
```

Set the switch to DOWNLOAD SW1 on the head PCB 

[![pcb_usb_and_switch](/docs/assets/pcb_usb_and_switch.png)]()

Plug the USB cable (the one shown in the image above, named USB2)

Your device should be visible as `/dev/sdx..` (something like that).

<details>
<summary>⚠️ Make sure it is unmounted</summary>


> Check by running : 
> 
> `lsblk`
>
> If you see `bootfs` and `rootfs` like below, it means it is **mounted**
> ```
> sda           8:0    1  14.6G  0 disk
> ├─sda1        8:1    1   512M  0 part /media/<username>/bootfs
> └─sda2        8:2    1  14.1G  0 part /media/<username>/rootfs
> ```
>
> Run this to **unmount**
> ```
> sudo umount /media/username/bootfs
> sudo umount /media/username/rootfs
> ```

</details>

## Flash the ISO 

```bash
sudo bmaptool copy <reachy_mini_os>.zip --bmap <reachy_mini_os>.bmap /dev/sda
```

For example with the `v0.0.10` release : 

```bash
sudo bmaptool copy image_2025-11-19-reachyminios-lite-v0.0.10.zip --bmap 2025-11-19-reachyminios-lite-v0.0.10.bmap /dev/sda
```


When it's done, you can ⚠️`move the switch back to DEBUG`⚠️ and reboot the robot.
