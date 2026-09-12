# Advanced: Reflash the Raspberry Pi OS image

> [!WARNING]
> **⚠️ Expert Guide Only**
>
> This guide explains how to reflash the ReachyMiniOS image to Reachy Mini's CM4. Doing this will _factory reset_ your Reachy Mini.
> 
> _Most users do not need this._ Reachy Mini comes pre-installed. Only follow these steps if you have a broken installation that you couldn't debug.

---

## The easy way: the Reachy Mini Flasher app (macOS / Windows)

On macOS and Windows, the [**Reachy Mini Flasher**](https://github.com/pollen-robotics/reachy_mini_flasher) does everything described on this page for you: it downloads the latest OS image, runs `rpiboot` to expose the CM4 eMMC (installing the USB driver first, on Windows), and writes the image to it.

Download the latest build for your system from the [releases page](https://github.com/pollen-robotics/reachy_mini_flasher/releases/latest) — a `.dmg` on macOS, a `.msi` or `.exe` installer on Windows — and open it.

> [!TIP]
> The installers are unsigned, so your system warns you the first time: on macOS, right-click the app and choose **Open**; on Windows, click **More info** then **Run anyway** in the SmartScreen dialog.

![Flasher welcome screen](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/flasher-welcome.png)

The app walks you through the hardware steps (open the head, set **SW1** to **DOWNLOAD**, plug **USB2**, power on):

![Flasher connect step](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/flasher-switch.png)

Once the robot is in download mode, it shows up in the app. Select it and continue:

![Reachy detected by the flasher](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/flasher-found.png)

Your system asks for permission (raw disk writes need admin rights: an administrator password on macOS, a UAC prompt on Windows), then the image is written. Keep the robot plugged in until it finishes:

![Flashing progress](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/flasher-progress.png)

The app then guides you through restoring the normal boot mode (switch back to **DEBUG**, unplug USB, close the head).

> [!NOTE]
> There is no Linux build. On Linux, follow the manual procedure below — which is also what the app automates, if you'd rather do it by hand.

---

## Download the OS image (and bmap)

First, download the latest OS image and `.bmap` file from:  
https://github.com/pollen-robotics/reachy-mini-os/releases

> [!TIP]
> The `.bmap` file is used by `bmaptool` (Linux/macOS). If you're flashing with Raspberry Pi Imager (Windows), you only need the OS image file.

---

## Install rpiboot

<hfoptions id="rpiboot">
<hfoption id="Linux / macOS">

Follow the building instructions here:  
https://github.com/raspberrypi/usbboot?tab=readme-ov-file#building-1



</hfoption>
<hfoption id="Windows">

Download and install the rpiboot GUI from the official Raspberry Pi repository:  
https://github.com/raspberrypi/usbboot/raw/master/win32/rpiboot_setup.exe

</hfoption>
</hfoptions>

---

## Install a flashing tool

<hfoptions id="flash-tool">
<hfoption id="Linux">

Install bmaptool:

```bash
sudo apt install bmap-tools
```

> [!TIP]
> _Linux users can use either `bmaptool` or Raspberry Pi Imager (the Windows option)._ Raspberry Pi Imager is generally _much slower_ than `bmaptool` for this workflow, so prefer `bmaptool` when available.

</hfoption>
<hfoption id="macOS">

Install bmaptool from the official repository:

```bash
python3 -m pip install --user "git+https://github.com/yoctoproject/bmaptool.git"
export PATH="$HOME/.local/bin:$PATH"
bmaptool --version
```

> [!TIP]
> _macOS users can use either `bmaptool` or Raspberry Pi Imager (the Windows option)._ Raspberry Pi Imager is generally _much slower_ than `bmaptool` for this workflow, so prefer `bmaptool` when available.

</hfoption>
<hfoption id="Windows">

Download and install Raspberry Pi Imager:  
https://www.raspberrypi.com/software/

</hfoption>
</hfoptions>

---

## Setup

1. **Shut down the robot completely** before proceeding.

2. Start **rpiboot** (it will wait for the robot to be connected):

   - **Linux / macOS**: run the `rpiboot` command in your terminal:

     ```bash
     sudo ./rpiboot -d mass-storage-gadget64
     ```

   - **Windows**: run the **RPiBoot** executable that you installed in the previous step.
   Make sure to use `rpiboot-CM4-CM5 - Mass storage Gadget` and not `rpiboot-CM-CM2-CM3`.
  
   ⚠️ In both cases, make sure you do not close the terminal window opened by rpiboot.
   
3. Set the switch to **DOWNLOAD (SW1)** on the head PCB:

   [![pcb_usb_and_switch](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/pcb_usb_and_switch.png)]()

4. Plug the USB cable (the one shown in the image above, named **USB2**).

5. **Power on the robot**.  
   `rpiboot` will run first and detect the robot, loading the mass-storage gadget (you can follow its progress in the terminal). When it completes, the internal eMMC appears as a mass-storage device.

---

## Unmount and Flash the ISO

<hfoptions id="flash-iso">
<hfoption id="Linux">

> [!WARNING]
> ⚠️ Make sure the device is unmounted before flashing.

### Check and Unmount the Device

Your device should be visible as `/dev/sdx` (something like `/dev/sda`).

**Check mounted partitions** by running: 

```bash
lsblk
```

If you see `bootfs` and `rootfs` like below, it means it is **mounted**:
```
sda           8:0    1  14.6G  0 disk
├─sda1        8:1    1   512M  0 part /media/<username>/bootfs
└─sda2        8:2    1  14.1G  0 part /media/<username>/rootfs
```

**Unmount** the partitions:
```bash
sudo umount /media/<username>/bootfs
sudo umount /media/<username>/rootfs
```

### Flash the ISO

```bash
sudo bmaptool copy <reachy_mini_os>.zip --bmap <reachy_mini_os>.bmap /dev/sda
```

For example with the `v0.0.10` release: 

```bash
sudo bmaptool copy image_2025-11-19-reachyminios-lite-v0.0.10.zip --bmap 2025-11-19-reachyminios-lite-v0.0.10.bmap /dev/sda
```

</hfoption>
<hfoption id="macOS">

> [!WARNING]
> ⚠️ Make sure the device is unmounted before flashing.

### Check and Unmount the Device

Your device should be visible as `/dev/diskX` (something like `/dev/disk4`).

**Check mounted partitions** by running:

```bash
mount
```

Look for entries like `/dev/disk4s1` or `/dev/disk4s2` that mention `bootfs` or `rootfs`.

**Unmount** the entire disk (not individual partitions):
```bash
diskutil unmountDisk /dev/disk4
```

Replace `/dev/disk4` with your actual disk identifier.

> [!NOTE]
> `unmountDisk` unmounts all volumes on the disk (`bootfs`, `rootfs`...) at once.

### Flash the ISO

> [!WARNING]
> Use `/dev/rdiskX` (note the **`r`** prefix!) instead of `/dev/diskX`. The `r` prefix provides raw disk access, which is mandatory for the flash command to succeed.

```bash
sudo bmaptool copy <reachy_mini_os>.zip --bmap <reachy_mini_os>.bmap /dev/rdiskX
```

For example with the `v0.0.10` release (replace `/dev/rdisk4` with your actual disk identifier):

```bash
sudo bmaptool copy image_2025-11-19-reachyminios-lite-v0.0.10.zip --bmap 2025-11-19-reachyminios-lite-v0.0.10.bmap /dev/rdisk4
```

</hfoption>
<hfoption id="Windows (Raspberry Pi Imager)">

Use the **Raspberry Pi Imager** executable to flash the OS image:

1. Open **Raspberry Pi Imager**
2. Select `Raspberry Pi 4` as the device
3. Select `Use custom` for the operating system, and provide the downloaded OS image (`.zip`, or the extracted `.img`)
4. Select the only available disk for the storage system (typically `RPi-MSD- 0001`)
5. Click **Write**

</hfoption>
</hfoptions>

---

## Restore Normal Boot Mode

1. **Power off the robot**
2. **Move the switch back to DEBUG**
3. **Disconnect the USB cable**
4. **Power the robot back on**

---

## Check that Everything Is Working

**Connect your computer to the robot's WiFi hotspot:**
- Network name: `reachy-mini-ap`
- Password: `reachy-mini`

SSH into the robot: 

```bash
ssh pollen@reachy-mini.local
# password: root
```

Then run: 

```bash
reachyminios_check
```

If successful, you should see:
```bash
Image validation PASSED
```
