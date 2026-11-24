#!/bin/bash
# Reflash this robot's eMMC with a new ReachyMiniOS image, without USB.
#
# Runs on the robot as root. Saves what the flasher needs (WiFi credentials,
# the list of state to carry over, installed apps), builds a one-shot flasher
# initramfs from this system's binaries, and reboots into it through the
# firmware's tryboot mode. With a URL the flasher downloads the image itself
# into RAM, so no free disk space is needed. See README.md.
#
# Usage: reachy-os-upgrade.sh [--dry-run] [--yes] [--sha256 HEX] <zip URL or path>
set -euo pipefail

PLAN_DIR=/var/lib/reachy-os-upgrade
BOOT=/boot/firmware
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# RAM left to the kernel, the flasher and the WiFi stack, outside the tmpfs.
RESERVED_MB=600

DRY_RUN=0
YES=0
ZIP_SHA256=""
SRC=""
while [ $# -gt 0 ]; do
	case "$1" in
	--dry-run) DRY_RUN=1 ;;
	--yes) YES=1 ;;
	--sha256) ZIP_SHA256="$2"; shift ;;
	-*) echo "unknown option $1" >&2; exit 2 ;;
	*) SRC="$1" ;;
	esac
	shift
done
[ -n "$SRC" ] || { echo "usage: $0 [--dry-run] [--yes] [--sha256 HEX] <zip URL or path>" >&2; exit 2; }
[ "$EUID" -eq 0 ] || { echo "run as root" >&2; exit 1; }

die() { echo "ERROR: $*" >&2; exit 1; }

# ---- preflight ----
grep -q "Compute Module 4" /proc/device-tree/model || die "not a CM4: $(tr -d '\0' < /proc/device-tree/model)"
[ -b /dev/mmcblk0p2 ] && findmnt -no SOURCE / | grep -q mmcblk0p2 || die "rootfs is not on eMMC p2"
mem_mb=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
[ "$mem_mb" -ge 3500 ] || die "need a 4 GB CM4, this one has ${mem_mb} MB"
[ -f "$BOOT/config.txt" ] || die "no $BOOT/config.txt"

rm -rf "$PLAN_DIR"
mkdir -p "$PLAN_DIR"
chmod 700 "$PLAN_DIR"

# ---- image source ----
ZIP=""
URL=""
ENTRY=""
if [[ "$SRC" =~ ^https?:// ]]; then
	URL="$SRC"
	zip_bytes=$(curl -fsIL "$URL" | tr -d '\r' | awk 'tolower($1) == "content-length:" {n = $2} END {print n}')
	[ -n "$zip_bytes" ] || die "cannot reach $URL"
else
	# A local zip stays where it is; the flasher copies it to RAM.
	ZIP=$(realpath "$SRC")
	zip_bytes=$(stat -c %s "$ZIP")
	ENTRY=$(unzip -Z1 "$ZIP" | grep -m1 '\.img$') || die "no .img in $ZIP"
	[ -n "$ZIP_SHA256" ] || ZIP_SHA256=$(sha256sum "$ZIP" | cut -d' ' -f1)
fi
zip_mb=$((zip_bytes / 1024 / 1024))
budget_mb=$((mem_mb - RESERVED_MB - zip_mb))
[ "$budget_mb" -ge 50 ] || die "image (${zip_mb} MB) does not fit in RAM"

# ---- WiFi for the flasher ----
if [ -n "$URL" ]; then
	conn=$(nmcli -t -f NAME,TYPE,DEVICE connection show --active | awk -F: '$2 == "802-11-wireless" && $3 == "wlan0" {print $1; exit}')
	[ -n "$conn" ] || die "not connected to WiFi on wlan0"
	[ "$(nmcli -s -g 802-11-wireless.mode connection show "$conn")" != ap ] ||
		die "robot is in hotspot mode, connect it to a WiFi network first"
	ssid=$(nmcli -s -g 802-11-wireless.ssid connection show "$conn")
	keymgmt=$(nmcli -s -g 802-11-wireless-security.key-mgmt connection show "$conn")
	psk=$(nmcli -s -g 802-11-wireless-security.psk connection show "$conn")
	SSID="$ssid" KEYMGMT="$keymgmt" PSK="$psk" python3 - > "$PLAN_DIR/wpa_supplicant.conf" <<'PYEOF'
import os
ssid = os.environ["SSID"].encode()
km = os.environ["KEYMGMT"]
psk = os.environ["PSK"]
print("ctrl_interface=/run/wpa_supplicant")
print("network={")
print(f"  ssid={ssid.hex()}")
print("  scan_ssid=1")
if km in ("", "none"):
    print("  key_mgmt=NONE")
elif km == "sae":
    print("  key_mgmt=SAE")
    print("  ieee80211w=2")
    print(f'  sae_password="{psk}"')
elif km == "wpa-psk":
    print("  key_mgmt=WPA-PSK")
    print(f"  psk={psk}" if len(psk) == 64 else f'  psk="{psk}"')
else:
    raise SystemExit(f"unsupported key-mgmt {km!r}")
print("}")
PYEOF
	chmod 600 "$PLAN_DIR/wpa_supplicant.conf"
	echo "Flasher will join WiFi '$ssid' ($keymgmt)"
fi

# ---- state to carry over (all saved WiFi networks included) ----
getent shadow pollen > "$PLAN_DIR/shadow-pollen"
(
	cd /
	for p in \
		etc/NetworkManager/system-connections \
		etc/ssh/ssh_host_* \
		home/pollen/.ssh \
		home/pollen/.config/reachy_mini \
		home/pollen/.CONFIGURED \
		home/pollen/.cache/huggingface/token \
		home/pollen/.cache/huggingface/*_token \
		home/pollen/.cache/huggingface/stored_tokens* \
		home/pollen/.cache/huggingface/reachy_mini_daemon_credentials.json; do
		[ -e "$p" ] && echo "$p"
	done
) > "$PLAN_DIR/preserve.list"
state_mb=$(cd / && du -scm $(cat "$PLAN_DIR/preserve.list") | tail -1 | cut -f1)

# The HF model cache is only a cache, but a large one to download again:
# carry it over when it fits in the RAM left next to the image.
hub=home/pollen/.cache/huggingface/hub
if [ -d "/$hub" ]; then
	hub_mb=$(du -sm "/$hub" | cut -f1)
	if [ $((state_mb + hub_mb)) -le "$budget_mb" ]; then
		echo "$hub" >> "$PLAN_DIR/preserve.list"
		echo "Keeping the HF cache (${hub_mb} MB)"
	else
		echo "Not keeping the HF cache (${hub_mb} MB > ${budget_mb} MB of RAM left)"
	fi
fi
echo "$(wc -l < "$PLAN_DIR/preserve.list") paths to carry over"

# Installed apps cannot be copied (the new image has its own Python and SDK):
# record them, the new OS reinstalls them from the app store on first boot.
if curl -fsS -m 20 http://localhost:8000/api/apps/list-available/installed > "$PLAN_DIR/installed.json"; then
	python3 -c 'import json,sys; print(json.dumps(sorted({a["name"] for a in json.load(sys.stdin)})))' \
		< "$PLAN_DIR/installed.json" > "$PLAN_DIR/restore-apps.json"
	echo "Apps to reinstall: $(cat "$PLAN_DIR/restore-apps.json")"
else
	echo "WARNING: daemon not reachable, installed apps will not be reinstalled"
	echo "[]" > "$PLAN_DIR/restore-apps.json"
fi
rm -f "$PLAN_DIR/installed.json"

cat > "$PLAN_DIR/plan.env" <<EOF
URL='$URL'
ZIP='$ZIP'
ENTRY='$ENTRY'
ZIP_SHA256='$ZIP_SHA256'
DRY_RUN=$DRY_RUN
EPOCH=$(date +%s)
EOF

# ---- flasher initramfs, built from this system's own binaries ----
echo "Building flasher initramfs..."
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
mkdir -p "$root"/{bin,proc,sys,dev,work,old,new,etc,run,mods,payload,lib/firmware/brcm}
ln -s bin "$root/sbin"
add_bin() {
	cp -L "$1" "$root/bin/"
	{ ldd "$1" || true; } | grep -o '/[^ ]*' | while read -r lib; do
		mkdir -p "$root$(dirname "$lib")"
		cp -L "$lib" "$root$lib"
	done
}
add_bin "$(command -v busybox)"
for applet in $(busybox --list); do
	[ "$applet" = busybox ] || ln -sf busybox "$root/bin/$applet"
done
# Info-ZIP unzip over busybox's applet: the image entry is > 4 GB (zip64).
rm -f "$root/bin/unzip"
add_bin "$(command -v unzip)"
add_bin "$(command -v curl)"
add_bin /usr/sbin/wpa_supplicant
cp -L /etc/ssl/certs/ca-certificates.crt "$root/etc/"

# WiFi driver stack, decompressed: busybox insmod cannot read .ko.xz.
kver=$(uname -r)
for m in rfkill cfg80211 brcmutil brcmfmac brcmfmac-cyw brcmfmac-wcc; do
	f=$(find "/lib/modules/$kver" -name "$m.ko*" | head -1)
	[ -n "$f" ] || continue
	case "$f" in
	*.xz) python3 -c 'import lzma,sys; sys.stdout.buffer.write(lzma.open(sys.argv[1]).read())' "$f" ;;
	*) cat "$f" ;;
	esac > "$root/mods/$m.ko"
	echo "$m" >> "$root/mods/order"
done
cp -L /lib/firmware/regulatory.db* "$root/lib/firmware/" 2>/dev/null || true
for f in /lib/firmware/brcm/brcmfmac43455-sdio.*; do
	case "$f" in
	*raspberrypi,4-compute-module* | */brcmfmac43455-sdio.bin | */brcmfmac43455-sdio.txt | */brcmfmac43455-sdio.clm_blob)
		cp -L "$f" "$root/lib/firmware/brcm/" ;;
	esac
done

install -m 0755 "$HERE/flasher-init.sh" "$root/init"
install -m 0755 "$HERE/restore_apps.py" "$root/payload/"
install -m 0644 "$HERE/reachy-restore-apps.service" "$root/payload/"
(cd "$root" && find . | cpio -o -H newc --quiet | gzip -1) > "$BOOT/reachy-flasher.img"
echo "  flasher: $(du -h "$BOOT/reachy-flasher.img" | cut -f1) on the boot partition"

# ---- one-shot boot config ----
sed 's/$/ rdinit=\/init/; s/ root=[^ ]*//' "$BOOT/cmdline.txt" > "$BOOT/reachy-flasher-cmdline.txt"
{
	cat "$BOOT/config.txt"
	echo
	echo "[all]"
	echo "initramfs reachy-flasher.img followkernel"
	echo "cmdline=reachy-flasher-cmdline.txt"
} > "$BOOT/tryboot.txt"
sync

echo
echo "Ready: ${URL:-$ZIP} (${zip_mb} MB), dry_run=$DRY_RUN."
if [ "$DRY_RUN" = 0 ]; then
	echo "The robot will reboot, overwrite its eMMC and reboot into the new OS."
	echo "Keep it powered: a power cut during the write needs a USB reflash."
fi
if [ "$YES" = 0 ]; then
	read -r -p "Reboot into the flasher now? [y/N] " ans
	[ "$ans" = y ] || { echo "Not rebooting. Run: reboot '0 tryboot'"; exit 0; }
fi
reboot '0 tryboot'
