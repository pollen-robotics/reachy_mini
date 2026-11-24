#!/bin/sh
# /init of the one-shot flasher initramfs (see README.md).
#
# Booted once through the firmware's tryboot mode, so any reboot that does not
# come from a successful flash lands back on the old OS: until the image write
# starts, every failure path is "log, reboot, nothing changed".
#
# Everything runs from RAM: the old rootfs is only mounted read-only to copy
# the plan and the state to carry over into tmpfs. The image is downloaded
# over WiFi into tmpfs (or copied from the old rootfs when the plan gives a
# local zip), and the old rootfs is unmounted before /dev/mmcblk0 is written.

export PATH=/bin
DISK=/dev/mmcblk0
ROOT_PART=${DISK}p2
PLAN_DIR=/var/lib/reachy-os-upgrade
LOG=/work/flasher.log

mount -t proc proc /proc
mount -t sysfs sys /sys
mount -t devtmpfs dev /dev
mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
mount -t tmpfs -o size=$((mem_kb * 85 / 100))k tmpfs /work

log() {
	msg="[flasher $(cut -d' ' -f1 /proc/uptime)s] $*"
	echo "$msg" >> "$LOG"
	echo "$msg" > /dev/kmsg 2>/dev/null
	echo "$msg" > /dev/console 2>/dev/null
}

# Append the log to whichever rootfs is now on the disk, so the next boot can
# read what happened in /var/log/reachy-os-flasher.log.
save_log() {
	if mount -t ext4 "$ROOT_PART" /old 2>/dev/null; then
		mkdir -p /old/var/log
		cat "$LOG" >> /old/var/log/reachy-os-flasher.log
		sync
		umount /old
	fi
}

# Before the point of no return: the disk is untouched, reboot into the old OS.
abort() {
	log "ABORT: $*"
	umount /old 2>/dev/null
	save_log
	sync
	reboot -f
}

wait_for() {
	i=0
	while [ ! -e "$1" ]; do
		i=$((i + 1))
		[ "$i" -gt "${2:-50}" ] && return 1
		sleep 0.2
	done
}

log "flasher started, kernel $(uname -r)"
# systemd can leave the hardware watchdog armed across the reboot; the flash
# takes longer than its reboot timeout, so keep feeding it.
if [ -c /dev/watchdog ]; then
	watchdog -t 5 -T 60 /dev/watchdog && log "feeding /dev/watchdog"
fi

wait_for "$ROOT_PART" || abort "$ROOT_PART never appeared"
mount -t ext4 -o ro "$ROOT_PART" /old || abort "cannot mount old rootfs"
[ -f "/old$PLAN_DIR/plan.env" ] || abort "no plan at $PLAN_DIR/plan.env"
. "/old$PLAN_DIR/plan.env"
# No RTC on the board: without this the clock says 1970 and TLS fails.
date -s "@$((EPOCH + 60))" > /dev/null
log "plan: url=$URL zip=$ZIP dry_run=$DRY_RUN"

cp "/old$PLAN_DIR/shadow-pollen" "/old$PLAN_DIR/restore-apps.json" /work/ || abort "plan files missing"
[ -z "$URL" ] || cp "/old$PLAN_DIR/wpa_supplicant.conf" /work/ || abort "no WiFi config"
log "saving state to carry over..."
# busybox tar has no -T: pass the list as arguments, split on newlines only.
old_ifs=$IFS
IFS='
'
tar -c -f /work/preserve.tar -C /old $(cat "/old$PLAN_DIR/preserve.list") 2>> "$LOG" ||
	abort "could not save the state to carry over"
IFS=$old_ifs
log "  $(du -m /work/preserve.tar | cut -f1) MB"
if [ -n "$ZIP" ]; then
	log "copying image zip to RAM..."
	cp "/old$ZIP" /work/image.zip || abort "copy of $ZIP failed"
fi
umount /old || abort "cannot unmount old rootfs"

wifi_up() {
	# brcmfmac loads its vendor module through the kernel's modprobe helper.
	rm -f /bin/modprobe
	printf '#!/bin/sh\nfor m; do :; done\ninsmod /mods/$(echo "$m" | tr _ -).ko 2>/dev/null\nexit 0\n' > /bin/modprobe
	chmod +x /bin/modprobe
	echo /bin/modprobe > /proc/sys/kernel/modprobe
	while read -r m; do
		insmod "/mods/$m.ko" 2>> "$LOG"
	done < /mods/order
	wait_for /sys/class/net/wlan0 100 || return 1
	ip link set wlan0 up
	mkdir -p /run/wpa_supplicant
	wpa_supplicant -B -D nl80211 -i wlan0 -c /work/wpa_supplicant.conf >> "$LOG" 2>&1 || return 1
	cat > /bin/dhcp-script <<-'EOF'
		#!/bin/sh
		case "$1" in bound | renew)
			ifconfig "$interface" "$ip" netmask "${subnet:-255.255.255.0}"
			[ -n "$router" ] && route add default gw "${router%% *}" dev "$interface"
			: > /etc/resolv.conf
			for d in $dns; do echo "nameserver $d" >> /etc/resolv.conf; done
			;;
		esac
	EOF
	chmod +x /bin/dhcp-script
	i=0
	until [ "$(cat /sys/class/net/wlan0/operstate)" = up ]; do
		i=$((i + 1))
		[ "$i" -gt 60 ] && return 1
		sleep 1
	done
	udhcpc -i wlan0 -s /bin/dhcp-script -n -q -t 20 -T 3 >> "$LOG" 2>&1
}

if [ -n "$URL" ]; then
	log "bringing up WiFi..."
	wifi_up || abort "WiFi did not come up"
	log "  $(ip -4 -o addr show wlan0 | awk '{print $4}')"
	log "downloading $URL..."
	n=0
	until curl -fsSL --cacert /etc/ca-certificates.crt --connect-timeout 20 \
		-C - -o /work/image.zip "$URL" 2>> "$LOG"; do
		n=$((n + 1))
		[ "$n" -ge 5 ] && abort "download failed"
		log "  download interrupted, resuming ($n)"
		sleep 5
	done
	log "  $(du -m /work/image.zip | cut -f1) MB downloaded"
	ENTRY=$(unzip -Z1 /work/image.zip | grep -m1 '\.img$') || abort "no .img in the zip"
fi

if [ -n "$ZIP_SHA256" ]; then
	log "checking sha256..."
	got=$(sha256sum /work/image.zip | cut -d' ' -f1)
	[ "$got" = "$ZIP_SHA256" ] || abort "sha256 mismatch: got $got want $ZIP_SHA256"
fi
log "testing zip CRC..."
unzip -tq /work/image.zip "$ENTRY" >> "$LOG" 2>&1 || abort "zip CRC test failed"

if [ "$DRY_RUN" = 1 ]; then
	log "DRY RUN OK: everything up to the image write succeeded"
	save_log
	sync
	reboot -f
fi

# ---- point of no return: from here a failure needs a USB (rpiboot) reflash ----
log "writing $ENTRY to $DISK..."
if ! unzip -p /work/image.zip "$ENTRY" | dd of="$DISK" bs=4M conv=fsync 2>> "$LOG"; then
	log "image write FAILED, dropping to a shell on the console"
	exec sh
fi
rm -f /work/image.zip
sync
log "image written"

blockdev --rereadpt "$DISK"
wait_for "$ROOT_PART" || { log "new rootfs never appeared"; exec sh; }

log "restoring state..."
mount -t ext4 "$ROOT_PART" /new || { log "cannot mount new rootfs"; exec sh; }
tar -x -f /work/preserve.tar -C /new >> "$LOG" 2>&1 || log "WARNING: state restore failed"
# Only pollen's password line: a whole /etc/shadow from the old image would
# drop the system users the new packages added.
line=$(cat /work/shadow-pollen)
if [ -n "$line" ]; then
	awk -v l="$line" -F: '$1 == "pollen" { print l; next } { print }' \
		/new/etc/shadow > /work/shadow && cat /work/shadow > /new/etc/shadow
fi
# The SSH host keys were carried over: keep first boot from regenerating them.
if ls /new/etc/ssh/ssh_host_*_key > /dev/null 2>&1; then
	rm -f /new/etc/systemd/system/multi-user.target.wants/regenerate_ssh_host_keys.service
fi
# Apps are reinstalled by the new OS itself, once its daemon is up.
if [ "$(cat /work/restore-apps.json)" != "[]" ]; then
	cp /payload/restore_apps.py /new/usr/local/lib/reachy-restore-apps.py
	cp /payload/reachy-restore-apps.service /new/etc/systemd/system/
	ln -sf /etc/systemd/system/reachy-restore-apps.service \
		/new/etc/systemd/system/multi-user.target.wants/reachy-restore-apps.service
	mkdir -p /new/home/pollen/.config/reachy_mini
	cp /work/restore-apps.json /new/home/pollen/.config/reachy_mini/restore-apps.json
	chown -R 1000:1000 /new/home/pollen/.config
	log "apps to reinstall on first boot: $(cat /work/restore-apps.json)"
fi
log "flash complete, rebooting into the new OS"
cat "$LOG" >> /new/var/log/reachy-os-flasher.log
sync
umount /new
sync
reboot -f
