/**
 * Offline-first latch for the daemon-update reboot watch.
 *
 * Problem: when the daemon restarts mid-update, the shell tears the
 * iframe down and waits for the robot to reappear on central before
 * telling `DaemonUpdateGate` the update is done. But central can keep
 * the robot's PRE-reboot registration listed for a while (the SSE
 * offline event or the listing TTL hasn't fired yet), so "the robot is
 * listed" alone reads the STALE entry as "back online" seconds after
 * the reboot started - the gate then declares success while the robot
 * is still down, and the user re-picks a dead robot.
 *
 * Peer-id exclusion doesn't fix this: the embed re-resolves the live
 * peer id right before dialing, so the shell doesn't reliably know
 * which id the stale listing carries. What IS reliable is the order of
 * observations: the dying relay connection makes central drop the old
 * registration (realtime SSE `roles: []` event), and only then does the
 * rebooted daemon re-register under a fresh id. So the watch requires
 * seeing the robot ABSENT from the list at least once (the sticky
 * `sawTargetOffline` boolean) before accepting its presence as "back
 * online". Callers must skip observations while the list is still in
 * its initial-loading window, where an empty array means "no data yet".
 *
 * If central never drops the stale entry (not observed in practice -
 * the relay socket dies with the daemon), the watch never completes and
 * the gate's reboot timeout produces an honest "check your robot"
 * message instead of a false success.
 */
import type { RobotInfo } from './sdk-types';

/** What the shell remembers about the robot it dropped mid-update. */
export interface RebootTarget {
  /** Stable hardware id (serial), the strongest match. */
  hardwareId: string | null;
  /** Advertised name, fallback for daemons without a hardware id. */
  name: string | null;
}

/**
 * Is the rebooting robot present in the central listing? Matched on
 * hardware id first (survives the peer-id rotation), then name. With
 * neither identity (older daemon), fall back to "any robot listed":
 * combined with the offline-first latch this means "the list emptied,
 * then repopulated", which is correct for single-robot accounts and
 * degrades to the gate's reboot timeout on multi-robot ones.
 */
export function isTargetListed(
  robots: readonly RobotInfo[],
  target: RebootTarget,
): boolean {
  if (target.hardwareId) {
    return robots.some((r) => r.hardwareId === target.hardwareId);
  }
  if (target.name) return robots.some((r) => r.meta?.name === target.name);
  return robots.length > 0;
}

/** Sticky offline latch, advanced on every settled observation: once
 *  the target has been seen absent, it stays seen. The watch completes
 *  when `sawTargetOffline(...) && isTargetListed(...)` - never on a run
 *  of presence-only observations (the stale-listing regression). */
export function sawTargetOffline(prev: boolean, targetListed: boolean): boolean {
  return prev || !targetListed;
}
