/**
 * Reboot-watch tests: the identity-matching tiers of `isTargetListed`
 * and the sticky offline latch of `sawTargetOffline`.
 *
 * The regression these guard: central kept the robot's PRE-reboot
 * registration listed for a few seconds after the daemon restarted, and
 * the old presence-only check read it as "back online", completing the
 * update gate while the robot was still down. The watch must therefore
 * NEVER complete without having first observed the target absent.
 */
import { describe, expect, it } from 'vitest';

import {
  isTargetListed,
  sawTargetOffline,
  type RebootTarget,
} from './rebootWatch';
import type { RobotInfo } from './sdk-types';

function robot(overrides: Partial<RobotInfo> & { id: string }): RobotInfo {
  return { busy: false, ...overrides };
}

const target = (
  hardwareId: string | null,
  name: string | null,
): RebootTarget => ({ hardwareId, name });

describe('isTargetListed', () => {
  it('matches on hardware id regardless of peer id or name', () => {
    const robots = [
      robot({ id: 'peer-new', hardwareId: 'RM-42', meta: { name: 'other' } }),
    ];
    expect(isTargetListed(robots, target('RM-42', 'reachy'))).toBe(true);
    expect(isTargetListed(robots, target('RM-99', 'other'))).toBe(false);
  });

  it('does NOT fall back to name when a hardware id is expected', () => {
    // A different robot with the same advertised name must not count
    // as the one we're waiting for.
    const robots = [robot({ id: 'p1', meta: { name: 'reachy' } })];
    expect(isTargetListed(robots, target('RM-42', 'reachy'))).toBe(false);
  });

  it('matches on name when no hardware id is known', () => {
    const robots = [robot({ id: 'p1', meta: { name: 'reachy' } })];
    expect(isTargetListed(robots, target(null, 'reachy'))).toBe(true);
    expect(isTargetListed(robots, target(null, 'nope'))).toBe(false);
  });

  it('with no identity at all, reads "any robot listed"', () => {
    expect(isTargetListed([], target(null, null))).toBe(false);
    expect(isTargetListed([robot({ id: 'p1' })], target(null, null))).toBe(
      true,
    );
  });
});

describe('sawTargetOffline', () => {
  it('latches once the target disappears, and stays latched', () => {
    expect(sawTargetOffline(false, false)).toBe(true);
    expect(sawTargetOffline(true, true)).toBe(true);
  });

  it('never completes on a run of presence-only observations', () => {
    // The exact stale-listing regression: listing present the whole
    // time (old registration never observed absent) must not complete
    // the watch (`sawOffline && listed`).
    let sawOffline = false;
    for (let i = 0; i < 10; i += 1) {
      sawOffline = sawTargetOffline(sawOffline, true);
    }
    expect(sawOffline && true).toBe(false);
  });

  it('completes only after absent-then-present', () => {
    let sawOffline = false;
    sawOffline = sawTargetOffline(sawOffline, true); // stale listing
    expect(sawOffline).toBe(false);
    sawOffline = sawTargetOffline(sawOffline, false); // dropped
    sawOffline = sawTargetOffline(sawOffline, true); // re-registered
    expect(sawOffline).toBe(true);
  });
});
