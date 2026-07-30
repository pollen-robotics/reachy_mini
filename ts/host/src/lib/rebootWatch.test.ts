/**
 * Reboot-watch tests: the identity-matching tiers of `isTargetListed`
 * and the offline-first latch of `advanceRebootWatch`.
 *
 * The regression these guard: central kept the robot's PRE-reboot
 * registration listed for a few seconds after the daemon restarted, and
 * the old presence-only check read it as "back online", completing the
 * update gate while the robot was still down. The latch must therefore
 * NEVER reach `back` without having first observed the target absent.
 */
import { describe, expect, it } from 'vitest';

import {
  advanceRebootWatch,
  isTargetListed,
  type RebootTarget,
  type RebootWatchPhase,
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

describe('advanceRebootWatch', () => {
  it('holds waiting-offline while the (stale) listing persists', () => {
    expect(advanceRebootWatch('waiting-offline', true)).toBe(
      'waiting-offline',
    );
  });

  it('latches to waiting-online once the target disappears', () => {
    expect(advanceRebootWatch('waiting-offline', false)).toBe(
      'waiting-online',
    );
    expect(advanceRebootWatch('waiting-online', false)).toBe('waiting-online');
  });

  it('completes only after absent-then-present', () => {
    expect(advanceRebootWatch('waiting-online', true)).toBe('back');
  });

  it('is terminal in back', () => {
    expect(advanceRebootWatch('back', true)).toBe('back');
    expect(advanceRebootWatch('back', false)).toBe('back');
  });

  it('never reaches back from a run of presence-only observations', () => {
    // The exact stale-listing regression: listing present the whole
    // time (old registration never observed absent) must not complete.
    let phase: RebootWatchPhase = 'waiting-offline';
    for (let i = 0; i < 10; i += 1) {
      phase = advanceRebootWatch(phase, true);
    }
    expect(phase).toBe('waiting-offline');
  });
});
