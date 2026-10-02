/**
 * ensureAwake() bring-up contract.
 *
 * The properties that must hold:
 *   - asleep (`motor_mode: 'disabled'`): plays `wake_up` and resolves only
 *     once the daemon reports the trajectory completed, so a caller can
 *     start commanding poses on resolution without fighting the emote;
 *   - a wake the daemon never confirms resolves anyway after the internal
 *     trajectory budget (degraded boot beats a boot trapped on the splash);
 *   - `gravity_compensation` inherited from a previous session: flip back
 *     to `enabled` (position control) WITHOUT replaying the wake emote,
 *     and nudge a state refresh so `isAwake()` readers converge;
 *   - already `enabled`: strict no-op, nothing on the wire;
 *   - unknown motor mode (fresh session): ask for a state snapshot first
 *     and fall through to the right branch once it lands.
 *
 * Same approach as the other SDK suites: private surface + a fake data
 * channel, fake timers for the budget paths.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReachyMini } from './reachy-mini.js';

/** Private surface the tests poke at, spelled out for the casts. */
interface EnsureAwakeInternals {
    _dc: { readyState: string; send: (msg: string) => void } | null;
    _handleRobotMessage(data: Record<string, unknown>): void;
}

function makeInstance(): {
    r: ReachyMini;
    internals: EnsureAwakeInternals;
    sent: Array<Record<string, unknown>>;
} {
    const r = new ReachyMini();
    const internals = r as unknown as EnsureAwakeInternals;
    const sent: Array<Record<string, unknown>> = [];
    internals._dc = {
        readyState: 'open',
        send: (msg: string) => { sent.push(JSON.parse(msg) as Record<string, unknown>); },
    };
    return { r, internals, sent };
}

/** Feed a state frame carrying the given motor mode into the mirror. */
function pushMotorMode(
    internals: EnsureAwakeInternals,
    mode: 'enabled' | 'disabled' | 'gravity_compensation',
): void {
    internals._handleRobotMessage({ state: { motor_mode: mode } });
}

beforeEach(() => {
    vi.useFakeTimers();
});

afterEach(() => {
    vi.useRealTimers();
});

describe('ensureAwake - asleep robot', () => {
    it('plays wake_up and resolves only on trajectory completion', async () => {
        const { r, internals, sent } = makeInstance();
        pushMotorMode(internals, 'disabled');
        const settled = vi.fn();

        const promise = r.ensureAwake().then(settled);
        await vi.advanceTimersByTimeAsync(0);

        // wakeUp() = enable motors + the wake_up command itself.
        expect(sent).toEqual([
            { type: 'set_motor_mode', mode: 'enabled' },
            { type: 'wake_up' },
        ]);
        // Command acked but trajectory not completed yet: must still be pending.
        expect(settled).not.toHaveBeenCalled();

        internals._handleRobotMessage({ command: 'wake_up', completed: true });
        await expect(promise).resolves.toBeUndefined();
        expect(settled).toHaveBeenCalled();
    });

    it('resolves anyway when the daemon never confirms (internal budget)', async () => {
        const { r, internals } = makeInstance();
        pushMotorMode(internals, 'disabled');

        const promise = r.ensureAwake();
        // Past the 5 s trajectory budget with no completion in sight.
        await vi.advanceTimersByTimeAsync(6000);

        await expect(promise).resolves.toBe(true);
    });
});

describe('ensureAwake - gravity compensation inherited from a fast handoff', () => {
    it('flips back to enabled without replaying the wake emote', async () => {
        const { r, internals, sent } = makeInstance();
        pushMotorMode(internals, 'gravity_compensation');

        await expect(r.ensureAwake()).resolves.toBe(true);

        const types = sent.map((m) => m.type ?? m.command);
        expect(types).toContain('set_motor_mode');
        expect(types).not.toContain('wake_up');
        expect(sent.find((m) => m.type === 'set_motor_mode')).toEqual({
            type: 'set_motor_mode',
            mode: 'enabled',
        });
        // Cache refresh so isAwake() readers converge on the new mode.
        expect(types).toContain('get_state');
    });
});

describe('ensureAwake - already under position control', () => {
    it('is a strict no-op', async () => {
        const { r, internals, sent } = makeInstance();
        pushMotorMode(internals, 'enabled');

        await expect(r.ensureAwake()).resolves.toBe(true);
        expect(sent).toEqual([]);
    });
});

describe('ensureAwake - unknown motor mode (fresh session)', () => {
    it('requests a snapshot, then wakes when the state says disabled', async () => {
        const { r, internals, sent } = makeInstance();
        const settled = vi.fn();

        const promise = r.ensureAwake().then(settled);
        await vi.advanceTimersByTimeAsync(0);
        // First move: ask for the state it doesn't have yet.
        expect(sent[0]).toEqual({ type: 'get_state' });

        pushMotorMode(internals, 'disabled');
        await vi.advanceTimersByTimeAsync(0);
        expect(sent.map((m) => m.type)).toContain('wake_up');
        expect(settled).not.toHaveBeenCalled();

        internals._handleRobotMessage({ command: 'wake_up', completed: true });
        await promise;
        expect(settled).toHaveBeenCalled();
    });

    it('falls through on the snapshot timeout and still resolves', async () => {
        const { r } = makeInstance();

        const promise = r.ensureAwake(1000);
        // Snapshot never answered (1 s) + wake budget never confirmed (5 s).
        await vi.advanceTimersByTimeAsync(6100);

        await expect(promise).resolves.toBe(true);
    });
});
