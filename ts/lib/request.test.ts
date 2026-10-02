/**
 * request() - the generic command round-trip escape hatch.
 *
 * The properties that must hold:
 *   - `request({type})` sends the command as-is and resolves with the
 *     first robot message whose `command` field echoes that type (the
 *     daemon's reply convention), so a new daemon command is usable
 *     without an SDK release;
 *   - unrelated replies don't settle the call, and a custom `match`
 *     predicate overrides the default echo matcher;
 *   - a daemon that never answers (predates the command) fail-opens to
 *     `null` on the shared round-trip timeout instead of hanging;
 *   - replies the SDK consumes internally (e.g. `get_imu`) are swallowed
 *     by their own handlers and never leak into a `request()` waiter;
 *   - a closed data channel rejects immediately.
 *
 * Same approach as the other SDK suites: private surface + a fake data
 * channel, fake timers for the fail-open path.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReachyMini } from './reachy-mini.js';

/** Private surface the tests poke at, spelled out for the casts. */
interface RequestInternals {
    _dc: { readyState: string; send: (msg: string) => void } | null;
    _handleRobotMessage(data: Record<string, unknown>): void;
}

function makeInstance(): {
    r: ReachyMini;
    internals: RequestInternals;
    sent: Array<Record<string, unknown>>;
} {
    const r = new ReachyMini();
    const internals = r as unknown as RequestInternals;
    const sent: Array<Record<string, unknown>> = [];
    internals._dc = {
        readyState: 'open',
        send: (msg: string) => { sent.push(JSON.parse(msg) as Record<string, unknown>); },
    };
    return { r, internals, sent };
}

beforeEach(() => {
    vi.useFakeTimers();
});

afterEach(() => {
    vi.useRealTimers();
});

describe('request - wire contract', () => {
    it('sends the command and resolves the echoed reply untouched', async () => {
        const { r, internals, sent } = makeInstance();

        const promise = r.request({ type: 'get_battery', detail: true });
        expect(sent).toEqual([{ type: 'get_battery', detail: true }]);

        internals._handleRobotMessage({ command: 'get_battery', level: 0.87 });
        await expect(promise).resolves.toEqual({ command: 'get_battery', level: 0.87 });
    });

    it('ignores unrelated replies while waiting', async () => {
        const { r, internals } = makeInstance();
        const settled = vi.fn();

        void r.request({ type: 'get_battery' }).then(settled);
        internals._handleRobotMessage({ command: 'get_something_else', value: 1 });
        await vi.advanceTimersByTimeAsync(0);

        expect(settled).not.toHaveBeenCalled();
    });

    it('honors a custom match predicate for non-echo replies', async () => {
        const { r, internals } = makeInstance();

        const promise = r.request(
            { type: 'get_battery' },
            { match: (m) => 'battery_level' in m },
        );
        internals._handleRobotMessage({ battery_level: 0.42 });

        await expect(promise).resolves.toEqual({ battery_level: 0.42 });
    });

    it('does not steal replies owned by the SDK\'s typed handlers', async () => {
        const { r, internals } = makeInstance();
        const settled = vi.fn();

        // get_imu replies are consumed by the imu reply slot before the
        // broadcast matcher runs, even with no getImu() call pending.
        void r.request({ type: 'get_imu' }).then(settled);
        internals._handleRobotMessage({ command: 'get_imu', imu: null });
        await vi.advanceTimersByTimeAsync(0);

        expect(settled).not.toHaveBeenCalled();
    });
});

describe('request - degraded paths', () => {
    it('fail-opens to null when the daemon never answers (old daemon)', async () => {
        const { r } = makeInstance();

        const promise = r.request({ type: 'get_battery' });
        await vi.advanceTimersByTimeAsync(4000);

        await expect(promise).resolves.toBeNull();
    });

    it('respects a caller-provided timeout', async () => {
        const { r } = makeInstance();

        const promise = r.request({ type: 'get_battery' }, { timeoutMs: 500 });
        await vi.advanceTimersByTimeAsync(500);

        await expect(promise).resolves.toBeNull();
    });

    it('rejects immediately when the data channel is not open', async () => {
        const { r, internals } = makeInstance();
        internals._dc = null;

        await expect(r.request({ type: 'get_battery' })).rejects.toThrow(/Data channel not open/);
    });
});
