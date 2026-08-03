/**
 * getImu() wire contract.
 *
 * The properties that must hold:
 *   - `getImu()` sends `{type: "get_imu"}` and resolves the daemon's
 *     `imu` payload as-is (accelerometer / gyroscope / quaternion /
 *     temperature);
 *   - an IMU-less robot (Lite, simulation) answers `imu: null` and the
 *     call resolves `null` promptly - distinguishable from a timeout;
 *   - an old daemon that doesn't know `get_imu` never answers: the
 *     shared slot timeout fail-opens to `null` instead of hanging;
 *   - a closed data channel rejects immediately (no waiter leaked).
 *
 * Same approach as the other SDK suites: private surface + a fake data
 * channel, fake timers for the fail-open path.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReachyMini } from './reachy-mini.js';
import type { ImuData } from './types.js';

/** Private surface the tests poke at, spelled out for the casts. */
interface ImuInternals {
    _dc: { readyState: string; send: (msg: string) => void } | null;
    _handleRobotMessage(data: Record<string, unknown>): void;
}

function makeInstance(): {
    r: ReachyMini;
    internals: ImuInternals;
    sent: Array<Record<string, unknown>>;
} {
    const r = new ReachyMini();
    const internals = r as unknown as ImuInternals;
    const sent: Array<Record<string, unknown>> = [];
    internals._dc = {
        readyState: 'open',
        send: (msg: string) => { sent.push(JSON.parse(msg) as Record<string, unknown>); },
    };
    return { r, internals, sent };
}

const READING: ImuData = {
    accelerometer: [0.01, -0.02, 9.81],
    gyroscope: [0.001, 0.002, -0.003],
    quaternion: [1, 0, 0, 0],
    temperature: 31.5,
};

beforeEach(() => {
    vi.useFakeTimers();
});

afterEach(() => {
    vi.useRealTimers();
});

describe('getImu - wire contract', () => {
    it('sends get_imu and resolves the daemon reading untouched', async () => {
        const { r, internals, sent } = makeInstance();

        const promise = r.getImu();
        expect(sent).toEqual([{ type: 'get_imu' }]);

        internals._handleRobotMessage({ command: 'get_imu', imu: READING });
        await expect(promise).resolves.toEqual(READING);
    });

    it('resolves null promptly on an IMU-less robot (imu: null reply)', async () => {
        const { r, internals } = makeInstance();

        const promise = r.getImu();
        internals._handleRobotMessage({ command: 'get_imu', imu: null });

        // No timer advance: this must be the reply, not the timeout.
        await expect(promise).resolves.toBeNull();
    });

    it('an unrelated message does not settle the pending call', async () => {
        const { r, internals } = makeInstance();
        const settled = vi.fn();

        void r.getImu().then(settled);
        internals._handleRobotMessage({ command: 'get_tracked_face', face_target: { detected: false } });
        await vi.advanceTimersByTimeAsync(0);

        expect(settled).not.toHaveBeenCalled();
    });
});

describe('getImu - degraded paths', () => {
    it('fail-opens to null when the daemon never answers (old daemon)', async () => {
        const { r } = makeInstance();

        const promise = r.getImu();
        await vi.advanceTimersByTimeAsync(4000);

        await expect(promise).resolves.toBeNull();
    });

    it('rejects immediately when the data channel is not open', async () => {
        const { r, internals } = makeInstance();
        internals._dc = null;

        await expect(r.getImu()).rejects.toThrow(/Data channel not open/);
    });
});
