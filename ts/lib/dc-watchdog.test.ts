/**
 * Data-channel silence watchdog tests.
 *
 * The properties that must hold:
 *   - inbound traffic keeps the watchdog quiet (no nudge, no escalation);
 *   - past DC_SILENCE_NUDGE_MS with no traffic, exactly one extra
 *     `get_state` nudge goes out; a reply resets the cycle;
 *   - past DC_SILENCE_FATAL_MS of TOTAL silence the transport is dead:
 *     hand over to the auto re-dial when enabled, else emit the fatal
 *     `error`;
 *   - a large gap between ticks (throttled / suspended timer) re-baselines
 *     instead of judging stale data;
 *   - stopSession() stops the watchdog.
 *
 * Same testing approach as redial.test.ts: private surface + stubbed
 * network, fake timers (which also fake Date.now, so silence accrues
 * exactly as fast as the advanced clock).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReachyMini } from './reachy-mini.js';
import type { SessionSupervisor } from './session-supervisor.js';

/** The supervisor's private clock state the throttle test rewinds. */
interface SupervisorClockPoke {
    _dcWatchdogLastTickAt: number;
    _lastDcInboundAt: number;
}

/** Private surface the tests poke at, spelled out for the casts. */
interface WatchdogInternals {
    _state: 'disconnected' | 'connected' | 'streaming';
    _selectedRobotId: string | null;
    _sessionId: string | null;
    _sessionResolve: (() => void) | null;
    _sessionReject: ((err: Error) => void) | null;
    _pc: { close: () => void } | null;
    _token: string | null;
    _supervisor: SessionSupervisor;
    _sendToServer(msg: Record<string, unknown>): Promise<unknown>;
    _handleRobotMessage(data: Record<string, unknown>): void;
    requestState(): boolean;
    _startSessionInternal(robotId: string): Promise<void>;
}

function makeStreamingInstance(
    options: { autoReconnect?: boolean } = {},
): {
    r: ReachyMini;
    internals: WatchdogInternals;
    nudges: ReturnType<typeof vi.fn>;
    dials: ReturnType<typeof vi.fn>;
} {
    const r = new ReachyMini(options);
    const internals = r as unknown as WatchdogInternals;
    internals._state = 'streaming';
    internals._selectedRobotId = 'robot-1';
    internals._sessionId = 'session-1';
    internals._sessionResolve = null;
    internals._sessionReject = null;
    internals._pc = { close: vi.fn() };
    internals._token = 'hf_test';
    internals._sendToServer = vi.fn().mockResolvedValue(null);
    const nudges = vi.fn().mockReturnValue(true);
    internals.requestState = nudges as unknown as WatchdogInternals['requestState'];
    const dials = vi.fn().mockResolvedValue(undefined);
    internals._startSessionInternal =
        dials as unknown as WatchdogInternals['_startSessionInternal'];
    return { r, internals, nudges, dials };
}

function events(r: ReachyMini, name: string): Array<Record<string, unknown>> {
    const seen: Array<Record<string, unknown>> = [];
    r.addEventListener(name, (e) => {
        seen.push((e as CustomEvent<Record<string, unknown>>).detail ?? {});
    });
    return seen;
}

beforeEach(() => {
    vi.useFakeTimers();
});

afterEach(() => {
    vi.useRealTimers();
});

describe('dc silence watchdog - quiet on a healthy link', () => {
    it('never nudges nor escalates while traffic flows', async () => {
        const { r, internals, nudges } = makeStreamingInstance();
        const errors = events(r, 'error');
        internals._supervisor.startDcWatchdog();

        // 12 s of clock with a message every 500 ms, like the real poll.
        for (let i = 0; i < 24; i++) {
            await vi.advanceTimersByTimeAsync(500);
            internals._handleRobotMessage({});
        }

        expect(nudges).not.toHaveBeenCalled();
        expect(errors).toEqual([]);
        expect(internals._supervisor.redialing).toBe(false);
        internals._supervisor.stopDcWatchdog();
    });

    it('a reply to the nudge resets the cycle without escalating', async () => {
        const { r, internals, nudges } = makeStreamingInstance();
        const errors = events(r, 'error');
        internals._supervisor.startDcWatchdog();

        // Cross the nudge threshold in silence…
        await vi.advanceTimersByTimeAsync(3000);
        expect(nudges).toHaveBeenCalledTimes(1);
        // …then the daemon answers.
        internals._handleRobotMessage({});
        // Well past the original fatal deadline: no escalation, and the
        // next silent window starts a fresh nudge cycle.
        await vi.advanceTimersByTimeAsync(3000);
        expect(nudges).toHaveBeenCalledTimes(2);
        expect(errors).toEqual([]);
        expect(internals._supervisor.redialing).toBe(false);
        internals._supervisor.stopDcWatchdog();
    });
});

describe('dc silence watchdog - dead transport escalation', () => {
    it('hands a silent transport over to the auto re-dial', async () => {
        const { r, internals, nudges, dials } = makeStreamingInstance();
        const reconnecting = events(r, 'sessionReconnecting');
        const reconnected = events(r, 'sessionReconnected');
        internals._supervisor.startDcWatchdog();

        // Total silence past the fatal threshold, then let the re-dial
        // loop run to completion (stubbed dial succeeds immediately).
        await vi.advanceTimersByTimeAsync(20_000);

        expect(nudges).toHaveBeenCalledTimes(1);
        expect(dials).toHaveBeenCalledWith('robot-1');
        expect(reconnecting).toHaveLength(1);
        expect(String(reconnecting[0]!.cause)).toMatch(/No data-channel traffic/);
        expect(reconnected).toEqual([{ attempt: 1 }]);
    });

    it('falls back to the fatal error when autoReconnect is off', async () => {
        const { r, internals } = makeStreamingInstance({ autoReconnect: false });
        const errors = events(r, 'error');
        internals._supervisor.startDcWatchdog();

        await vi.advanceTimersByTimeAsync(10_000);

        expect(errors).toHaveLength(1);
        expect(String((errors[0]!.error as Error).message)).toMatch(
            /No data-channel traffic/,
        );
        expect(internals._supervisor.redialing).toBe(false);
        // The fatal path stands the watchdog down: even a full extra
        // escalation cycle (nudge at 2.5 s, fatal at 8 s) later, the
        // error fired exactly once. Regression guard for the loop where
        // the interval kept running and re-emitted every ~8 s forever.
        await vi.advanceTimersByTimeAsync(10_000);
        expect(errors).toHaveLength(1);
        internals._supervisor.stopDcWatchdog();
    });
});

describe('dc silence watchdog - throttled-timer awareness', () => {
    it('re-baselines instead of judging after a large tick gap', async () => {
        const { r, internals, nudges } = makeStreamingInstance();
        const errors = events(r, 'error');
        internals._supervisor.startDcWatchdog();

        // Simulate a suspended tab: the interval did not fire for 30 s
        // and no message was stamped either.
        const clock = internals._supervisor as unknown as SupervisorClockPoke;
        clock._dcWatchdogLastTickAt -= 30_000;
        clock._lastDcInboundAt -= 30_000;
        await vi.advanceTimersByTimeAsync(1000);

        // The gap tick re-baselined: no nudge, no error, and the silence
        // clock restarts from the resume point.
        expect(nudges).not.toHaveBeenCalled();
        expect(errors).toEqual([]);
        expect(internals._supervisor.redialing).toBe(false);
        internals._supervisor.stopDcWatchdog();
    });
});

describe('dc silence watchdog - lifecycle', () => {
    it('stopSession() stops the watchdog', async () => {
        const { r, internals, nudges } = makeStreamingInstance();
        internals._supervisor.startDcWatchdog();

        await r.stopSession();
        await vi.advanceTimersByTimeAsync(20_000);

        expect(nudges).not.toHaveBeenCalled();
        expect(internals._supervisor.redialing).toBe(false);
    });

    it('ignores silence when the session is not streaming', async () => {
        const { internals, nudges } = makeStreamingInstance();
        internals._supervisor.startDcWatchdog();
        internals._state = 'connected';

        await vi.advanceTimersByTimeAsync(20_000);

        expect(nudges).not.toHaveBeenCalled();
        internals._supervisor.stopDcWatchdog();
    });
});
