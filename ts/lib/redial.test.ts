/**
 * Auto-reconnect (session re-dial) tests.
 *
 * The properties that must hold:
 *   - a dead transport on an ESTABLISHED session triggers a re-dial,
 *     never during the initial startSession handshake;
 *   - a successful attempt emits `sessionReconnecting` then
 *     `sessionReconnected` and clears the redial state;
 *   - exhausted attempts emit a terminal `sessionStopped` with reason
 *     `reconnect_failed` (plus an `error`);
 *   - stopSession()/disconnect()/an external startSession() cancel a
 *     pending re-dial silently;
 *   - `autoReconnect: false` restores the legacy fatal-`error` path.
 *
 * The loop is exercised through the supervisor's surface
 * (`maybeBeginRedial`) with the public `startSession`/`connect` stubbed
 * out on the instance (the supervisor's deps are closures over the
 * instance, so stubs are picked up transparently): the dial itself
 * (real WebRTC/central handshake) is only exercised manually on
 * hardware - what matters here is the state machine around it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReachyMini } from './reachy-mini.js';
import type { SessionSupervisor } from './session-supervisor.js';

/** Private surface the tests poke at, spelled out for the casts. */
interface RedialInternals {
    _state: 'disconnected' | 'connected' | 'streaming';
    _selectedRobotId: string | null;
    _sessionId: string | null;
    _sessionResolve: (() => void) | null;
    _sessionReject: ((err: Error) => void) | null;
    _pc: { close: () => void } | null;
    _token: string | null;
    _supervisor: SessionSupervisor;
    _teardownForRedial(): void;
    _sendToServer(msg: Record<string, unknown>): Promise<unknown>;
    _startSessionInternal(robotId: string): Promise<void>;
    connect(token?: string): Promise<void>;
}

function makeStreamingInstance(
    options: { autoReconnect?: boolean } = {},
): { r: ReachyMini; internals: RedialInternals; dials: ReturnType<typeof vi.fn> } {
    const r = new ReachyMini(options);
    const internals = r as unknown as RedialInternals;
    // Established-session fixture: startSession promise settled, pc live.
    internals._state = 'streaming';
    internals._selectedRobotId = 'robot-1';
    internals._sessionId = 'session-1';
    internals._sessionResolve = null;
    internals._sessionReject = null;
    internals._pc = { close: vi.fn() };
    internals._token = 'hf_test';
    // No network in tests: endSession and friends resolve null.
    internals._sendToServer = vi.fn().mockResolvedValue(null);
    // The redial loop dials through the private path (the public
    // startSession cancels a pending redial, see below).
    const dials = vi.fn().mockResolvedValue(undefined);
    internals._startSessionInternal =
        dials as unknown as RedialInternals['_startSessionInternal'];
    return { r, internals, dials };
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

describe('auto-reconnect trigger gating', () => {
    it('does not trigger when autoReconnect is disabled', () => {
        const { internals } = makeStreamingInstance({ autoReconnect: false });
        expect(internals._supervisor.maybeBeginRedial('test')).toBe(false);
        expect(internals._supervisor.redialing).toBe(false);
    });

    it('does not trigger mid-setup (startSession promise still pending)', () => {
        const { internals } = makeStreamingInstance();
        internals._sessionReject = () => { /* pending dial */ };
        expect(internals._supervisor.maybeBeginRedial('test')).toBe(false);
    });

    it('does not trigger without a selected robot or pc', () => {
        const { internals } = makeStreamingInstance();
        internals._selectedRobotId = null;
        expect(internals._supervisor.maybeBeginRedial('test')).toBe(false);
        internals._selectedRobotId = 'robot-1';
        internals._pc = null;
        expect(internals._supervisor.maybeBeginRedial('test')).toBe(false);
    });

    it('reports already-in-progress as handled', () => {
        const { internals } = makeStreamingInstance();
        expect(internals._supervisor.maybeBeginRedial('first')).toBe(true);
        expect(internals._supervisor.maybeBeginRedial('second')).toBe(true);
        internals._supervisor.cancelRedial();
    });
});

describe('auto-reconnect success path', () => {
    it('re-dials the same robot and emits reconnecting → reconnected', async () => {
        const { r, internals, dials } = makeStreamingInstance();
        const reconnecting = events(r, 'sessionReconnecting');
        const reconnected = events(r, 'sessionReconnected');
        const stopped = events(r, 'sessionStopped');

        expect(internals._supervisor.maybeBeginRedial('ICE connection failed')).toBe(true);
        await vi.runAllTimersAsync();

        expect(dials).toHaveBeenCalledTimes(1);
        expect(dials).toHaveBeenCalledWith('robot-1');
        expect(reconnecting).toEqual([
            { attempt: 1, maxAttempts: 5, cause: 'ICE connection failed' },
        ]);
        expect(reconnected).toEqual([{ attempt: 1 }]);
        expect(stopped).toEqual([]);
        expect(internals._supervisor.redialing).toBe(false);
    });

    it('tears the dead transport down before dialing (pc closed, endSession sent)', async () => {
        const { internals } = makeStreamingInstance();
        const pc = internals._pc!;
        internals._supervisor.maybeBeginRedial('test');
        await vi.runAllTimersAsync();

        expect(pc.close).toHaveBeenCalled();
        expect(internals._sendToServer).toHaveBeenCalledWith({
            type: 'endSession',
            sessionId: 'session-1',
        });
    });

    it('recovers on a later attempt after transient failures', async () => {
        const { r, internals, dials } = makeStreamingInstance();
        dials
            .mockRejectedValueOnce(new Error('robot_busy_local'))
            .mockRejectedValueOnce(new Error('robot_busy_local'))
            .mockResolvedValueOnce(undefined);
        const reconnecting = events(r, 'sessionReconnecting');
        const reconnected = events(r, 'sessionReconnected');

        internals._supervisor.maybeBeginRedial('test');
        await vi.runAllTimersAsync();

        expect(dials).toHaveBeenCalledTimes(3);
        expect(reconnecting.map((d) => d.attempt)).toEqual([1, 2, 3]);
        expect(reconnected).toEqual([{ attempt: 3 }]);
    });
});

describe('auto-reconnect give-up path', () => {
    it('emits a terminal sessionStopped(reconnect_failed) after 5 failed attempts', async () => {
        const { r, internals, dials } = makeStreamingInstance();
        dials.mockRejectedValue(new Error('still dead'));
        const stopped = events(r, 'sessionStopped');
        const errors = events(r, 'error');

        internals._supervisor.maybeBeginRedial('ICE connection failed');
        await vi.runAllTimersAsync();

        expect(dials).toHaveBeenCalledTimes(5);
        expect(stopped).toHaveLength(1);
        expect(stopped[0]!.reason).toBe('reconnect_failed');
        expect(errors).toHaveLength(1);
        expect(internals._supervisor.redialing).toBe(false);
    });
});

describe('auto-reconnect cancellation', () => {
    it('stopSession() aborts the loop with no further events', async () => {
        const { r, internals, dials } = makeStreamingInstance();
        dials.mockRejectedValue(new Error('still dead'));
        const stopped = events(r, 'sessionStopped');

        internals._supervisor.maybeBeginRedial('test');
        // Let attempt 1 fail, then cancel during the 2 s backoff.
        await vi.advanceTimersByTimeAsync(100);
        await r.stopSession();
        await vi.runAllTimersAsync();

        expect(dials).toHaveBeenCalledTimes(1);
        // No reconnect_failed: the stop was deliberate.
        expect(stopped.filter((d) => d.reason === 'reconnect_failed')).toEqual([]);
        expect(internals._supervisor.redialing).toBe(false);
    });

    it('setAutoReconnect(false) aborts an in-flight re-dial and blocks new ones', async () => {
        const { r, internals, dials } = makeStreamingInstance();
        dials.mockRejectedValue(new Error('still dead'));
        const stopped = events(r, 'sessionStopped');

        internals._supervisor.maybeBeginRedial('test');
        await vi.advanceTimersByTimeAsync(100);
        r.setAutoReconnect(false);
        await vi.runAllTimersAsync();

        expect(dials).toHaveBeenCalledTimes(1);
        expect(stopped).toEqual([]);
        expect(internals._supervisor.redialing).toBe(false);
        // And no new redial can start while disabled.
        internals._pc = { close: vi.fn() };
        expect(internals._supervisor.maybeBeginRedial('again')).toBe(false);
    });

    it('teardown settles a pending dial promise instead of orphaning it', () => {
        const { internals } = makeStreamingInstance();
        // Simulate a timed-out attempt: startSession left its resolvers armed.
        const reject = vi.fn();
        internals._sessionResolve = vi.fn();
        internals._sessionReject = reject;

        internals._teardownForRedial();

        expect(reject).toHaveBeenCalledTimes(1);
        expect(internals._sessionResolve).toBeNull();
        expect(internals._sessionReject).toBeNull();
    });

    it('teardown without a token skips the fire-and-forget endSession', () => {
        const { internals } = makeStreamingInstance();
        internals._token = null;

        internals._teardownForRedial();

        expect(internals._sendToServer).not.toHaveBeenCalled();
        expect(internals._sessionId).toBeNull();
    });

    it('an external startSession() supersedes a pending re-dial', async () => {
        const { r, internals, dials } = makeStreamingInstance();
        dials.mockRejectedValue(new Error('still dead'));

        internals._supervisor.maybeBeginRedial('test');
        await vi.advanceTimersByTimeAsync(100);
        // The app dials another robot mid-backoff: the public
        // startSession cancels the pending re-dial, then dials through
        // the internal path itself.
        dials.mockResolvedValueOnce(undefined);
        await r.startSession('robot-2');
        await vi.runAllTimersAsync();

        // Attempt 1 of the loop, then the external dial — nothing after.
        expect(dials).toHaveBeenCalledTimes(2);
        expect(dials).toHaveBeenLastCalledWith('robot-2');
        expect(internals._supervisor.redialing).toBe(false);
    });
});
