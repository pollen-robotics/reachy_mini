/**
 * ICE-blip grace tests, run against an isolated SessionSupervisor with
 * fully faked deps (no ReachyMini instance involved).
 *
 * The properties that must hold:
 *   - `disconnected` is debounced ~3 s and escalates only if ICE is
 *     still bad when the grace expires; healing (event or re-check)
 *     stands the timer down;
 *   - `failed` gets a shorter ~1 s debounce;
 *   - repeated identical transitions coalesce onto the original timer
 *     (no clock reset); a changed reason replaces the timer;
 *   - mid-setup failures reject the pending startSession() promise and
 *     never start a redial;
 *   - while the tab is hidden the grace defers to the next foreground
 *     frame; past the visibility ceiling the session is expired
 *     immediately (straight to redial when enabled);
 *   - escalation funnels into the redial when auto-reconnect is on and
 *     falls back to the fatal `error` when it's off;
 *   - network listeners forward browser events and uninstall cleanly.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { SessionSupervisor } from './session-supervisor.js';
import type { SessionSupervisorDeps } from './session-supervisor.js';

type MockedDeps = { [K in keyof SessionSupervisorDeps]: ReturnType<typeof vi.fn> };

function makeSupervisor(
    overrides: Partial<MockedDeps> = {},
    options: { autoReconnect: boolean } = { autoReconnect: true },
): { sup: SessionSupervisor; deps: MockedDeps } {
    const deps: MockedDeps = {
        iceState: vi.fn().mockReturnValue('disconnected'),
        hasPc: vi.fn().mockReturnValue(true),
        isStreaming: vi.fn().mockReturnValue(true),
        isMidSetup: vi.fn().mockReturnValue(false),
        isSignalingDown: vi.fn().mockReturnValue(false),
        selectedRobotId: vi.fn().mockReturnValue('robot-1'),
        rejectPendingSession: vi.fn().mockReturnValue(false),
        reconnectSignaling: vi.fn().mockResolvedValue(undefined),
        dial: vi.fn().mockResolvedValue(undefined),
        teardownForRedial: vi.fn(),
        nudgeState: vi.fn(),
        emit: vi.fn(),
        ...overrides,
    };
    return { sup: new SessionSupervisor(deps as unknown as SessionSupervisorDeps, options), deps };
}

/** Details of every `emit(name, …)` call, in order. */
function emitted(deps: MockedDeps, name: string): unknown[] {
    return deps.emit.mock.calls.filter((c) => c[0] === name).map((c) => c[1]);
}

/** The `error` emits, unwrapped to their Error payloads. */
function emittedErrors(deps: MockedDeps): Error[] {
    return (emitted(deps, 'error') as Array<{ error: Error }>).map((d) => d.error);
}

/** jsdom's document.hidden is read-only; swap in a controllable getter. */
let hidden = false;
beforeEach(() => {
    vi.useFakeTimers();
    hidden = false;
    Object.defineProperty(document, 'hidden', {
        configurable: true,
        get: () => hidden,
    });
});

afterEach(() => {
    vi.useRealTimers();
});

describe('disconnected grace (foreground)', () => {
    it('escalates only after the grace expires with ICE still down', async () => {
        const { sup, deps } = makeSupervisor({}, { autoReconnect: false });
        sup.onIceDisconnected();

        await vi.advanceTimersByTimeAsync(2999);
        expect(emittedErrors(deps)).toHaveLength(0);

        await vi.advanceTimersByTimeAsync(1);
        expect(emittedErrors(deps)).toHaveLength(1);
        expect(String(emittedErrors(deps)[0])).toMatch(/ICE stuck in 'disconnected'/);
    });

    it('funnels into the redial instead of the error when auto-reconnect is on', async () => {
        const { sup, deps } = makeSupervisor();
        sup.onIceDisconnected();
        await vi.runAllTimersAsync();

        expect(emittedErrors(deps)).toHaveLength(0);
        expect(deps.teardownForRedial).toHaveBeenCalled();
        expect(deps.dial).toHaveBeenCalledWith('robot-1');
        expect(emitted(deps, 'sessionReconnected')).toEqual([{ attempt: 1 }]);
    });

    it('stands down when ICE heals before the grace expires', async () => {
        const { sup, deps } = makeSupervisor({}, { autoReconnect: false });
        sup.onIceDisconnected();
        await vi.advanceTimersByTimeAsync(1000);
        sup.onIceHealed();
        await vi.runAllTimersAsync();
        expect(emittedErrors(deps)).toHaveLength(0);
    });

    it('re-checks the live state at expiry (healed without an event)', async () => {
        const { sup, deps } = makeSupervisor(
            { iceState: vi.fn().mockReturnValue('connected') },
            { autoReconnect: false },
        );
        sup.onIceDisconnected();
        await vi.runAllTimersAsync();
        expect(emittedErrors(deps)).toHaveLength(0);
    });

    it('coalesces repeated identical transitions onto the original clock', async () => {
        const { sup, deps } = makeSupervisor({}, { autoReconnect: false });
        sup.onIceDisconnected();
        await vi.advanceTimersByTimeAsync(2000);
        // A flurry of identical events must not reset the 3 s clock…
        sup.onIceDisconnected();
        await vi.advanceTimersByTimeAsync(1000);
        // …so the grace fires 3 s after the FIRST transition.
        expect(emittedErrors(deps)).toHaveLength(1);
    });
});

describe('failed grace', () => {
    it('escalates after ~1 s when ICE stays failed', async () => {
        const { sup, deps } = makeSupervisor(
            { iceState: vi.fn().mockReturnValue('failed') },
            { autoReconnect: false },
        );
        sup.onIceFailed();
        await vi.advanceTimersByTimeAsync(1000);
        expect(emittedErrors(deps)).toHaveLength(1);
        expect(String(emittedErrors(deps)[0])).toMatch(/ICE connection failed/);
    });

    it('a disconnected → failed transition replaces the pending grace', async () => {
        const { sup, deps } = makeSupervisor(
            { iceState: vi.fn().mockReturnValue('failed') },
            { autoReconnect: false },
        );
        sup.onIceDisconnected();
        sup.onIceFailed();
        // The failed grace (1 s) wins over the disconnected one (3 s).
        await vi.advanceTimersByTimeAsync(1000);
        expect(emittedErrors(deps)).toHaveLength(1);
        expect(String(emittedErrors(deps)[0])).toMatch(/ICE connection failed/);
    });

    it('rejects a pending startSession() and leaves the retry to its caller', async () => {
        const rejectPendingSession = vi.fn().mockReturnValue(true);
        const { sup, deps } = makeSupervisor({
            iceState: vi.fn().mockReturnValue('failed'),
            rejectPendingSession,
        });
        sup.onIceFailed();
        await vi.advanceTimersByTimeAsync(1000);

        expect(rejectPendingSession).toHaveBeenCalledTimes(1);
        // Not an auto-reconnect case: the promise owner decides.
        expect(deps.dial).not.toHaveBeenCalled();
        // Outside a redial the classic fatal error still surfaces.
        expect(emittedErrors(deps)).toHaveLength(1);
    });
});

describe('visibility-deferred grace (hidden tab)', () => {
    it('defers the grace to the foreground, then escalates if still down', async () => {
        const { sup, deps } = makeSupervisor({}, { autoReconnect: false });
        hidden = true;
        sup.onIceDisconnected();

        // Hidden: timers may be throttled, nothing may fire meanwhile.
        await vi.advanceTimersByTimeAsync(10_000);
        expect(emittedErrors(deps)).toHaveLength(0);

        // Back to the foreground within the ceiling: a normal 3 s grace runs.
        hidden = false;
        document.dispatchEvent(new Event('visibilitychange'));
        expect(emittedErrors(deps)).toHaveLength(0);
        await vi.advanceTimersByTimeAsync(3000);
        expect(emittedErrors(deps)).toHaveLength(1);
        expect(String(emittedErrors(deps)[0])).toMatch(/ICE stuck in 'disconnected'/);
    });

    it('does nothing when ICE healed while backgrounded', async () => {
        const { sup, deps } = makeSupervisor({}, { autoReconnect: false });
        hidden = true;
        sup.onIceDisconnected();
        deps.iceState.mockReturnValue('connected');
        hidden = false;
        document.dispatchEvent(new Event('visibilitychange'));
        await vi.runAllTimersAsync();
        expect(emittedErrors(deps)).toHaveLength(0);
    });

    it('past the ceiling, expires the session immediately (straight to redial)', async () => {
        const { sup, deps } = makeSupervisor();
        hidden = true;
        sup.onIceDisconnected();

        // Fake timers also fake Date.now: cross the 60 s ceiling.
        await vi.advanceTimersByTimeAsync(61_000);
        hidden = false;
        document.dispatchEvent(new Event('visibilitychange'));
        await vi.runAllTimersAsync();

        // No extra foreground grace — the redial starts right away.
        expect(emitted(deps, 'sessionReconnecting')).toEqual([
            expect.objectContaining({ cause: 'Session expired while tab was backgrounded' }),
        ]);
        expect(deps.dial).toHaveBeenCalledWith('robot-1');
    });

    it('clearIceGrace disarms the pending visibility handler', async () => {
        const { sup, deps } = makeSupervisor({}, { autoReconnect: false });
        hidden = true;
        sup.onIceDisconnected();
        sup.clearIceGrace();

        hidden = false;
        document.dispatchEvent(new Event('visibilitychange'));
        await vi.runAllTimersAsync();
        expect(emittedErrors(deps)).toHaveLength(0);
    });
});

describe('network listeners', () => {
    it('forwards online/offline while installed, stops after uninstall', () => {
        const { sup, deps } = makeSupervisor();
        sup.installNetworkListeners();

        window.dispatchEvent(new Event('online'));
        window.dispatchEvent(new Event('offline'));
        expect(emitted(deps, 'networkOnline')).toHaveLength(1);
        expect(emitted(deps, 'networkOffline')).toHaveLength(1);

        sup.uninstallNetworkListeners();
        window.dispatchEvent(new Event('online'));
        expect(emitted(deps, 'networkOnline')).toHaveLength(1);
    });

    it('is idempotent: double install does not double the events', () => {
        const { sup, deps } = makeSupervisor();
        sup.installNetworkListeners();
        sup.installNetworkListeners();
        window.dispatchEvent(new Event('online'));
        expect(emitted(deps, 'networkOnline')).toHaveLength(1);
        sup.uninstallNetworkListeners();
    });
});
