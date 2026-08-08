/**
 * PendingReplies ledger tests.
 *
 * The properties that must hold, per discipline:
 *   - reply slots: single-flight per slot — a newer call supersedes the
 *     older waiter with `null`; a missing daemon reply fails open to
 *     `null` after the roundtrip timeout; a stale settle can't clobber
 *     a newer call's registration;
 *   - JSON-RPC: responses are routed by id; error responses carry the
 *     message and machine-readable `reason`; a late reply to a
 *     timed-out call is swallowed (still counts as response-shaped);
 *   - motion completions: FIFO per command; replies that are neither a
 *     completion nor an error fall through to the caller;
 *   - broadcast waiters: predicate match consumes the most recent
 *     waiter; timeout carries the debug label;
 *   - settleAll(): one call settles all four ledgers — slots resolve
 *     `null`, everything else rejects — so no caller can be left
 *     hanging on a dead channel.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { PendingReplies } from './pending-replies.js';

let pending: PendingReplies;

beforeEach(() => {
    vi.useFakeTimers();
    pending = new PendingReplies();
});

afterEach(() => {
    vi.useRealTimers();
});

describe('reply slots', () => {
    it('resolves the waiter when the daemon reply lands', async () => {
        const send = vi.fn();
        const p = pending.slotRoundtrip('version', send);
        expect(send).toHaveBeenCalledTimes(1);
        expect(pending.settleReplySlot('version', '1.2.3')).toBe(true);
        await expect(p).resolves.toBe('1.2.3');
    });

    it('reports an unclaimed reply so the handler can fall through', () => {
        expect(pending.settleReplySlot('version', '1.2.3')).toBe(false);
    });

    it('supersedes a pending call on the same slot with null', async () => {
        const first = pending.slotRoundtrip('volume', vi.fn());
        const second = pending.slotRoundtrip('volume', vi.fn());
        await expect(first).resolves.toBeNull();
        pending.settleReplySlot('volume', 42);
        await expect(second).resolves.toBe(42);
    });

    it('keeps unrelated slots independent', async () => {
        const version = pending.slotRoundtrip('version', vi.fn());
        const volume = pending.slotRoundtrip('volume', vi.fn());
        pending.settleReplySlot('volume', 10);
        pending.settleReplySlot('version', 'v');
        await expect(volume).resolves.toBe(10);
        await expect(version).resolves.toBe('v');
    });

    it('fails open to null when the daemon never answers (old daemon)', async () => {
        const p = pending.slotRoundtrip('hardware_id', vi.fn());
        await vi.advanceTimersByTimeAsync(4000);
        await expect(p).resolves.toBeNull();
        // The slot is free again: a late daemon reply finds no waiter.
        expect(pending.settleReplySlot('hardware_id', 'late')).toBe(false);
    });

    it('a timed-out call does not clear a newer call on the same slot', async () => {
        const first = pending.slotRoundtrip('robot_name', vi.fn());
        // Just before the first call's timeout, a second call supersedes it…
        await vi.advanceTimersByTimeAsync(3999);
        const second = pending.slotRoundtrip('robot_name', vi.fn());
        await expect(first).resolves.toBeNull();
        // …and the second call's registration survives the first's timer.
        await vi.advanceTimersByTimeAsync(1);
        expect(pending.settleReplySlot('robot_name', 'Reachy')).toBe(true);
        await expect(second).resolves.toBe('Reachy');
    });
});

describe('JSON-RPC', () => {
    /** Capture the id `rpcRoundtrip` mints for the send callback. */
    function roundtrip(method: string, timeoutMs: number): { id: string; p: Promise<unknown> } {
        let id = '';
        const p = pending.rpcRoundtrip(method, timeoutMs, (mintedId) => {
            id = mintedId;
            return true;
        });
        return { id, p };
    }

    it('routes a result response to its waiter by id', async () => {
        const { id, p } = roundtrip('apps.start', 1000);
        expect(pending.settleRpcResponse({ id, result: { ok: true } })).toBe(true);
        await expect(p).resolves.toEqual({ ok: true });
    });

    it('rejects immediately when the send reports a closed channel', async () => {
        await expect(
            pending.rpcRoundtrip('apps.start', 1000, () => false),
        ).rejects.toThrow('rpcCall(apps.start): data channel not open');
    });

    it('rejects with message and machine-readable reason on an error response', async () => {
        const { id, p } = roundtrip('apps.start', 1000);
        expect(pending.settleRpcResponse({
            id,
            error: { message: 'nope', data: { reason: 'app_not_found' } },
        })).toBe(true);
        await expect(p).rejects.toMatchObject({
            message: 'nope',
            reason: 'app_not_found',
        });
    });

    it('swallows a late reply to a timed-out call (still response-shaped)', async () => {
        const { id, p } = roundtrip('conversation.say', 1000);
        void p.catch(() => { /* asserted below */ });
        await vi.advanceTimersByTimeAsync(1000);
        await expect(p).rejects.toThrow(/timed out after 1000ms/);
        // The late reply must not fall through to the notification path.
        expect(pending.settleRpcResponse({ id, result: 'late' })).toBe(true);
    });

    it('leaves notifications (no id) to the caller', () => {
        expect(pending.settleRpcResponse({ method: 'conversation.turn', params: {} })).toBe(false);
        // id present but neither result nor error: not a response either.
        expect(pending.settleRpcResponse({ id: 'rpc-1', method: 'x' })).toBe(false);
    });

    it('mints unique correlation ids', () => {
        const first = roundtrip('a', 1000);
        const second = roundtrip('b', 1000);
        expect(first.id).not.toBe(second.id);
        void first.p.catch(() => { /* timeout */ });
        void second.p.catch(() => { /* timeout */ });
    });
});

describe('motion completions', () => {
    it('resolves waiters in FIFO order per command', async () => {
        const first = pending.awaitMotion('wake_up', 1000);
        const second = pending.awaitMotion('wake_up', 1000);
        const order: string[] = [];
        void first.then(() => order.push('first'));
        void second.then(() => order.push('second'));

        expect(pending.settleMotion('wake_up', { completed: true })).toBe(true);
        expect(pending.settleMotion('wake_up', { completed: true })).toBe(true);
        await vi.runAllTimersAsync();
        expect(order).toEqual(['first', 'second']);
    });

    it('rejects the oldest waiter on an error reply', async () => {
        const p = pending.awaitMotion('goto_sleep', 1000);
        expect(pending.settleMotion('goto_sleep', { error: 'motors off' })).toBe(true);
        await expect(p).rejects.toThrow('goto_sleep: motors off');
    });

    it('falls through when the reply is neither completion nor error', () => {
        void pending.awaitMotion('wake_up', 1000).catch(() => { /* timeout later */ });
        // An intermediate ack without `completed` must not consume the waiter.
        expect(pending.settleMotion('wake_up', { status: 'started' })).toBe(false);
    });

    it('falls through when nothing is pending', () => {
        expect(pending.settleMotion('wake_up', { completed: true })).toBe(false);
    });

    it('rejects on timeout and forgets the waiter', async () => {
        const p = pending.awaitMotion('wake_up', 1000);
        void p.catch(() => { /* asserted below */ });
        await vi.advanceTimersByTimeAsync(1000);
        await expect(p).rejects.toThrow(/wake_up timed out/);
        expect(pending.settleMotion('wake_up', { completed: true })).toBe(false);
    });
});

describe('broadcast waiters', () => {
    it('the first matching broadcast resolves the waiter', async () => {
        const p = pending.awaitBroadcast((m) => m.type === 'move_done');
        expect(pending.matchBroadcast({ type: 'other' })).toBe(false);
        expect(pending.matchBroadcast({ type: 'move_done', id: 7 })).toBe(true);
        await expect(p).resolves.toEqual({ type: 'move_done', id: 7 });
    });

    it('the most recent matching waiter wins', async () => {
        const older = pending.awaitBroadcast((m) => m.type === 'x', { debugLabel: 'older' });
        const newer = pending.awaitBroadcast((m) => m.type === 'x', { debugLabel: 'newer' });
        expect(pending.matchBroadcast({ type: 'x', n: 1 })).toBe(true);
        await expect(newer).resolves.toEqual({ type: 'x', n: 1 });
        expect(pending.matchBroadcast({ type: 'x', n: 2 })).toBe(true);
        await expect(older).resolves.toEqual({ type: 'x', n: 2 });
    });

    it('rejects on timeout with the debug label', async () => {
        const p = pending.awaitBroadcast(() => false, { timeoutMs: 2000, debugLabel: 'move ack' });
        void p.catch(() => { /* asserted below */ });
        await vi.advanceTimersByTimeAsync(2000);
        await expect(p).rejects.toThrow('broadcast timeout (2000 ms): move ack');
    });
});

describe('settleAll (session teardown)', () => {
    it('settles all four ledgers in one call', async () => {
        const slot = pending.slotRoundtrip('version', vi.fn());
        const rpc = pending.rpcRoundtrip('apps.start', 60_000, () => true);
        const motion = pending.awaitMotion('wake_up', 60_000);
        const broadcast = pending.awaitBroadcast(() => true, { timeoutMs: 60_000 });
        // Pre-attach handlers so the synchronous rejections below don't
        // trip Node's late-handling warning before the assertions run.
        for (const p of [rpc, motion, broadcast]) void p.catch(() => { /* asserted below */ });

        pending.settleAll(new Error('Session stopped'));

        // Slots fail open (all slot callers handle null)…
        await expect(slot).resolves.toBeNull();
        // …everything else rejects with the teardown error.
        await expect(rpc).rejects.toThrow('Session stopped');
        await expect(motion).rejects.toThrow('Session stopped');
        await expect(broadcast).rejects.toThrow('Session stopped');
    });

    it('leaves the ledger empty afterwards', () => {
        void pending.slotRoundtrip('version', vi.fn());
        void pending.awaitMotion('wake_up', 1000).catch(() => { /* settled below */ });
        void pending.awaitBroadcast(() => true).catch(() => { /* settled below */ });
        pending.settleAll(new Error('gone'));

        expect(pending.settleReplySlot('version', 'v')).toBe(false);
        expect(pending.settleMotion('wake_up', { completed: true })).toBe(false);
        expect(pending.matchBroadcast({ type: 'any' })).toBe(false);
    });

    it('cancels the pending timers (no late rejection fires)', async () => {
        // A rejection firing after settleAll would be an unhandled one —
        // runAllTimersAsync below would surface it.
        const motion = pending.awaitMotion('wake_up', 1000);
        void motion.catch(() => { /* asserted below */ });
        pending.settleAll(new Error('gone'));
        await expect(motion).rejects.toThrow('gone');
        await vi.runAllTimersAsync();
    });
});
