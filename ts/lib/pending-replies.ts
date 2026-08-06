/**
 * Pending-reply bookkeeping for the ReachyMini SDK.
 *
 * The data channel multiplexes four request/response disciplines, each
 * with its own correlation rule:
 *
 *   1. Reply slots — legacy `{type: ...}` commands whose replies carry
 *      no request id, correlated by message TYPE. At most one call per
 *      slot is in flight; a newer call supersedes the older waiter.
 *   2. JSON-RPC — app-control calls correlated by an `id` we generate.
 *   3. Motion completions — `wake_up` / `goto_sleep` acks, correlated
 *      by command name, FIFO per command.
 *   4. Broadcast waiters — daemon broadcasts (move / audio playback
 *      lifecycle) matched by caller-supplied predicate.
 *
 * This class owns the WAITERS only: sending stays with `ReachyMini`
 * (it owns the data channel), which passes send actions in as
 * callbacks. Every session-teardown path settles the whole ledger at
 * once via `settleAll`, so no caller is ever left hanging on a dead
 * channel — and no future mechanism can be forgotten on a teardown
 * path, because there is only one.
 */

import type { FaceTarget, ImuData } from './types.js';

/**
 * Fail-open ceiling for a single reply-slot request/response.
 * Every slot command has a strict "one reply per request" contract, so
 * a missing reply means the daemon either never got it or - crucially -
 * predates that command entirely: an older daemon silently drops an
 * unknown `type` and sends nothing back, which would otherwise leave the
 * caller's promise pending forever (e.g. a newer SDK calling a command a
 * 1.8.x daemon doesn't implement). Resolving `null` on timeout maps
 * cleanly onto the "unsupported / failed" value every slot caller already
 * handles. 4 s is comfortably above a WebRTC data-channel round trip on a
 * congested phone link while still failing fast enough that a gated UI
 * doesn't feel hung.
 */
export const SLOT_ROUNDTRIP_TIMEOUT_MS = 4000;

/**
 * Value type carried by each single-slot reply waiter. These daemon
 * replies are correlated by message TYPE — the legacy command set
 * carries no request ids — so at most one call per slot is in flight.
 */
export interface ReplySlotValues {
    version: string | null;
    hardware_id: string | null;
    volume: number | null;
    mic_volume: number | null;
    tracked_face: FaceTarget | null;
    robot_name: string | null;
    delete_hf_token: boolean | null;
    apply_audio_config: boolean | null;
    read_audio_parameter: number[] | null;
    imu: ImuData | null;
}
export type ReplySlotKey = keyof ReplySlotValues;

export type MotionCommand = 'wake_up' | 'goto_sleep';

/**
 * Sentinel for a broadcast waiter that ran out its own timeout (as
 * opposed to being rejected by a session teardown via `settleAll`).
 * Callers that fail-open on timeout — `request()` — detect it by type
 * instead of coupling to the message string.
 */
export class BroadcastTimeoutError extends Error {}

interface PendingRpc {
    resolve: (v: unknown) => void;
    reject: (e: Error) => void;
    timer: ReturnType<typeof setTimeout>;
}

interface PendingMotion {
    resolve: () => void;
    reject: (err: Error) => void;
    timer: ReturnType<typeof setTimeout>;
}

interface BroadcastWaiter {
    predicate: (m: Record<string, unknown>) => boolean;
    resolve: (m: Record<string, unknown>) => void;
    reject: (err: Error) => void;
    timer: ReturnType<typeof setTimeout>;
}

export class PendingReplies {
    private readonly _replySlots = new Map<ReplySlotKey, (v: unknown) => void>();
    private _rpcCounter = 0;
    private readonly _pendingRpc = new Map<string, PendingRpc>();
    private readonly _pendingMotion: Record<MotionCommand, PendingMotion[]> = {
        wake_up: [],
        goto_sleep: [],
    };
    private _broadcastWaiters: BroadcastWaiter[] = [];

    /* ─── Reply slots ────────────────────────────────────────────────── */

    /**
     * Register a slot waiter, run `send`, and await the matching daemon
     * response. If a previous request on the same slot is still pending
     * when a new one comes in, the older promise is resolved to `null`
     * so its caller doesn't hang forever.
     */
    slotRoundtrip<K extends ReplySlotKey>(
        slot: K,
        send: () => void,
    ): Promise<ReplySlotValues[K] | null> {
        return new Promise<ReplySlotValues[K] | null>((resolve) => {
            const prev = this._replySlots.get(slot);
            if (prev) prev(null);
            let timer: ReturnType<typeof setTimeout> | undefined;
            // Single settle path shared by the daemon response, supersession
            // by a newer call, and the fail-open timeout below. It clears the
            // timer once and detaches itself from the slot only if it's still
            // the current occupant, so a stale settle (timed-out or superseded)
            // can't clear a newer call's slot registration.
            // Note: slots are keyed by command type, not request id, so this
            // does not prevent a genuinely late daemon reply from being routed
            // to a newer same-command caller — that cross-talk is inherent to
            // the single-flight slot design.
            const settle = (v: unknown): void => {
                if (timer !== undefined) {
                    clearTimeout(timer);
                    timer = undefined;
                }
                if (this._replySlots.get(slot) === settle) this._replySlots.delete(slot);
                resolve(v as ReplySlotValues[K] | null);
            };
            this._replySlots.set(slot, settle);
            timer = setTimeout(() => settle(null), SLOT_ROUNDTRIP_TIMEOUT_MS);
            send();
        });
    }

    /**
     * Deliver a daemon reply to the pending waiter for `slot`, if any.
     * Returns `true` when a waiter consumed the value — some message-handler
     * branches use that to decide whether the message was theirs to swallow.
     */
    settleReplySlot<K extends ReplySlotKey>(slot: K, value: ReplySlotValues[K]): boolean {
        const waiter = this._replySlots.get(slot);
        if (!waiter) return false;
        waiter(value);
        return true;
    }

    /* ─── JSON-RPC ───────────────────────────────────────────────────── */

    /**
     * One JSON-RPC round-trip: mint the correlation id, hand it to
     * `send` (which reports `false` when the channel isn't open — the
     * promise then rejects immediately), and await the matching
     * response; rejects on timeout. Same shape as `slotRoundtrip`.
     */
    rpcRoundtrip(
        method: string,
        timeoutMs: number,
        send: (id: string) => boolean,
    ): Promise<unknown> {
        const id = `rpc-${++this._rpcCounter}`;
        if (!send(id)) {
            return Promise.reject(new Error(`rpcCall(${method}): data channel not open`));
        }
        return new Promise<unknown>((resolve, reject) => {
            const timer = setTimeout(() => {
                this._pendingRpc.delete(id);
                reject(new Error(`rpcCall(${method}) timed out after ${timeoutMs}ms`));
            }, timeoutMs);
            this._pendingRpc.set(id, { resolve, reject, timer });
        });
    }

    /**
     * Route a JSON-RPC RESPONSE (id + result/error) to its waiter.
     * Returns `true` when the message was response-shaped — even if the
     * waiter already timed out — so the caller can stop processing it;
     * `false` for notifications and anything else.
     */
    settleRpcResponse(data: Record<string, unknown>): boolean {
        if (!('id' in data) || data.id == null || !('result' in data || 'error' in data)) {
            return false;
        }
        const pending = this._pendingRpc.get(data.id as string);
        if (!pending) return true; // late reply to a timed-out call
        this._pendingRpc.delete(data.id as string);
        clearTimeout(pending.timer);
        if ('error' in data && data.error) {
            const err = data.error as { message?: string; data?: { reason?: string } };
            const e = new Error(err.message ?? 'rpc error');
            (e as Error & { reason?: string }).reason = err.data?.reason;
            pending.reject(e);
        } else {
            pending.resolve((data as { result?: unknown }).result);
        }
        return true;
    }

    /* ─── Motion completions (wake_up / goto_sleep) ──────────────────── */

    /** FIFO waiter for one `wake_up` / `goto_sleep` ack; rejects on timeout. */
    awaitMotion(command: MotionCommand, timeoutMs: number): Promise<void> {
        return new Promise<void>((resolve, reject) => {
            const entry: PendingMotion = {
                resolve,
                reject,
                timer: setTimeout(() => {
                    const queue = this._pendingMotion[command];
                    const idx = queue.indexOf(entry);
                    if (idx !== -1) queue.splice(idx, 1);
                    reject(new Error(`${command} timed out after ${timeoutMs}ms`));
                }, timeoutMs),
            };
            this._pendingMotion[command].push(entry);
        });
    }

    /**
     * Deliver a `wake_up` / `goto_sleep` reply to the oldest waiter.
     * Returns `true` when a waiter consumed it; `false` when nothing was
     * pending or the reply is neither a completion nor an error (the
     * message then falls through to the caller's other branches).
     */
    settleMotion(command: MotionCommand, data: Record<string, unknown>): boolean {
        const queue = this._pendingMotion[command];
        if (data.completed === true && queue.length > 0) {
            const entry = queue.shift()!;
            clearTimeout(entry.timer);
            entry.resolve();
            return true;
        }
        if (data.error && queue.length > 0) {
            const entry = queue.shift()!;
            clearTimeout(entry.timer);
            entry.reject(new Error(`${command}: ${data.error}`));
            return true;
        }
        return false;
    }

    /* ─── Broadcast waiters ──────────────────────────────────────────── */

    /** Await the first broadcast matching `predicate`; rejects on timeout. */
    awaitBroadcast(
        predicate: (m: Record<string, unknown>) => boolean,
        { timeoutMs = 5000, debugLabel = '' }: { timeoutMs?: number; debugLabel?: string } = {},
    ): Promise<Record<string, unknown>> {
        return new Promise<Record<string, unknown>>((resolve, reject) => {
            const slot: BroadcastWaiter = {
                predicate,
                resolve,
                reject,
                timer: setTimeout(() => {
                    const i = this._broadcastWaiters.indexOf(slot);
                    if (i !== -1) this._broadcastWaiters.splice(i, 1);
                    reject(new BroadcastTimeoutError(`broadcast timeout (${timeoutMs} ms): ${debugLabel}`));
                }, timeoutMs),
            };
            this._broadcastWaiters.push(slot);
        });
    }

    /**
     * Offer a broadcast to the waiters; the first (most recent) matching
     * waiter consumes it. Returns `true` when one did.
     */
    matchBroadcast(data: Record<string, unknown>): boolean {
        for (let i = this._broadcastWaiters.length - 1; i >= 0; i--) {
            const slot = this._broadcastWaiters[i]!;
            if (slot.predicate(data)) {
                this._broadcastWaiters.splice(i, 1);
                clearTimeout(slot.timer);
                slot.resolve(data);
                return true;
            }
        }
        return false;
    }

    /* ─── Teardown ───────────────────────────────────────────────────── */

    /**
     * Settle the whole ledger against a dead channel: reply slots
     * resolve `null` (the "no answer" value all slot callers handle;
     * public wrappers coerce where needed, e.g. `applyAudioConfig` maps
     * it to `false`), everything else rejects with `err`. Called from
     * every session-teardown path.
     */
    settleAll(err: Error): void {
        for (const waiter of [...this._replySlots.values()]) {
            waiter(null);
        }
        this._replySlots.clear();

        for (const pending of this._pendingRpc.values()) {
            clearTimeout(pending.timer);
            pending.reject(err);
        }
        this._pendingRpc.clear();

        for (const command of Object.keys(this._pendingMotion) as MotionCommand[]) {
            const queue = this._pendingMotion[command];
            while (queue.length) {
                const entry = queue.shift()!;
                clearTimeout(entry.timer);
                entry.reject(err);
            }
        }

        for (const slot of this._broadcastWaiters) {
            clearTimeout(slot.timer);
            slot.reject(err);
        }
        this._broadcastWaiters = [];
    }
}
