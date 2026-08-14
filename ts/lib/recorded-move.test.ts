/**
 * playRecordedMove / playRecordedMoveAndWait - daemon-side named-move playback.
 *
 * The properties that must hold:
 *   - both variants put the same payload on the wire, so the awaited one
 *     can't drift from the fire-and-forget one;
 *   - the awaited variant resolves `true` on the daemon's dispatch ack and
 *     `false` when the daemon reports it couldn't load the move;
 *   - that failure ack carries NO `move_name` (only `command`, `status`,
 *     `error`), so the reply matcher must key on the command echo alone -
 *     matching on the move name would miss every failure and fall through
 *     to the timeout, reporting a load error as "no answer";
 *   - a daemon that never answers fail-opens to `null`, on a deadline long
 *     enough to cover a cold dataset download rather than a round trip;
 *   - a closed data channel is a rejection, not a silent `false`.
 *
 * Same approach as the other SDK suites: private surface + a fake data
 * channel, fake timers for the fail-open path.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReachyMini } from './reachy-mini.js';

/** Private surface the tests poke at, spelled out for the casts. */
interface RecordedMoveInternals {
    _dc: { readyState: string; send: (msg: string) => void } | null;
    _handleRobotMessage(data: Record<string, unknown>): void;
}

function makeInstance(): {
    r: ReachyMini;
    internals: RecordedMoveInternals;
    sent: Array<Record<string, unknown>>;
} {
    const r = new ReachyMini();
    const internals = r as unknown as RecordedMoveInternals;
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

describe('playRecordedMove - wire contract', () => {
    it('sends the move name alone by default', () => {
        const { r, sent } = makeInstance();

        expect(r.playRecordedMove('cheerful1')).toBe(true);
        expect(sent).toEqual([{ type: 'play_recorded_move', move_name: 'cheerful1' }]);
    });

    it('forwards dataset and lead-in goto when given', () => {
        const { r, sent } = makeInstance();

        r.playRecordedMove('wave', { dataset: 'someone/moves', initialGotoDuration: 0.4 });

        expect(sent).toEqual([{
            type: 'play_recorded_move',
            move_name: 'wave',
            dataset_name: 'someone/moves',
            initial_goto_duration: 0.4,
        }]);
    });

    it('puts the same payload on the wire as the awaited variant', () => {
        const { r: a, sent: sentSync } = makeInstance();
        const { r: b, sent: sentAwaited } = makeInstance();

        a.playRecordedMove('wave', { dataset: 'someone/moves', initialGotoDuration: 0.4 });
        void b.playRecordedMoveAndWait('wave', {
            dataset: 'someone/moves',
            initialGotoDuration: 0.4,
        });

        expect(sentAwaited).toEqual(sentSync);
    });
});

describe('playRecordedMoveAndWait - dispatch ack', () => {
    it('resolves true once the daemon acks the dispatch', async () => {
        const { r, internals } = makeInstance();

        const promise = r.playRecordedMoveAndWait('cheerful1');
        internals._handleRobotMessage({
            command: 'play_recorded_move',
            status: 'ok',
            move_name: 'cheerful1',
        });

        await expect(promise).resolves.toBe(true);
    });

    it('resolves false on a load failure ack, which carries no move_name', async () => {
        const { r, internals } = makeInstance();

        const promise = r.playRecordedMoveAndWait('nope', { dataset: 'someone/moves' });
        internals._handleRobotMessage({
            command: 'play_recorded_move',
            status: 'error',
            error: 'Move nope not found in recorded moves library someone/moves',
        });

        await expect(promise).resolves.toBe(false);
    });

    it('waits well past a data-channel round trip, for a cold dataset download', async () => {
        const { r, internals } = makeInstance();
        const settled = vi.fn();

        void r.playRecordedMoveAndWait('cheerful1').then(settled);

        // A download can easily outlast the SDK's 4 s command round-trip
        // budget; the ack that lands afterwards must still be honoured.
        await vi.advanceTimersByTimeAsync(30000);
        expect(settled).not.toHaveBeenCalled();

        internals._handleRobotMessage({
            command: 'play_recorded_move',
            status: 'ok',
            move_name: 'cheerful1',
        });
        await vi.advanceTimersByTimeAsync(0);

        expect(settled).toHaveBeenCalledWith(true);
    });

    it('fail-opens to null when the daemon never answers', async () => {
        const { r } = makeInstance();

        const promise = r.playRecordedMoveAndWait('cheerful1');
        await vi.advanceTimersByTimeAsync(120000);

        await expect(promise).resolves.toBeNull();
    });

    it('respects a caller-provided timeout', async () => {
        const { r } = makeInstance();

        const promise = r.playRecordedMoveAndWait('cheerful1', { timeoutMs: 500 });
        await vi.advanceTimersByTimeAsync(500);

        await expect(promise).resolves.toBeNull();
    });

    it('rejects when the data channel is not open', async () => {
        const { r, internals } = makeInstance();
        internals._dc = null;

        await expect(r.playRecordedMoveAndWait('cheerful1')).rejects.toThrow(
            /Data channel not open/,
        );
    });
});
