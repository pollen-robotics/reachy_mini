/**
 * MessageReassembler tests.
 *
 * The frames are the ones the daemon's `split_for_data_channel` produces:
 * ordered `message_chunk` frames whose `data` slices join back into the
 * original JSON text.
 */
import { describe, expect, it } from 'vitest';

import { MessageReassembler } from './message-chunks.js';

/** Split like the daemon does, into `size`-character slices. */
function chunk(message: Record<string, unknown>, id: string, size: number) {
    const text = JSON.stringify(message);
    const parts: string[] = [];
    for (let i = 0; i < text.length; i += size) parts.push(text.slice(i, i + size));
    return parts.map((data, index) => ({ type: 'message_chunk', id, index, count: parts.length, data }));
}

describe('MessageReassembler', () => {
    it('passes a frame that is not a chunk straight through', () => {
        const frame = { jsonrpc: '2.0', id: 1, result: 'ok' };
        expect(new MessageReassembler().accept(frame)).toBe(frame);
    });

    it('returns the message once its last chunk lands', () => {
        const reply = { jsonrpc: '2.0', id: 'sorry_bro', result: { svg: '<svg/>'.repeat(5000) } };
        const frames = chunk(reply, 'a', 4096);
        const reassembler = new MessageReassembler();

        const results = frames.map((f) => reassembler.accept(f));

        expect(results.slice(0, -1).every((r) => r === null)).toBe(true);
        expect(results.at(-1)).toEqual(reply);
    });

    it('keeps two messages apart when their chunks interleave', () => {
        const first = { jsonrpc: '2.0', id: 1, result: 'a'.repeat(9000) };
        const second = { jsonrpc: '2.0', id: 2, result: 'b'.repeat(9000) };
        const [a, b] = [chunk(first, 'one', 4096), chunk(second, 'two', 4096)];
        const reassembler = new MessageReassembler();

        const done = [0, 1, 2]
            .flatMap((i) => [a[i]!, b[i]!])
            .map((f) => reassembler.accept(f))
            .filter((r) => r !== null);

        expect(done).toEqual([first, second]);
    });

    it('drops a message with a missing chunk instead of corrupting it', () => {
        const frames = chunk({ jsonrpc: '2.0', id: 1, result: 'x'.repeat(9000) }, 'a', 4096);
        const reassembler = new MessageReassembler();

        expect(reassembler.accept(frames[0]!)).toBeNull();
        expect(reassembler.accept(frames[2]!)).toBeNull();
        // The next message still reassembles.
        const next = { jsonrpc: '2.0', id: 2, result: 'ok' };
        const again = chunk(next, 'b', 8);
        expect(again.map((f) => reassembler.accept(f)).at(-1)).toEqual(next);
    });
});
