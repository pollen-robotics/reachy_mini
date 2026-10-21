/**
 * Reassembly of daemon messages too large for one data channel frame.
 *
 * The data channel silently drops any message over 64 KiB, so the daemon
 * splits larger ones into ordered `message_chunk` frames (see
 * `split_for_data_channel` in `reachy_mini/media/webrtc_utils.py`). Each
 * carries a slice of the original text in `data`; joined in order they are
 * the message the daemon meant to send.
 *
 * One instance per control channel: the channel is ordered, and a new
 * channel means a new session, so nothing survives across the two.
 */

/** Upper bound on one reassembled message, against a runaway sender. */
const MAX_CHUNKS = 4096;

export class MessageReassembler {
    private readonly _partials = new Map<string, string[]>();

    /**
     * Feed one parsed frame. Returns the message to handle: the frame itself
     * when it is not a chunk, the reassembled message when this chunk
     * completes one, or `null` while a message is still arriving.
     */
    accept(frame: Record<string, unknown>): Record<string, unknown> | null {
        if (frame.type !== 'message_chunk') return frame;
        const { id, index, count, data } = frame;
        if (
            typeof id !== 'string' ||
            typeof index !== 'number' ||
            typeof count !== 'number' ||
            typeof data !== 'string' ||
            count < 1 ||
            count > MAX_CHUNKS
        ) {
            return null;
        }

        let parts = this._partials.get(id);
        if (index === 0) {
            parts = [];
            this._partials.set(id, parts);
        }
        // The channel is ordered, so a gap means the message is lost.
        if (!parts || index !== parts.length) {
            this._partials.delete(id);
            return null;
        }

        parts.push(data);
        if (parts.length < count) return null;

        this._partials.delete(id);
        return JSON.parse(parts.join('')) as Record<string, unknown>;
    }
}
