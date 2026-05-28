/**
 * Wire-level constants and pure helpers used by playMove / uploadAudio /
 * playUploadedAudio. Private to the SDK — apps call the public methods.
 */

// Conservative per-message size for the data channel. 16 KB is the cross-
// browser safe ceiling; we slice payloads at 12 KB and let the JSON envelope
// add ~80 bytes.
export const UPLOAD_CHUNK_SIZE = 12 * 1024;

// Backpressure thresholds: pause sending if `bufferedAmount` climbs over the
// high watermark; resume once it drains below the low watermark. WebRTC's
// SCTP can buffer plenty, but spiking it to tens of megabytes degrades every
// other channel on the same peer connection.
export const UPLOAD_BUFFERED_HIGH_WATER = 1 * 1024 * 1024;
export const UPLOAD_BUFFERED_LOW_WATER = 512 * 1024;

export function hasCompressionStream(): boolean {
    return typeof CompressionStream !== 'undefined';
}

/** Cheap unique upload id; collision odds within a session are negligible. */
export function makeUploadId(): string {
    return 'u' + Math.random().toString(36).slice(2, 11)
        + Date.now().toString(36);
}

/** Base64-encode a Uint8Array, chunking to avoid call-stack overflow on multi-MB blobs. */
export function bytesToBase64(bytes: Uint8Array): string {
    let str = '';
    const STEP = 0x8000;
    for (let i = 0; i < bytes.length; i += STEP) {
        str += String.fromCharCode.apply(
            null,
            Array.from(bytes.subarray(i, i + STEP)),
        );
    }
    return btoa(str);
}

/** Base64(gzip(utf8(s))) via the browser CompressionStream API. */
export async function gzipBase64(jsonStr: string): Promise<string> {
    const enc = new TextEncoder().encode(jsonStr);
    const compressed = await new Response(
        new Blob([enc as BlobPart]).stream().pipeThrough(new CompressionStream('gzip')),
    ).arrayBuffer();
    return bytesToBase64(new Uint8Array(compressed));
}

/** Clamp a volume to [0, 100] and round to integer — mirrors the server-side
 *  Field(..., ge=0, le=100) validator so calling setVolume(150) doesn't 400. */
export function clampVolume(v: number): number {
    const n = Math.round(Number(v) || 0);
    return Math.max(0, Math.min(100, n));
}
