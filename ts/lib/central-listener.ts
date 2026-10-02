/**
 * Long-lived SSE listener channel to Hugging Face central: a realtime
 * view of the user's robot fleet WITHOUT claiming a peer slot. The
 * canonical implementation shared by every first-party picker / scan
 * screen (the per-app copies used to drift).
 *
 * Why not just use the `ReachyMini` SDK class? Its signaling stream
 * registers a *peer* at central, and central enforces a 1:1
 * `token → peer_id` mapping - so a fleet-watching UI that kept an SDK
 * instance open would clobber the peer slot the next session's WebRTC
 * handshake needs. This module registers with `roles: ["listener"]`
 * only, and callers `close()` it before any SDK session starts on the
 * same token.
 *
 * `fetch` + manual stream reader rather than the native `EventSource`
 * so the HF token travels in the `Authorization` header instead of
 * the URL (no secret in DevTools / proxy logs / browser history).
 */

const INITIAL_BACKOFF_MS = 1_000;
const MAX_BACKOFF_MS = 30_000;
const DEFAULT_LISTENER_APP_NAME = 'Reachy Mini (listener)';

// Heartbeat: the server runs a TTL sweeper that evicts peers whose
// `last_seen` is older than `LEASE_SECONDS` (30 s by default; cf.
// `reachy_mini_central/app.py`). `last_seen` only refreshes on inbound
// `POST /send`, so a pure SSE listener that registers once and stays
// silent gets evicted silently after one lease window - the SSE stays
// open but every subsequent broadcast is dropped because the server's
// `broadcast_to_listeners` iterates `self.peers`, which no longer
// contains us. Symptom: the picker sees `busy=true` when a session
// starts (within the first 30 s) but never receives the matching
// `busy=false` because the listener slot has been reaped by then.
//
// Mirror the daemon's `_heartbeat_loop` (cf. `central_signaling_relay.py`):
// re-emit `setPeerStatus(listener)` periodically, at the cadence the
// welcome recommends (`recommended_heartbeat_interval_seconds`, with a
// `lease_seconds / 3` fallback).
const HEARTBEAT_DEFAULT_INTERVAL_MS = 10_000;

const negotiateHeartbeatIntervalMs = (
    welcomeMsg: Record<string, unknown>,
): number => {
    const raw = welcomeMsg['recommended_heartbeat_interval_seconds'];
    if (typeof raw === 'number' && Number.isFinite(raw) && raw > 0) {
        return raw * 1000;
    }
    const lease = welcomeMsg['lease_seconds'];
    if (typeof lease === 'number' && Number.isFinite(lease) && lease > 0) {
        return (lease * 1000) / 3;
    }
    return HEARTBEAT_DEFAULT_INTERVAL_MS;
};

/**
 * Shape of a producer entry in the SSE `list` frame and in
 * `/api/robot-status`. Mirrors `get_producers_list` server-side.
 * Kept loose so future fields appear without a wire bump.
 */
export interface CentralStreamProducer {
    id: string;
    meta?: Record<string, unknown>;
    busy?: boolean;
    activeApp?: string | null;
}

export interface CentralPeerStatusChangedEvent {
    peerId: string;
    roles: string[];
    meta?: Record<string, unknown>;
}

export interface CentralSessionStateChangedEvent {
    peerId: string;
    busy: boolean;
    activeApp?: string | null;
    meta?: Record<string, unknown>;
}

export interface OpenCentralListenerOpts {
    token: string;
    /** Central base URL, e.g. from the consumer's env/config. */
    signalingUrl: string;
    /**
     * `setPeerStatus(listener)` payload's `meta.name`. Surfaces in
     * central logs as the listener identity - pass something that
     * identifies the surface (e.g. "Reachy Mini Host (picker)").
     */
    appName?: string;
    /** First batch of producers received after each (re)connect. */
    onList?: (producers: CentralStreamProducer[]) => void;
    /** A robot came online / went offline (same-owner). */
    onPeerStatusChanged?: (event: CentralPeerStatusChangedEvent) => void;
    /** A robot transitioned busy ↔ free (same-owner). */
    onSessionStateChanged?: (event: CentralSessionStateChangedEvent) => void;
    /** SSE is open and `setPeerStatus(listener)` has been POSTed. */
    onConnect?: () => void;
    /** SSE dropped (network, central restart, scheduled reconnect). */
    onDisconnect?: (reason: string) => void;
    /** Non-fatal errors. Fatal auth errors end the stream and call
     *  onDisconnect with a meaningful reason instead. */
    onError?: (err: Error) => void;
}

export interface CentralListenerHandle {
    /** Tear the stream down. Idempotent. No further callbacks
     *  fire after this returns. */
    close(): void;
}

export function openCentralListener(
    opts: OpenCentralListenerOpts,
): CentralListenerHandle {
    const { signalingUrl, token } = opts;
    const appName = opts.appName ?? DEFAULT_LISTENER_APP_NAME;
    let closed = false;
    let abortController: AbortController | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
    let backoff = INITIAL_BACKOFF_MS;

    const stopHeartbeat = (): void => {
        if (heartbeatTimer !== null) {
            clearInterval(heartbeatTimer);
            heartbeatTimer = null;
        }
    };

    const close = (): void => {
        if (closed) return;
        closed = true;
        stopHeartbeat();
        if (reconnectTimer !== null) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
        if (abortController !== null) {
            abortController.abort();
            abortController = null;
        }
    };

    const scheduleReconnect = (reason: string): void => {
        // The `reconnectTimer` check is a re-entrancy guard: stream EOF and
        // an in-flight heartbeat's eviction branch can both land here in the
        // same tick, and arming twice would spawn two concurrent runStream()
        // loops (the abortController only tracks the latest).
        if (closed || reconnectTimer !== null) return;
        // Stop the heartbeat eagerly: the SSE is dead, so any outstanding
        // setPeerStatus POST would fail anyway and the next welcome will
        // start a fresh interval with a freshly-negotiated cadence.
        stopHeartbeat();
        opts.onDisconnect?.(reason);
        const jitter = backoff * (Math.random() - 0.5) * 0.5;
        const delay = Math.min(MAX_BACKOFF_MS, Math.round(backoff + jitter));
        reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            backoff = Math.min(MAX_BACKOFF_MS, backoff * 2);
            void runStream();
        }, delay);
    };

    /**
     * POST `setPeerStatus(roles=['listener'])`.
     *
     * Used both as the one-shot post-welcome registration and as the
     * recurring heartbeat re-emission. The wire payload is identical
     * by design: central treats repeated setPeerStatus from the same
     * peer as idempotent and uses the inbound POST to refresh
     * `last_seen` (cf. `signaling.touch()` in central app.py).
     *
     * Returns the HTTP status code on a completed round-trip (so the
     * caller can distinguish "peer evicted server-side" from "transient
     * network blip"), or `null` on abort / network error.
     */
    const sendListenerRegistration = async (): Promise<number | null> => {
        try {
            const resp = await fetch(`${signalingUrl}/send`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify({
                    type: 'setPeerStatus',
                    roles: ['listener'],
                    meta: { name: appName },
                }),
                signal: abortController?.signal,
            });
            if (!resp.ok) {
                opts.onError?.(
                    new Error(`setPeerStatus(listener) returned HTTP ${resp.status}`),
                );
            }
            return resp.status;
        } catch (err) {
            if (err instanceof DOMException && err.name === 'AbortError') return null;
            opts.onError?.(err instanceof Error ? err : new Error(String(err)));
            return null;
        }
    };

    const startHeartbeat = (intervalMs: number): void => {
        stopHeartbeat();
        if (closed) return;
        heartbeatTimer = setInterval(() => {
            if (closed) return;
            void (async () => {
                const status = await sendListenerRegistration();
                if (closed) return;
                // Server-side eviction: `disconnect_peer` (TTL sweeper or
                // explicit) clears `token_to_peer`, so the next POST /send
                // returns HTTP 400 ("Connect to /events first"). The SSE is
                // still half-open from our side - keepalive pings keep it
                // looking alive on the wire even though the central no longer
                // routes broadcasts to us. Abort the dead stream AND schedule
                // the reconnect ourselves: `runStream()` deliberately swallows
                // aborts (they normally mean `close()`), so it would never
                // reach its own `scheduleReconnect()` on this path. The
                // reconnect mints a fresh peer and replays `welcome` + `list`.
                if (status !== null && status >= 400 && status < 500) {
                    if (abortController !== null) {
                        abortController.abort();
                    }
                    scheduleReconnect(`listener evicted (HTTP ${status})`);
                }
            })();
        }, intervalMs);
    };

    const handleMessage = (
        msg: { type?: string } & Record<string, unknown>,
    ): void => {
        switch (msg.type) {
            case 'welcome':
                backoff = INITIAL_BACKOFF_MS;
                // Register (central requires the SSE channel be established
                // before it accepts a /send for this peer), THEN report
                // connected - so `onConnect` consumers can immediately issue
                // REST calls without racing the listener slot creation.
                void (async () => {
                    await sendListenerRegistration();
                    if (closed) return;
                    startHeartbeat(negotiateHeartbeatIntervalMs(msg));
                    opts.onConnect?.();
                })();
                return;
            // Frames come from the first-party central, so they are cast, not
            // field-validated — same trust level as the SDK's own signaling
            // parser.
            case 'list':
                opts.onList?.(
                    Array.isArray(msg.producers)
                        ? (msg.producers as CentralStreamProducer[])
                        : [],
                );
                return;
            case 'peerStatusChanged':
                opts.onPeerStatusChanged?.(
                    msg as unknown as CentralPeerStatusChangedEvent,
                );
                return;
            case 'sessionStateChanged':
                opts.onSessionStateChanged?.(
                    msg as unknown as CentralSessionStateChangedEvent,
                );
                return;
            // Other types (`peer`, `startSession`, `endSession`, `ping`)
            // only fire when this peer becomes consumer / producer. As
            // a pure listener we ignore them.
            default:
                return;
        }
    };

    const runStream = async (): Promise<void> => {
        if (closed) return;
        abortController = new AbortController();
        const signal = abortController.signal;

        let resp: Response;
        try {
            resp = await fetch(`${signalingUrl}/events`, {
                method: 'GET',
                headers: { Authorization: `Bearer ${token}` },
                signal,
            });
        } catch (err) {
            if (closed) return;
            const reason = err instanceof Error ? err.message : String(err);
            scheduleReconnect(`fetch failed: ${reason}`);
            return;
        }

        if (resp.status === 401 || resp.status === 403) {
            opts.onError?.(
                new Error(
                    `Hugging Face central rejected the token (HTTP ${resp.status})`,
                ),
            );
            close();
            opts.onDisconnect?.('auth_rejected');
            return;
        }
        if (!resp.ok || resp.body === null) {
            scheduleReconnect(`HTTP ${resp.status}`);
            return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() ?? '';
                for (const line of lines) {
                    if (!line.startsWith('data:')) continue;
                    const payload = line.slice(5).trim();
                    if (payload.length === 0) continue;
                    try {
                        handleMessage(JSON.parse(payload) as Record<string, unknown>);
                    } catch (err) {
                        opts.onError?.(
                            err instanceof Error ? err : new Error(String(err)),
                        );
                    }
                }
            }
            // `signal.aborted` covers both `close()` and the heartbeat's
            // eviction path (which schedules its own reconnect); some fetch
            // implementations surface an abort as a clean EOF rather than an
            // AbortError, so check it here too to avoid double-scheduling.
            if (closed || signal.aborted) return;
            scheduleReconnect('stream ended');
        } catch (err) {
            if (signal.aborted || closed) return;
            const reason = err instanceof Error ? err.message : String(err);
            scheduleReconnect(`read failed: ${reason}`);
        }
    };

    void runStream();

    return { close };
}
