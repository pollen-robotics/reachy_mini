/**
 * Session resilience supervisor for the ReachyMini SDK.
 *
 * Owns the three transport-resilience mechanisms so the main class only
 * has to forward browser/pc events and expose narrow accessors:
 *
 *   1. ICE-blip debounce + network awareness — grace windows around
 *      `iceConnectionState` transitions (visibility-deferred while the
 *      tab is hidden) and `online`/`offline`/`connection.change`
 *      forwarding as public events.
 *   2. Automatic session re-dial — the daemon's `webrtcbin` cannot do a
 *      standards ICE restart (missing upstream support), so the recovery
 *      unit is the whole SESSION: tear down the dead RTCPeerConnection
 *      and dial the same robot again through central. Runs inside the
 *      SDK so every consumer (host embed, mobile app, third-party apps)
 *      inherits it.
 *   3. Data-channel silence watchdog — application-level heartbeat
 *      without any daemon change: a streaming session already receives
 *      constant inbound traffic (get_state poll replies and/or pushed
 *      pose frames), so prolonged TOTAL silence is a dead transport that
 *      ICE hasn't noticed yet.
 *
 * The supervisor never touches the RTCPeerConnection, data channel or
 * pending promises directly: every side effect goes through
 * `SessionSupervisorDeps`, implemented by `ReachyMini` as closures over
 * its own private state. That keeps the ownership boundary honest — the
 * class owns the transport, the supervisor owns the RECOVERY POLICY.
 */

/* ─── Tunables ─────────────────────────────────────────────────────── */

/**
 * How long we tolerate `iceConnectionState === 'disconnected'` before
 * surfacing it as an error. The spec defines this state as transient
 * (browsers keep STUN keep-alives running and usually heal in 1-2 s
 * on WiFi blips, AP roams, brief 4G dropouts). Consumers watching
 * `iceStateChange` directly should outlive this window before
 * showing any fatal UI.
 */
const ICE_DISCONNECT_GRACE_MS = 3000;

/**
 * Grace before treating `iceConnectionState === 'failed'` as terminal.
 * The spec says `failed` IS terminal, but we've observed real
 * `failed → connected` flips on rapid AP roams and iOS BT route
 * changes — 1 s of debounce absorbs those without noticeably
 * delaying a real failure.
 */
const ICE_FAILED_GRACE_MS = 1000;

/**
 * Ceiling on how long we'll keep `armIceGraceOnVisibility` waiting
 * for the tab to come back. The daemon's `webrtcsink` runs a STUN
 * consent-freshness check (RFC 7675, ~30 s default) and unilaterally
 * tears its side of the session down past that window, releasing the
 * producer slot on central. If the user backgrounded the tab for
 * longer than this, running another 3 s foreground grace is a lie —
 * the underlying transport is gone, nothing can recover. Give up
 * straight away so the host shows the real "session expired" UX
 * instead of a fake "Reconnecting…" badge that's never going to
 * heal. 60 s gives a 2× margin over the daemon-side timeout — long
 * enough to absorb a "phone in pocket for 45 s" case, short enough
 * to be honest with the user.
 */
const MAX_VISIBILITY_DEFER_MS = 60_000;

/**
 * Backoff schedule for the automatic session re-dial (one entry per
 * attempt, in ms of wait BEFORE that attempt). A dead link on a phone
 * usually heals within the first two attempts (WiFi re-associates,
 * 4G comes back); the tail exists for slower recoveries like a router
 * reboot. Total window ≈ 22 s + dial time, capped so a genuinely gone
 * robot surfaces a terminal `sessionStopped` in well under a minute.
 */
const REDIAL_BACKOFF_MS = [0, 2000, 4000, 8000, 8000] as const;

/**
 * Ceiling on a single re-dial attempt (startSession handshake). The
 * signaling round-trip plus ICE + DTLS normally lands in 2-4 s; if an
 * attempt hasn't settled after this long the robot side is likely
 * still holding the previous (dead) session, so we tear the attempt
 * down and let the next backoff slot retry against a freed slot.
 */
const REDIAL_DIAL_TIMEOUT_MS = 15_000;

/**
 * Data-channel silence watchdog. ICE only notices a dead path when
 * STUN consent checks fail, which half-open links (NAT rebind, AP
 * roam onto a blackholing path, asymmetric loss) can dodge for tens
 * of seconds while `iceConnectionState` sits happily on 'connected'.
 * The application layer has a much better liveness signal for free:
 * during any streaming session the daemon talks to us constantly —
 * `get_state` poll replies every ≤500 ms and/or ~30 Hz pushed pose
 * frames. The watchdog measures the time since the LAST inbound
 * data-channel message and escalates in two steps:
 *
 *   - past `DC_SILENCE_NUDGE_MS`, send one extra `get_state` so a
 *     paused poll (fresh pose stream that just died) can't be
 *     mistaken for a dead link;
 *   - past `DC_SILENCE_FATAL_MS` of TOTAL silence, treat the
 *     transport as dead and hand over to the auto re-dial (or emit
 *     the classic fatal `error` when auto-reconnect is off).
 *
 * Hidden tabs throttle timers AND keep receiving messages
 * unpredictably, so silence measured there is meaningless: ticks
 * re-baseline while hidden, and a tick arriving after a large gap
 * (throttled / suspended timer) re-baselines too instead of judging
 * stale data. 8 s of zero inbound traffic on a link that normally
 * delivers 2-30 messages per second is unambiguous, and still ~4×
 * faster than the RFC 7675 consent-freshness teardown (~30 s).
 */
const DC_WATCHDOG_TICK_MS = 1000;
const DC_SILENCE_NUDGE_MS = 2500;
const DC_SILENCE_FATAL_MS = 8000;

/* ─── Dependency surface ───────────────────────────────────────────── */

/**
 * Everything the supervisor needs from `ReachyMini`, as read accessors
 * and side-effect callbacks. Implemented with arrow closures so every
 * call reads the CURRENT class state (tests stub instance methods and
 * the closures pick the stubs up transparently).
 */
export interface SessionSupervisorDeps {
    /** Live ICE state, or `null` when no RTCPeerConnection exists. */
    iceState(): RTCIceConnectionState | null;
    /** True while a live RTCPeerConnection exists. */
    hasPc(): boolean;
    /** True once the session reached `streaming`. */
    isStreaming(): boolean;
    /** True while startSession() is mid-handshake (resolvers armed). */
    isMidSetup(): boolean;
    /** True when the signaling SSE feed is down (`state === 'disconnected'`). */
    isSignalingDown(): boolean;
    /** Robot id of the current / last dialed session. */
    selectedRobotId(): string | null;
    /**
     * Reject the pending startSession() promise, if armed, and clear the
     * resolvers. Returns `true` when one was actually pending — the
     * supervisor then leaves the retry decision to the original caller.
     */
    rejectPendingSession(err: Error): boolean;
    /** Re-establish the signaling feed (the SSE died with the network). */
    reconnectSignaling(): Promise<void>;
    /** One dial attempt: the class's full startSession() flow. */
    dial(robotId: string): Promise<void>;
    /** Transport-only teardown between attempts (owned by the class). */
    teardownForRedial(): void;
    /** Send one `get_state` probe over the data channel. */
    nudgeState(): void;
    emitError(error: Error): void;
    emitReconnecting(detail: { attempt: number; maxAttempts: number; cause: string }): void;
    emitReconnected(detail: { attempt: number }): void;
    emitSessionStopped(detail: { reason: string; message: string }): void;
    emitNetworkOnline(): void;
    emitNetworkOffline(): void;
    emitNetworkChange(detail: {
        effectiveType?: string;
        downlink?: number;
        rtt?: number;
        saveData?: boolean;
    }): void;
}

/* ─── Supervisor ───────────────────────────────────────────────────── */

export class SessionSupervisor {
    private readonly _deps: SessionSupervisorDeps;

    // ICE-blip debounce. All handler slots are scoped to the lifetime
    // of a live session (armed from the pc's state handler, cleared by
    // every teardown path via `clearIceGrace`).
    private _iceGraceTimer: ReturnType<typeof setTimeout> | null = null;
    private _iceGraceReason: 'disconnected' | 'failed' | null = null;
    private _pendingVisibilityHandler: (() => void) | null = null;

    // Network awareness (installed in startSession, removed on teardown).
    private _onlineHandler: (() => void) | null = null;
    private _offlineHandler: (() => void) | null = null;
    private _connectionChangeHandler: (() => void) | null = null;

    // Automatic re-dial. Mutable via `setAutoReconnect()` so flows that
    // EXPECT a transport death (the daemon self-update reboot) can stand
    // the machinery down.
    private _autoReconnect: boolean;
    private _redialing = false;
    private _redialTimer: ReturnType<typeof setTimeout> | null = null;
    private _redialWake: (() => void) | null = null;
    // True while the redial loop itself is inside dial(), so the
    // "external startSession cancels a pending redial" guard can tell
    // the two apart.
    private _redialInternalDial = false;

    // Data-channel silence watchdog.
    private _dcWatchdogId: ReturnType<typeof setInterval> | null = null;
    private _lastDcInboundAt = 0;
    private _dcWatchdogLastTickAt = 0;
    private _dcSilenceNudged = false;

    constructor(deps: SessionSupervisorDeps, options: { autoReconnect: boolean }) {
        this._deps = deps;
        this._autoReconnect = options.autoReconnect;
    }

    /** True while the re-dial loop is running. */
    get redialing(): boolean {
        return this._redialing;
    }

    /**
     * Enable/disable the automatic session re-dial at runtime.
     * Disabling also aborts any re-dial already in flight.
     */
    setAutoReconnect(enabled: boolean): void {
        this._autoReconnect = enabled;
        if (!enabled) this.cancelRedial();
    }

    /**
     * An explicit dial from the app supersedes any in-flight
     * auto-reconnect (possibly towards a different robot). No-op when
     * the caller is the redial loop's own dial attempt.
     */
    noteExternalDial(): void {
        if (this._redialing && !this._redialInternalDial) this.cancelRedial();
    }

    /* ─── ICE-blip debounce + network awareness ──────────────────────
     *
     * Both halves below are intentionally generic (they don't know
     * about motion, audio, or the FSM): they just smooth out
     * browser-level events so the consumer's own state machine doesn't
     * get torn down by routine WiFi/4G/screen-off noise.
     */

    /** ICE healed back to `connected`/`completed`. */
    onIceHealed(): void {
        this.clearIceGrace();
    }

    /**
     * ICE hit `disconnected` — TRANSIENT per spec, debounce before
     * escalating. If the tab is hidden, JS timers are throttled and
     * would fire unpredictably late, so defer the grace window to the
     * next foreground frame.
     */
    onIceDisconnected(): void {
        if (typeof document !== 'undefined' && document.hidden) {
            this._armIceGraceOnVisibility();
        } else {
            this._scheduleIceGrace(ICE_DISCONNECT_GRACE_MS, 'disconnected');
        }
    }

    /**
     * ICE hit `failed` — terminal per spec, but in practice we've seen
     * `failed → connected` on rapid AP roams / BT route changes on iOS.
     * Give the ICE agent a short window to surprise us before rejecting
     * the session.
     */
    onIceFailed(): void {
        this._scheduleIceGrace(ICE_FAILED_GRACE_MS, 'failed');
    }

    /**
     * Cancel any pending ICE grace timer and visibility handler. Called
     * on a healed `connected`/`completed` transition AND from the
     * lifecycle teardown paths so a callback can't fire after the pc
     * is closed.
     */
    clearIceGrace(): void {
        if (this._iceGraceTimer !== null) {
            clearTimeout(this._iceGraceTimer);
            this._iceGraceTimer = null;
        }
        this._iceGraceReason = null;
        if (this._pendingVisibilityHandler && typeof document !== 'undefined') {
            document.removeEventListener('visibilitychange', this._pendingVisibilityHandler);
        }
        this._pendingVisibilityHandler = null;
    }

    /**
     * Start a grace window. After `ms`, re-check the live ICE state:
     *   - If we healed back to `connected`/`completed`, the timer was
     *     already cancelled in the pc's state handler, so we never get
     *     here.
     *   - If we're still in the originally-observed bad state (or
     *     worse), surface the error and reject any pending session
     *     promise. The original code path is preserved verbatim so
     *     downstream consumers see the same `error` payload shape.
     */
    private _scheduleIceGrace(ms: number, reason: 'disconnected' | 'failed'): void {
        // Coalesce: if a grace is already pending and the reason hasn't
        // changed, keep the original timer so a flurry of identical
        // transitions doesn't reset the clock. If the reason changed
        // (typically `disconnected` → `failed`, but also the reverse on
        // some Android WebViews), replace the timer with the new
        // (reason, ms) pair — the latest signal wins.
        if (this._iceGraceTimer !== null) {
            if (this._iceGraceReason === reason) return;
            clearTimeout(this._iceGraceTimer);
        }
        this._iceGraceReason = reason;
        this._iceGraceTimer = setTimeout(() => {
            this._iceGraceTimer = null;
            const r = this._iceGraceReason;
            this._iceGraceReason = null;
            const s = this._deps.iceState();
            if (s === 'connected' || s === 'completed') return; // healed
            if (r === 'disconnected' && s === 'disconnected') {
                if (this.maybeBeginRedial(`ICE stuck in 'disconnected' for > ${ms}ms`)) return;
                this._deps.emitError(new Error(`ICE stuck in 'disconnected' for > ${ms}ms`));
                return;
            }
            if (r === 'failed' || s === 'failed') {
                const err = new Error('ICE connection failed');
                if (this._deps.rejectPendingSession(err)) {
                    // Mid-setup failure: keep the promise contract — the
                    // caller of startSession() owns the retry decision.
                    // When that caller is the redial loop itself, skip
                    // the fatal `error` emit: the loop retries and the
                    // app only sees `sessionReconnecting`.
                    if (!this._redialing) this._deps.emitError(err);
                    return;
                }
                if (this.maybeBeginRedial('ICE connection failed')) return;
                this._deps.emitError(err);
            }
        }, ms);
    }

    /**
     * `disconnected` while the tab is hidden. JS timers are throttled
     * in background tabs (Chrome clamps to ~1 Hz, Safari can pause
     * altogether), so a foreground grace timer would either miss the
     * window or fire long after the connection healed. Wait for the
     * tab to come back, then re-evaluate.
     */
    private _armIceGraceOnVisibility(): void {
        if (this._pendingVisibilityHandler) return;
        const deferredAt = Date.now();
        const handler = (): void => {
            if (typeof document !== 'undefined' && document.hidden) return;
            document.removeEventListener('visibilitychange', handler);
            this._pendingVisibilityHandler = null;
            if (!this._deps.hasPc()) return;
            const s = this._deps.iceState();
            if (s === 'connected' || s === 'completed') return; // healed in bg

            // Ceiling: if the user backgrounded past the daemon's
            // ICE-consent freshness window the session is gone from
            // the daemon's side regardless of what the pc reports
            // locally. Running another foreground grace would tell
            // the user "Reconnecting…" for a recovery that can never
            // happen. Escalate immediately so the host renders the
            // real "session expired" UX. See MAX_VISIBILITY_DEFER_MS.
            if (Date.now() - deferredAt > MAX_VISIBILITY_DEFER_MS) {
                const err = new Error('Session expired while tab was backgrounded');
                if (this._deps.rejectPendingSession(err)) {
                    if (!this._redialing) this._deps.emitError(err);
                    return;
                }
                // The transport is unrecoverable (daemon dropped its side
                // past the consent-freshness window) but the user is BACK
                // and looking at the app — the perfect moment to re-dial
                // rather than render "session expired".
                if (this.maybeBeginRedial('Session expired while tab was backgrounded')) return;
                this._deps.emitError(err);
                return;
            }

            if (s === 'failed') {
                this._scheduleIceGrace(ICE_FAILED_GRACE_MS, 'failed');
                return;
            }
            // Still disconnected when we came back — give it a normal
            // foreground grace window now that timers fire reliably.
            this._scheduleIceGrace(ICE_DISCONNECT_GRACE_MS, 'disconnected');
        };
        document.addEventListener('visibilitychange', handler);
        this._pendingVisibilityHandler = handler;
    }

    /**
     * Install browser-level network listeners and forward them as
     * public `networkOnline` / `networkOffline` / `networkChange`
     * events on the instance. Idempotent: called from `startSession()`,
     * removed by `uninstallNetworkListeners` on teardown. Reachable
     * only when there's a live `window` (defensive guard for SSR /
     * test environments).
     *
     * `online` / `offline` are semantically about CONNECTIVITY:
     * "does the OS think we can reach the internet". They flip
     * symmetrically.
     *
     * `connection.change` (NetworkInformation API, Chrome / Android
     * WebView only) is semantically about the TRANSPORT: it fires
     * on Wi-Fi → 4G swaps, AP roams, etc. without necessarily going
     * through `offline`. We forward it as its own `networkChange`
     * event rather than aliasing it onto `networkOnline`, so
     * consumers don't have to guess whether they're seeing a real
     * connectivity recovery or a silent transport swap.
     */
    installNetworkListeners(): void {
        if (this._onlineHandler || typeof window === 'undefined') return;
        const onOnline = (): void => this._deps.emitNetworkOnline();
        const onOffline = (): void => this._deps.emitNetworkOffline();
        window.addEventListener('online', onOnline);
        window.addEventListener('offline', onOffline);
        this._onlineHandler = onOnline;
        this._offlineHandler = onOffline;

        const conn = (navigator as Navigator & {
            connection?: {
                effectiveType?: string;
                downlink?: number;
                rtt?: number;
                saveData?: boolean;
                addEventListener?: (type: string, listener: () => void) => void;
                removeEventListener?: (type: string, listener: () => void) => void;
            };
        }).connection;
        if (conn && typeof conn.addEventListener === 'function') {
            const onChange = (): void => this._deps.emitNetworkChange({
                effectiveType: conn.effectiveType,
                downlink: conn.downlink,
                rtt: conn.rtt,
                saveData: conn.saveData,
            });
            conn.addEventListener('change', onChange);
            this._connectionChangeHandler = onChange;
        }
    }

    /** Counterpart to `installNetworkListeners`. */
    uninstallNetworkListeners(): void {
        if (typeof window !== 'undefined') {
            if (this._onlineHandler) {
                window.removeEventListener('online', this._onlineHandler);
            }
            if (this._offlineHandler) {
                window.removeEventListener('offline', this._offlineHandler);
            }
        }
        const conn = (navigator as Navigator & {
            connection?: {
                removeEventListener?: (type: string, listener: () => void) => void;
            };
        }).connection;
        if (conn && this._connectionChangeHandler && typeof conn.removeEventListener === 'function') {
            conn.removeEventListener('change', this._connectionChangeHandler);
        }
        this._onlineHandler = null;
        this._offlineHandler = null;
        this._connectionChangeHandler = null;
    }

    /* ─── Automatic session re-dial ─────────────────────────────────── */

    /**
     * Escalation funnel for dead-transport signals. Returns `true` when
     * the failure is being handled by a re-dial (callers should then skip
     * their fatal `error` emit). Only fires for an ESTABLISHED session:
     * mid-setup failures keep rejecting the startSession() promise so the
     * original caller stays in charge of retries.
     */
    maybeBeginRedial(cause: string): boolean {
        if (!this._autoReconnect) return false;
        if (this._redialing) return true; // already in progress
        if (this._deps.isMidSetup()) return false;
        const robotId = this._deps.selectedRobotId();
        if (!robotId || !this._deps.hasPc()) return false;
        this._redialing = true;
        console.info(`[reachy-mini] auto-reconnect: ${cause} — re-dialing ${robotId}`);
        void this._runRedialLoop(robotId, cause);
        return true;
    }

    /**
     * The re-dial loop. One teardown, then up to REDIAL_BACKOFF_MS.length
     * attempts, each preceded by its backoff slot. Every attempt re-runs
     * the normal startSession() flow (and connect() first if the SSE feed
     * died with the network), so a success is indistinguishable from a
     * fresh session: `streaming` re-fires, videoTrack re-attaches, the
     * pose subscription is re-asserted by the session-ready path.
     *
     * Exits early — without emitting anything further — when the user
     * calls stopSession()/disconnect() or dials another robot, all of
     * which flip `_redialing` off via `cancelRedial`.
     */
    private async _runRedialLoop(robotId: string, cause: string): Promise<void> {
        const maxAttempts = REDIAL_BACKOFF_MS.length;
        this._deps.teardownForRedial();
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            if (!this._redialing) return;
            this._deps.emitReconnecting({ attempt, maxAttempts, cause });
            await this._redialSleep(REDIAL_BACKOFF_MS[attempt - 1]!);
            if (!this._redialing) return;
            try {
                if (this._deps.isSignalingDown()) {
                    // The network drop also killed the signaling SSE feed —
                    // re-establish it before dialing.
                    await this._deps.reconnectSignaling();
                }
                if (!this._redialing) return;
                this._redialInternalDial = true;
                try {
                    await withTimeout(
                        this._deps.dial(robotId),
                        REDIAL_DIAL_TIMEOUT_MS,
                        'auto-reconnect dial timed out',
                    );
                } finally {
                    this._redialInternalDial = false;
                }
                if (!this._redialing) return;
                this._redialing = false;
                console.info(`[reachy-mini] auto-reconnect: recovered on attempt ${attempt}`);
                this._deps.emitReconnected({ attempt });
                return;
            } catch (e) {
                console.warn(
                    `[reachy-mini] auto-reconnect attempt ${attempt}/${maxAttempts} failed:`,
                    e,
                );
                // A failed attempt can leave a half-open pc (e.g. the dial
                // timed out mid-ICE). Clean it so the next attempt starts
                // from a blank slate; also frees the robot-side slot via
                // endSession when one was allocated.
                if (this._redialing) this._deps.teardownForRedial();
            }
        }
        if (!this._redialing) return;
        this._redialing = false;
        const message = `Auto-reconnect gave up after ${maxAttempts} attempts (${cause})`;
        console.warn(`[reachy-mini] ${message}`);
        this._deps.emitSessionStopped({ reason: 'reconnect_failed', message });
        this._deps.emitError(new Error(message));
    }

    /** Cancellable backoff sleep — `cancelRedial` wakes it immediately. */
    private _redialSleep(ms: number): Promise<void> {
        if (ms <= 0) return Promise.resolve();
        return new Promise<void>((resolve) => {
            this._redialWake = resolve;
            this._redialTimer = setTimeout(() => {
                this._redialTimer = null;
                this._redialWake = null;
                resolve();
            }, ms);
        });
    }

    /**
     * Abort a pending re-dial. Called from stopSession(), disconnect()
     * and an external startSession() — every path where the user (or the
     * consumer app) takes over the session lifecycle.
     */
    cancelRedial(): void {
        if (!this._redialing && !this._redialWake) return;
        this._redialing = false;
        if (this._redialTimer !== null) {
            clearTimeout(this._redialTimer);
            this._redialTimer = null;
        }
        if (this._redialWake) {
            const wake = this._redialWake;
            this._redialWake = null;
            wake();
        }
    }

    /* ─── Data-channel silence watchdog ─────────────────────────────── */

    /** (Re)start the watchdog. Called when a session reaches ready. */
    startDcWatchdog(): void {
        this.stopDcWatchdog();
        const now = Date.now();
        this._lastDcInboundAt = now;
        this._dcWatchdogLastTickAt = now;
        this._dcSilenceNudged = false;
        this._dcWatchdogId = setInterval(() => this._dcWatchdogTick(), DC_WATCHDOG_TICK_MS);
    }

    stopDcWatchdog(): void {
        if (this._dcWatchdogId !== null) {
            clearInterval(this._dcWatchdogId);
            this._dcWatchdogId = null;
        }
    }

    /**
     * Liveness stamp: every inbound data-channel message — control
     * replies, broadcasts, pose frames — proves the transport is alive.
     */
    stampDcInbound(): void {
        this._lastDcInboundAt = Date.now();
        this._dcSilenceNudged = false;
    }

    private _dcWatchdogTick(): void {
        const now = Date.now();
        const tickGap = now - this._dcWatchdogLastTickAt;
        this._dcWatchdogLastTickAt = now;

        if (!this._deps.isStreaming()) return;

        // A tick that arrives way late means the timer was throttled or
        // suspended (hidden tab, iOS app switch). Whatever silence we
        // measure against that stale baseline is meaningless — and the
        // background-transport question is already owned by the
        // visibility-deferred ICE grace. Re-baseline and judge again
        // from the next healthy tick.
        if (tickGap > DC_WATCHDOG_TICK_MS * 3) {
            this._lastDcInboundAt = now;
            this._dcSilenceNudged = false;
            return;
        }
        if (typeof document !== 'undefined' && document.hidden) {
            this._lastDcInboundAt = now;
            this._dcSilenceNudged = false;
            return;
        }

        const silence = now - this._lastDcInboundAt;
        if (silence < DC_SILENCE_NUDGE_MS) return;

        if (!this._dcSilenceNudged) {
            // One extra get_state so a legitimately quiet channel (pose
            // stream just died, poll on hold) gets a chance to answer
            // before we call the transport dead. Any inbound message
            // clears the flag via `stampDcInbound`.
            this._dcSilenceNudged = true;
            this._deps.nudgeState();
            return;
        }

        if (silence < DC_SILENCE_FATAL_MS) return;

        // Dead transport. Stop judging (the re-dial tears this interval
        // down anyway via the class's teardown; the error path keeps the
        // session nominally up, so re-baseline to avoid a repeat every
        // tick) and escalate through the same funnel as the ICE grace.
        this._lastDcInboundAt = now;
        this._dcSilenceNudged = false;
        const cause = `No data-channel traffic for ${silence}ms`;
        if (this.maybeBeginRedial(cause)) return;
        this._deps.emitError(new Error(cause));
    }
}

/** Reject `p` with `label` if it hasn't settled within `ms`. */
function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
    return new Promise<T>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(label)), ms);
        p.then(
            (v) => { clearTimeout(timer); resolve(v); },
            (e) => { clearTimeout(timer); reject(e); },
        );
    });
}
