/**
 * ReachyMini — Browser SDK for controlling a Reachy Mini robot over WebRTC.
 * See `../reachy-mini-sdk.ts` for the package's barrel and the README for
 * a quick-start guide.
 */

import {
    oauthHandleRedirectIfPresent,
    oauthLoginUrl,
} from '@huggingface/hub';

import { degToRad, rpyToMatrix } from './math.js';
import {
    consumeFragmentCredentials,
    readPreselectedRobotIdFromUrl,
    sdpHasAudioSendRecv,
} from './url-helpers.js';
import {
    UPLOAD_CHUNK_SIZE,
    UPLOAD_BUFFERED_HIGH_WATER,
    UPLOAD_BUFFERED_LOW_WATER,
    hasCompressionStream,
    makeUploadId,
    bytesToBase64,
    gzipBase64,
    clampVolume,
    audioUploadEncoding,
} from './upload-helpers.js';
import type {
    ApplyAudioConfigOptions,
    AudioConfigEntry,
    AutoConnectOptions,
    AutoConnectResult,
    AutoConnectRobotChoice,
    FaceTarget,
    MotionAwaitOptions,
    MoveData,
    PlayMoveOptions,
    PlayMoveResult,
    PlayUploadedAudioOptions,
    ReachyMiniEventMap,
    ReachyMiniInstance,
    ReachyMiniOptions,
    RobotInfo,
    RobotState,
    SessionRejectError,
    SubscribeLogsOptions,
    UploadAudioOptions,
} from './types.js';

interface PendingMotion {
    resolve: () => void;
    reject: (err: Error) => void;
    timer: ReturnType<typeof setTimeout>;
}

interface BroadcastWaiter {
    predicate: (m: Record<string, unknown>) => boolean;
    resolve: (m: Record<string, unknown>) => void;
    timer: ReturnType<typeof setTimeout>;
}

interface LogSubscriber {
    onLine: (entry: { timestamp: string; line: string }) => void;
    onError?: (error: string) => void;
}

/** One progress event from an in-flight `startDaemonUpdate()`. */
export interface UpdateProgressEvent {
    status: 'in_progress' | 'done' | 'failed';
    /** A log line of the underlying update job (when `in_progress`). */
    line?: string;
    /** Set when `status === 'failed'`. */
    error?: string;
}

type UpdateProgressCallback = (event: UpdateProgressEvent) => void;

interface SignalingMessage {
    type?: string;
    sessionId?: string;
    peerId?: string;
    producers?: RobotInfo[];
    reason?: string;
    activeApp?: string | null;
    sdp?: { type: RTCSdpType; sdp: string };
    ice?: {
        candidate: string;
        sdpMLineIndex: number | null;
        sdpMid: string | null;
    };
    [key: string]: unknown;
}

/** Mimics the `@huggingface/hub` OAuth callback result we care about. */
interface OAuthRedirectResult {
    accessToken: string;
    accessTokenExpiresAt: Date | string;
    userInfo: { preferred_username?: string; name?: string };
}

// ─── Internal constants ──────────────────────────────────────────────────────

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
 * Ceiling on how long we'll keep `_armIceGraceOnVisibility` waiting
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

export class ReachyMini extends EventTarget implements ReachyMiniInstance {

    // ─── Config ──────────────────────────────────────────────────────────
    private readonly _signalingUrl: string;
    private readonly _clientId: string | null;
    private readonly _appName: string;
    private readonly _videoJitterBufferTargetMs: number;
    private _autoStartFromUrl: boolean;
    private _autoStartAttempted: boolean;

    // ─── Public-ish state mirrors ────────────────────────────────────────
    private _state: 'disconnected' | 'connected' | 'streaming' = 'disconnected';
    private _robots: RobotInfo[] = [];
    private _robotState: RobotState = {};
    private readonly _preselectedRobotId: string | null;

    // ─── Auth ────────────────────────────────────────────────────────────
    private _token: string | null = null;
    private _username: string | null = null;
    private _tokenExpires: string | Date | null = null;

    // ─── Signaling ───────────────────────────────────────────────────────
    private _sseAbortController: AbortController | null = null;

    // ─── WebRTC ──────────────────────────────────────────────────────────
    _pc: RTCPeerConnection | null = null;
    private _dc: RTCDataChannel | null = null;
    private _sessionId: string | null = null;
    private _selectedRobotId: string | null = null;
    private _pendingRemoteIce: NonNullable<SignalingMessage['ice']>[] = [];

    // ─── Audio ───────────────────────────────────────────────────────────
    _micStream: MediaStream | null = null;
    private _micMuted = true;
    private _audioMuted = true;
    private _micSupported = false;

    // ─── Timers ──────────────────────────────────────────────────────────
    private _latencyMonitorId: ReturnType<typeof setInterval> | null = null;
    private _stateRefreshInterval: ReturnType<typeof setInterval> | null = null;

    // ─── Single-slot promise resolvers ───────────────────────────────────
    private _versionResolve: ((v: string | null) => void) | null = null;
    private _hardwareIdResolve: ((v: string | null) => void) | null = null;
    private _volumeResolve: ((v: number | null) => void) | null = null;
    private _micVolumeResolve: ((v: number | null) => void) | null = null;
    private _trackedFaceResolve: ((v: FaceTarget | null) => void) | null = null;
    // applyAudioConfig() / readAudioParameter() share the same single-slot
    // pattern as the volume helpers. Separate slots so the two can be
    // in-flight concurrently without collision.
    private _applyAudioConfigResolve: ((v: boolean | null) => void) | null = null;
    private _readAudioParameterResolve: ((v: number[] | null) => void) | null = null;

    // ─── Log subscribers ─────────────────────────────────────────────────
    private readonly _logSubscribers: Set<LogSubscriber> = new Set();
    private readonly _updateProgressSubscribers: Set<UpdateProgressCallback> = new Set();

    // ─── JSON-RPC app control (over the same DataChannel) ────────────────
    // rpcCall() sends {jsonrpc,id,method,params} and awaits the matching
    // response; onNotification() subscribes to one-way events (no id) the
    // robot/app pushes (conversation.phase/turn/transcript, ...).
    private _rpcCounter = 0;
    private readonly _pendingRpc = new Map<
        string,
        { resolve: (v: unknown) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> }
    >();
    private readonly _rpcListeners = new Map<string, Set<(params: Record<string, unknown>) => void>>();

    // ─── Broadcast waiters (playMove / playUploadedAudio) ────────────────
    private _broadcastWaiters: BroadcastWaiter[] = [];

    // ─── Active upload ids for no-arg cancels ────────────────────────────
    private _activeMoveUploadId: string | null = null;
    private _activeAudioUploadId: string | null = null;

    // ─── Session promise plumbing ────────────────────────────────────────
    private _sessionResolve: (() => void) | null = null;
    private _sessionReject: ((err: Error) => void) | null = null;
    private _iceConnected = false;
    private _dcOpen = false;

    // ─── Resilience: ICE-blip debounce + network awareness ──────────────
    // Backs `_scheduleIceGrace` / `_armIceGraceOnVisibility` and the
    // `networkOnline` / `networkOffline` / `networkChange` forwarders.
    // All three handler slots are scoped to the lifetime of a live
    // session (installed in `startSession`, cleared in
    // `stopSession` / `disconnect` / `_handleEndSession` /
    // `_failSessionRejected`).
    private _iceGraceTimer: ReturnType<typeof setTimeout> | null = null;
    private _iceGraceReason: 'disconnected' | 'failed' | null = null;
    private _pendingVisibilityHandler: (() => void) | null = null;
    private _onlineHandler: (() => void) | null = null;
    private _offlineHandler: (() => void) | null = null;
    private _connectionChangeHandler: (() => void) | null = null;

    // ─── Motion completion plumbing (wake_up / goto_sleep) ───────────────
    private readonly _pendingMotionCompletions: Record<'wake_up' | 'goto_sleep', PendingMotion[]> = {
        wake_up: [],
        goto_sleep: [],
    };

    // ─── Video element ───────────────────────────────────────────────────
    private _videoElement: HTMLVideoElement | null = null;

    constructor(options: ReachyMiniOptions = {}) {
        super();
        this._signalingUrl = options.signalingUrl || 'https://pollen-robotics-reachy-mini-central.hf.space';
        // `enableMicrophone` is intentionally NOT stored: the SDK no longer
        // calls getUserMedia (see startSession). Apps that still pass it for
        // backward compatibility have their value silently ignored — matches
        // the @deprecated annotation on the option type.
        this._clientId = options.clientId || null;
        this._appName = options.appName || 'unknown';
        this._videoJitterBufferTargetMs = options.videoJitterBufferTargetMs ?? 0;
        this._autoStartFromUrl = options.autoStartFromUrl === true;
        this._autoStartAttempted = false;
        this._preselectedRobotId = readPreselectedRobotIdFromUrl();
    }

    // ─── Read-only properties ────────────────────────────────────────────

    get state(): 'disconnected' | 'connected' | 'streaming' { return this._state; }
    get robots(): RobotInfo[] { return this._robots; }
    get robotState(): RobotState { return this._robotState; }
    get username(): string | null { return this._username; }
    get isAuthenticated(): boolean { return !!this._token; }
    get micSupported(): boolean { return this._micSupported; }
    get micMuted(): boolean { return this._micMuted; }
    get audioMuted(): boolean { return this._audioMuted; }
    get preselectedRobotId(): string | null { return this._preselectedRobotId; }
    get isEmbedded(): boolean { return this._preselectedRobotId !== null; }

    /**
     * Internal: try to honour the `autoStartFromUrl` constructor
     * option. Called from the signaling-message handler after every
     * `robotsChanged` emit, so a robot that comes online after the
     * SDK is already `connected` still triggers the auto-start.
     */
    _maybeAutoStart(): void {
        if (!this._autoStartFromUrl) return;
        if (this._autoStartAttempted) return;
        if (!this._preselectedRobotId) return;
        if (this._state !== 'connected') return;
        const match = this._robots.find((r) => r.id === this._preselectedRobotId);
        if (!match) return;
        this._autoStartAttempted = true;
        const peerId = this._preselectedRobotId;
        setTimeout(() => {
            if (this._state !== 'connected') return;
            this.startSession(peerId).catch((err) => {
                console.warn('[reachy-mini] autoStartFromUrl: startSession rejected:', err);
            });
        }, 0);
    }

    // ─── Auth ────────────────────────────────────────────────────────────

    async authenticate(): Promise<boolean> {
        try {
            consumeFragmentCredentials();

            const result = (await oauthHandleRedirectIfPresent()) as OAuthRedirectResult | false | null;
            if (result) {
                this._username = result.userInfo.preferred_username || result.userInfo.name || null;
                this._token = result.accessToken;
                this._tokenExpires = result.accessTokenExpiresAt;
                sessionStorage.setItem('hf_token', this._token);
                sessionStorage.setItem('hf_username', this._username ?? '');
                sessionStorage.setItem(
                    'hf_token_expires',
                    typeof this._tokenExpires === 'string'
                        ? this._tokenExpires
                        : this._tokenExpires.toISOString(),
                );
                return true;
            }

            const t = sessionStorage.getItem('hf_token');
            const u = sessionStorage.getItem('hf_username');
            const e = sessionStorage.getItem('hf_token_expires');
            if (t && u && e && new Date(e) > new Date()) {
                this._token = t;
                this._username = u;
                this._tokenExpires = e;
                return true;
            }
            return false;
        } catch (e) {
            console.error('Auth error:', e);
            return false;
        }
    }

    async login(): Promise<void> {
        const opts: { clientId?: string } = {};
        if (this._clientId) opts.clientId = this._clientId;
        window.location.href = await oauthLoginUrl(opts);
    }

    logout(): void {
        sessionStorage.removeItem('hf_token');
        sessionStorage.removeItem('hf_username');
        sessionStorage.removeItem('hf_token_expires');
        this._username = null;
        this._tokenExpires = null;
        this.disconnect();
    }

    // ─── Lifecycle ───────────────────────────────────────────────────────

    async connect(token?: string): Promise<void> {
        if (this._state !== 'disconnected') throw new Error('Already connected');
        if (token) this._token = token;
        if (!this._token) throw new Error('No token — call authenticate() first or pass a token');
        this._sseAbortController = new AbortController();

        let res: Response;
        try {
            res = await fetch(
                `${this._signalingUrl}/events`,
                {
                    signal: this._sseAbortController.signal,
                    headers: { 'Authorization': `Bearer ${this._token}` },
                },
            );
        } catch (e) {
            this._sseAbortController = null;
            throw e;
        }
        if (!res.ok) {
            this._sseAbortController = null;
            throw new Error(`HTTP ${res.status}`);
        }

        return new Promise<void>((resolve, reject) => {
            let welcomed = false;
            const reader = res.body!.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            const readLoop = async (): Promise<void> => {
                try {
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop() ?? '';
                        for (const line of lines) {
                            if (!line.startsWith('data:')) continue;
                            try {
                                const msg = JSON.parse(line.slice(5).trim()) as SignalingMessage;
                                if (!welcomed && msg.type === 'welcome' && msg.peerId) {
                                    welcomed = true;
                                    this._state = 'connected';
                                    await this._sendToServer({
                                        type: 'setPeerStatus',
                                        roles: ['listener'],
                                        meta: { name: this._appName },
                                    });
                                    this._emit('connected', { peerId: msg.peerId });
                                    resolve();
                                }
                                this._handleSignalingMessage(msg);
                            } catch { /* malformed JSON — skip */ }
                        }
                    }
                } catch (e) {
                    if ((e as Error).name !== 'AbortError') {
                        this._emit('error', { source: 'signaling', error: e as Error });
                    }
                    if (!welcomed) { reject(e as Error); return; }
                }
                if (this._state !== 'disconnected') {
                    this._state = 'disconnected';
                    this._emit('disconnected', { reason: 'SSE closed' });
                }
                if (!welcomed) reject(new Error('Connection closed before welcome'));
            };

            readLoop();
        });
    }

    async autoConnect(options: AutoConnectOptions = {}): Promise<AutoConnectResult> {
        const {
            token,
            pickRobot,
            autoPickIfSingle = true,
            filterBusy = true,
            wakeOnConnect = true,
        } = options;

        if (this._state === 'streaming') {
            const cur = this._robots?.find((r) => r.id === this._selectedRobotId);
            return {
                robotId: this._selectedRobotId!,
                robotName: cur?.meta?.name ?? null,
                isEmbedded: this.isEmbedded,
                alreadyStreaming: true,
            };
        }

        const _prevAutoStartFromUrl = this._autoStartFromUrl;
        this._autoStartFromUrl = false;

        try {
            if (token) {
                this._token = token;
            } else if (!this._token) {
                const ok = await this.authenticate();
                if (!ok) {
                    throw new Error('Not authenticated — call login() or pass a token');
                }
            }

            if (this._state === 'disconnected') {
                await this.connect();
            }

            let robotId: string;
            let robotName: string | null = null;
            if (this.isEmbedded) {
                robotId = this._preselectedRobotId!;
                try {
                    await this._waitForRobotInList(robotId, 5000);
                } catch { /* fall through */ }
                const found = this._robots?.find((r) => r.id === robotId);
                robotName = found?.meta?.name ?? null;
            } else {
                const robots = await this._fetchOwnedRobots({ filterBusy });
                if (robots.length === 0) {
                    throw new Error('No reachable robots');
                }
                if (autoPickIfSingle && robots.length === 1 && !robots[0]!.busy) {
                    robotId = robots[0]!.id;
                    robotName = robots[0]!.name;
                } else if (pickRobot) {
                    const picked = await pickRobot(robots);
                    if (!picked) throw new Error('Robot selection cancelled');
                    robotId = picked;
                    robotName = robots.find((r) => r.id === picked)?.name ?? null;
                } else {
                    throw new Error(
                        'Multiple robots available — pass a pickRobot callback to autoConnect()',
                    );
                }
            }

            await this.startSession(robotId);

            if (wakeOnConnect && typeof this.ensureAwake === 'function') {
                try { await this.ensureAwake(); }
                catch (e) { console.warn('[reachy-mini] autoConnect: ensureAwake failed:', e); }
            }

            return { robotId, robotName, isEmbedded: this.isEmbedded };
        } finally {
            this._autoStartFromUrl = _prevAutoStartFromUrl;
        }
    }

    private async _fetchOwnedRobots(
        { filterBusy = true }: { filterBusy?: boolean } = {},
    ): Promise<AutoConnectRobotChoice[]> {
        try {
            const res = await fetch(`${this._signalingUrl}/api/robot-status`, {
                headers: { 'Authorization': `Bearer ${this._token}` },
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const json = await res.json() as {
                robots?: Array<{
                    peerId: string;
                    robotName?: string | null;
                    meta?: { name?: string; install_id?: string; hardware_id?: string };
                    busy?: boolean;
                    activeApp?: string | null;
                    last_seen_age_seconds?: number | null;
                }>;
            };
            const seen = new Map<string, AutoConnectRobotChoice>();
            for (const r of (json.robots || [])) {
                if (filterBusy && r.busy) continue;
                const key = r.meta?.install_id ?? r.meta?.hardware_id ?? r.peerId;
                seen.set(key, {
                    id: r.peerId,
                    name: r.robotName ?? r.meta?.name ?? null,
                    busy: !!r.busy,
                    activeApp: r.activeApp ?? null,
                    meta: (r.meta ?? {}) as Record<string, unknown>,
                    lastSeenAgeSeconds: r.last_seen_age_seconds ?? null,
                });
            }
            return Array.from(seen.values()).sort(
                (a, b) => (a.lastSeenAgeSeconds ?? Infinity) - (b.lastSeenAgeSeconds ?? Infinity),
            );
        } catch (e) {
            console.warn('[reachy-mini] /api/robot-status unavailable, using SSE list:', e);
            return (this._robots || []).map((r) => ({
                id: r.id,
                name: r.meta?.name ?? null,
                busy: false,
                activeApp: null,
                meta: (r.meta ?? {}) as Record<string, unknown>,
                lastSeenAgeSeconds: null,
            }));
        }
    }

    private _waitForRobotInList(robotId: string, timeoutMs: number): Promise<void> {
        if (this._robots?.find((r) => r.id === robotId)) return Promise.resolve();
        return new Promise<void>((resolve, reject) => {
            const onChange = (): void => {
                if (this._robots?.find((r) => r.id === robotId)) {
                    this.removeEventListener('robotsChanged', onChange);
                    clearTimeout(timeoutId);
                    resolve();
                }
            };
            const timeoutId = setTimeout(() => {
                this.removeEventListener('robotsChanged', onChange);
                reject(new Error(`Timeout waiting for robot ${robotId} in list`));
            }, timeoutMs);
            this.addEventListener('robotsChanged', onChange);
        });
    }

    async startSession(robotId: string): Promise<void> {
        if (this._state !== 'connected') throw new Error('Not connected');
        this._selectedRobotId = robotId;
        this._iceConnected = false;
        this._dcOpen = false;
        this._micSupported = false;
        this._pendingRemoteIce = [];

        // Silent placeholder audio track for the WebRTC audio sender.
        // The SDK does NOT call navigator.mediaDevices.getUserMedia — the
        // user's microphone is the app's responsibility. WebRTC needs a
        // sendrecv audio sender for robot-speaker output to work, so we
        // always set up a 0-gain oscillator → MediaStreamDestination as
        // the initial track. Apps that want to send actual audio (TTS,
        // prerecorded files, the user's mic for teleop, …) do so by
        // calling sender.replaceTrack() on the audio sender exposed via
        // this._pc after the `streaming` event fires.
        try {
            // Safari (and the iOS WKWebView Tauri ships on) exposes
            // AudioContext only under the `webkitAudioContext` prefix.
            // Narrow once, locally, so we don't sprinkle vendor casts
            // through the code.
            const w = window as Window & {
                AudioContext?: typeof AudioContext;
                webkitAudioContext?: typeof AudioContext;
            };
            const Ctx = w.AudioContext ?? w.webkitAudioContext;
            if (!Ctx) throw new Error('AudioContext not supported');
            const ctx = new Ctx();
            const dst = ctx.createMediaStreamDestination();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            gain.gain.value = 0;
            osc.connect(gain).connect(dst);
            osc.start();
            const stream = dst.stream;
            stream.getAudioTracks().forEach((t) => { t.enabled = false; });
            this._micStream = stream;
            this._micMuted = true;
        } catch (e) {
            console.warn('Audio sender placeholder setup failed:', e);
            this._micStream = null;
        }

        this._pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] satisfies RTCIceServer[],
        });

        // Scope `networkOnline` / `networkOffline` / `networkChange`
        // event forwarding to the lifetime of this session.
        this._installNetworkListeners();

        return new Promise<void>((resolve, reject) => {
            this._sessionResolve = resolve;
            this._sessionReject = reject;

            this._pc!.ontrack = (e) => {
                if (e.track.kind === 'video') {
                    const ms = this._videoJitterBufferTargetMs;
                    try {
                        (e.receiver as RTCRtpReceiver & { jitterBufferTarget?: number })
                            .jitterBufferTarget = ms;
                    } catch { /* ignore */ }
                    try {
                        (e.receiver as RTCRtpReceiver & { playoutDelayHint?: number })
                            .playoutDelayHint = ms / 1000;
                    } catch { /* ignore */ }
                    this._emit('videoTrack', { track: e.track, stream: e.streams[0]! });
                }
            };

            this._pc!.onicecandidate = async (e) => {
                if (e.candidate && this._sessionId) {
                    await this._sendToServer({
                        type: 'peer',
                        sessionId: this._sessionId,
                        ice: {
                            candidate: e.candidate.candidate,
                            sdpMLineIndex: e.candidate.sdpMLineIndex,
                            sdpMid: e.candidate.sdpMid,
                        },
                    });
                }
            };

            this._pc!.oniceconnectionstatechange = () => {
                const s = this._pc?.iceConnectionState;
                if (!s) return;
                // Public, granular event: every transition is visible to
                // consumers so they can render finer UX (e.g. a transient
                // "Reconnecting…" badge during `disconnected`) without
                // having to attach their own handler to `_pc`.
                this._emit('iceStateChange', { state: s });

                if (s === 'connected' || s === 'completed') {
                    // Healed — cancel any pending grace from a previous blip.
                    this._clearIceGrace();
                    this._iceConnected = true;
                    this._checkSessionReady();
                    return;
                }
                if (s === 'disconnected') {
                    // TRANSIENT per spec — debounce before escalating.
                    // If the tab is hidden, JS timers are throttled and
                    // would fire unpredictably late, so defer the grace
                    // window to the next foreground frame.
                    if (typeof document !== 'undefined' && document.hidden) {
                        this._armIceGraceOnVisibility();
                    } else {
                        this._scheduleIceGrace(ICE_DISCONNECT_GRACE_MS, 'disconnected');
                    }
                    return;
                }
                if (s === 'failed') {
                    // Terminal per spec, but in practice we've seen
                    // `failed → connected` on rapid AP roams / BT route
                    // changes on iOS. Give the ICE agent a short window
                    // to surprise us before rejecting the session.
                    this._scheduleIceGrace(ICE_FAILED_GRACE_MS, 'failed');
                    return;
                }
            };

            this._pc!.ondatachannel = (e) => {
                this._dc = e.channel;
                this._dc.onopen = () => {
                    this._dcOpen = true;
                    this._checkSessionReady();
                };
                this._dc.onmessage = (ev) => this._handleRobotMessage(JSON.parse(ev.data));
            };

            this._sendToServer({ type: 'startSession', peerId: robotId }).then((r) => {
                if (r?.type === 'sessionRejected') {
                    this._failSessionRejected(r);
                    return;
                }
                if (r?.sessionId) this._sessionId = r.sessionId;
            });
        });
    }

    private _failSessionRejected(msg: SignalingMessage): void {
        const err = new Error(
            msg.reason === 'robot_busy'
                ? `Robot is busy: "${msg.activeApp || 'another app'}" is already connected`
                : `Session rejected: ${msg.reason || 'unknown reason'}`,
        ) as SessionRejectError;
        err.reason = msg.reason ?? null;
        err.activeApp = msg.activeApp ?? null;

        // Resilience teardown BEFORE closing `_pc` so a queued grace
        // callback can't dereference a dead handle.
        this._clearIceGrace();
        this._uninstallNetworkListeners();
        if (this._pc) { this._pc.close(); this._pc = null; }
        if (this._micStream) { this._micStream.getTracks().forEach((t) => t.stop()); this._micStream = null; }
        this._iceConnected = false;
        this._dcOpen = false;
        this._micMuted = true;
        this._micSupported = false;

        this._emit('sessionRejected', { reason: msg.reason, activeApp: msg.activeApp });

        if (this._sessionReject) {
            const reject = this._sessionReject;
            this._sessionResolve = null;
            this._sessionReject = null;
            reject(err);
        }
    }

    async stopSession(): Promise<void> {
        if (this._versionResolve) { this._versionResolve(null); this._versionResolve = null; }
        if (this._hardwareIdResolve) { this._hardwareIdResolve(null); this._hardwareIdResolve = null; }
        if (this._volumeResolve) { this._volumeResolve(null); this._volumeResolve = null; }
        if (this._micVolumeResolve) { this._micVolumeResolve(null); this._micVolumeResolve = null; }
        if (this._trackedFaceResolve) { this._trackedFaceResolve(null); this._trackedFaceResolve = null; }
        if (this._applyAudioConfigResolve) { this._applyAudioConfigResolve(false); this._applyAudioConfigResolve = null; }
        if (this._readAudioParameterResolve) { this._readAudioParameterResolve(null); this._readAudioParameterResolve = null; }
        this._logSubscribers.clear();
        this._updateProgressSubscribers.clear();
        this._rejectPendingMotionCompletions(new Error('Session stopped'));
        this._rejectPendingRpc(new Error('Session stopped'));
        // Tear down resilience plumbing BEFORE closing `_pc` so a
        // queued grace callback can't dereference a dead handle.
        this._clearIceGrace();
        this._uninstallNetworkListeners();
        if (this._sessionReject) {
            this._sessionReject(new Error('Session stopped'));
            this._sessionResolve = null;
            this._sessionReject = null;
        }

        if (this._stateRefreshInterval) { clearInterval(this._stateRefreshInterval); this._stateRefreshInterval = null; }
        if (this._latencyMonitorId) { clearInterval(this._latencyMonitorId); this._latencyMonitorId = null; }

        if (this._sessionId) {
            await this._sendToServer({ type: 'endSession', sessionId: this._sessionId });
        }

        if (this._micStream) { this._micStream.getTracks().forEach((t) => t.stop()); this._micStream = null; }
        this._micMuted = true;
        this._micSupported = false;

        if (this._pc) { this._pc.close(); this._pc = null; }
        if (this._dc) { this._dc.close(); this._dc = null; }

        this._sessionId = null;
        this._iceConnected = false;
        this._dcOpen = false;

        const wasStreaming = this._state === 'streaming';
        if (wasStreaming) {
            this._state = 'connected';
            this._emit('sessionStopped', { reason: 'user' });
        }
    }

    disconnect(): void {
        if (this._sseAbortController) { this._sseAbortController.abort(); this._sseAbortController = null; }

        if (this._versionResolve) { this._versionResolve(null); this._versionResolve = null; }
        if (this._hardwareIdResolve) { this._hardwareIdResolve(null); this._hardwareIdResolve = null; }
        if (this._volumeResolve) { this._volumeResolve(null); this._volumeResolve = null; }
        if (this._micVolumeResolve) { this._micVolumeResolve(null); this._micVolumeResolve = null; }
        if (this._trackedFaceResolve) { this._trackedFaceResolve(null); this._trackedFaceResolve = null; }
        if (this._applyAudioConfigResolve) { this._applyAudioConfigResolve(false); this._applyAudioConfigResolve = null; }
        if (this._readAudioParameterResolve) { this._readAudioParameterResolve(null); this._readAudioParameterResolve = null; }
        this._logSubscribers.clear();
        this._updateProgressSubscribers.clear();
        this._rejectPendingMotionCompletions(new Error('Disconnected'));
        // Mirrors the resilience teardown in `stopSession()`.
        this._clearIceGrace();
        this._uninstallNetworkListeners();
        if (this._sessionReject) {
            this._sessionReject(new Error('Disconnected'));
            this._sessionResolve = null;
            this._sessionReject = null;
        }

        if (this._stateRefreshInterval) { clearInterval(this._stateRefreshInterval); this._stateRefreshInterval = null; }
        if (this._latencyMonitorId) { clearInterval(this._latencyMonitorId); this._latencyMonitorId = null; }

        if (this._sessionId && this._token) {
            this._sendToServer({ type: 'endSession', sessionId: this._sessionId });
        }

        if (this._micStream) { this._micStream.getTracks().forEach((t) => t.stop()); this._micStream = null; }
        if (this._pc) { this._pc.close(); this._pc = null; }
        if (this._dc) { this._dc.close(); this._dc = null; }

        this._sessionId = null;
        this._micMuted = true;
        this._micSupported = false;
        this._iceConnected = false;
        this._dcOpen = false;
        this._robots = [];
        this._state = 'disconnected';
        this._emit('disconnected', { reason: 'user' });
    }

    // ─── Resilience: ICE-blip debounce + network awareness ───────────────
    //
    // Both halves below are intentionally generic (they don't know about
    // motion, audio, or the FSM): they just smooth out browser-level
    // events so the consumer's own state machine doesn't get torn down
    // by routine WiFi/4G/screen-off noise.

    /**
     * Cancel any pending ICE grace timer and visibility handler. Called
     * on a healed `connected`/`completed` transition AND from the
     * lifecycle teardown paths so a callback can't fire after `_pc`
     * is closed.
     */
    private _clearIceGrace(): void {
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
     *     already cancelled in `oniceconnectionstatechange`, so we
     *     never get here.
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
            const s = this._pc?.iceConnectionState;
            if (s === 'connected' || s === 'completed') return; // healed
            if (r === 'disconnected' && s === 'disconnected') {
                this._emit('error', {
                    source: 'webrtc',
                    error: new Error(`ICE stuck in 'disconnected' for > ${ms}ms`),
                });
                return;
            }
            if (r === 'failed' || s === 'failed') {
                const err = new Error('ICE connection failed');
                if (this._sessionReject) {
                    this._sessionReject(err);
                    this._sessionResolve = null;
                    this._sessionReject = null;
                }
                this._emit('error', { source: 'webrtc', error: err });
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
            if (!this._pc) return;
            const s = this._pc.iceConnectionState;
            if (s === 'connected' || s === 'completed') return; // healed in bg

            // Ceiling: if the user backgrounded past the daemon's
            // ICE-consent freshness window the session is gone from
            // the daemon's side regardless of what `_pc` reports
            // locally. Running another foreground grace would tell
            // the user "Reconnecting…" for a recovery that can never
            // happen. Escalate immediately so the host renders the
            // real "session expired" UX. See MAX_VISIBILITY_DEFER_MS.
            if (Date.now() - deferredAt > MAX_VISIBILITY_DEFER_MS) {
                const err = new Error(
                    'Session expired while tab was backgrounded',
                );
                if (this._sessionReject) {
                    this._sessionReject(err);
                    this._sessionResolve = null;
                    this._sessionReject = null;
                }
                this._emit('error', { source: 'webrtc', error: err });
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
     * events on this instance. Idempotent: called from
     * `startSession()`, removed by `_uninstallNetworkListeners` on
     * teardown. Reachable only when there's a live `window`
     * (defensive guard for SSR / test environments).
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
    private _installNetworkListeners(): void {
        if (this._onlineHandler || typeof window === 'undefined') return;
        const onOnline = (): void => this._emit('networkOnline', {});
        const onOffline = (): void => this._emit('networkOffline', {});
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
            const onChange = (): void => this._emit('networkChange', {
                effectiveType: conn.effectiveType,
                downlink: conn.downlink,
                rtt: conn.rtt,
                saveData: conn.saveData,
            });
            conn.addEventListener('change', onChange);
            this._connectionChangeHandler = onChange;
        }
    }

    /** Counterpart to `_installNetworkListeners`. */
    private _uninstallNetworkListeners(): void {
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

    // ─── Commands ────────────────────────────────────────────────────────

    /**
     * Atomic raw-units pose update over the data channel. Channels you
     * omit are held at their last commanded value (per-axis, independent).
     *
     * **Head pose is in the WORLD frame.** The daemon's IK splits the
     * requested head world-yaw between body rotation and the stewart
     * platform, subject to the mechanical limit
     * `|head_yaw_world − body_yaw| ≤ 65°`.
     *
     * **If you want the head to FOLLOW the body** (tank-style rotation):
     * a `setTarget({ body_yaw })` on its own does NOT rotate the head —
     * the head's commanded world yaw is unchanged, so its gaze stays
     * pinned in world frame while the body turns under it. To make the
     * head turn with the body, include a `head` matrix in the SAME call
     * with the body-yaw delta added to the head RPY's yaw:
     *
     * ```ts
     * // Body-yaw drag handler: tank-couple the head so it follows.
     * const delta = newBodyDeg - lastCommandedBodyDeg;
     * const nextHeadYaw = lastCommandedHeadYawDeg + delta;
     * robot.setTarget({
     *   head: rpyToMatrix(headRoll, headPitch, nextHeadYaw).flat(),
     *   body_yaw: degToRad(newBodyDeg),
     * });
     * lastCommandedHeadYawDeg = nextHeadYaw;
     * lastCommandedBodyDeg    = newBodyDeg;
     * ```
     *
     * **Baseline must be the last COMMANDED value, not telemetry.** For
     * continuous-input controllers (slider drag, joystick), do not use
     * `state.head` from the `state` event as the baseline for incremental
     * commands — telemetry lags one WebRTC round-trip, so cumulative
     * deltas computed against it stall (every iteration in a rapid drag
     * adds the same `delta` to the same stale baseline → the head fails
     * to keep up). Track the last-commanded RPY in your own buffer.
     *
     * @param head      Flat row-major 4×4 matrix (16 finite numbers) in
     *                  the world frame. Omit to hold the previous head target.
     * @param antennas  `[rightRad, leftRad]` (radians). Omit to hold.
     * @param body_yaw  Signed radians. Omit to hold.
     * @returns `true` if the command was queued on the data channel,
     *          `false` if the channel is not open.
     */
    setTarget(
        { head, antennas, body_yaw }: { head?: number[]; antennas?: number[]; body_yaw?: number } = {},
    ): boolean {
        const cmd: Record<string, unknown> = { type: 'set_full_target' };
        if (head !== undefined) {
            if (!Array.isArray(head) || head.length !== 16
                || !head.every((n) => Number.isFinite(n))) {
                throw new TypeError(
                    'setTarget: head must be a 16-element flat row-major 4×4 matrix '
                    + `of finite numbers; got ${Array.isArray(head) ? `Array(${head.length})` : typeof head}`,
                );
            }
            cmd.head = head;
        }
        if (antennas !== undefined) {
            if (!Array.isArray(antennas) || antennas.length !== 2
                || !antennas.every((n) => Number.isFinite(n))) {
                throw new TypeError(
                    'setTarget: antennas must be [rightRad, leftRad] (2 finite numbers); '
                    + `got ${Array.isArray(antennas) ? `Array(${antennas.length})` : typeof antennas}`,
                );
            }
            cmd.antennas = antennas;
        }
        if (body_yaw !== undefined) {
            if (!Number.isFinite(body_yaw)) {
                throw new TypeError(
                    `setTarget: body_yaw must be a finite number (radians); got ${body_yaw}`,
                );
            }
            cmd.body_yaw = body_yaw;
        }
        return this._sendCommand(cmd);
    }

    gotoTarget(
        { head, antennas, body_yaw, duration }:
            { head?: number[]; antennas?: number[]; body_yaw?: number; duration: number },
    ): boolean {
        const cmd: Record<string, unknown> = { type: 'goto_target' };
        if (head !== undefined) {
            if (!Array.isArray(head) || head.length !== 16
                || !head.every((n) => Number.isFinite(n))) {
                throw new TypeError(
                    'gotoTarget: head must be a 16-element flat row-major 4×4 matrix '
                    + `of finite numbers; got ${Array.isArray(head) ? `Array(${head.length})` : typeof head}`,
                );
            }
            cmd.head = head;
        }
        if (antennas !== undefined) {
            if (!Array.isArray(antennas) || antennas.length !== 2
                || !antennas.every((n) => Number.isFinite(n))) {
                throw new TypeError(
                    'gotoTarget: antennas must be [rightRad, leftRad] (2 finite numbers); '
                    + `got ${Array.isArray(antennas) ? `Array(${antennas.length})` : typeof antennas}`,
                );
            }
            cmd.antennas = antennas;
        }
        if (body_yaw !== undefined) {
            if (!Number.isFinite(body_yaw)) {
                throw new TypeError(
                    `gotoTarget: body_yaw must be a finite number (radians); got ${body_yaw}`,
                );
            }
            cmd.body_yaw = body_yaw;
        }
        if (!Number.isFinite(duration) || duration <= 0) {
            throw new TypeError(
                `gotoTarget: duration must be a positive finite number (seconds); got ${duration}`,
            );
        }
        cmd.duration = duration;
        return this._sendCommand(cmd);
    }

    setHeadRpyDeg(rollDeg: number, pitchDeg: number, yawDeg: number): boolean {
        return this.setTarget({ head: rpyToMatrix(rollDeg, pitchDeg, yawDeg).flat() });
    }

    setAntennasDeg(rightDeg: number, leftDeg: number): boolean {
        return this.setTarget({ antennas: [degToRad(rightDeg), degToRad(leftDeg)] });
    }

    setBodyYawDeg(yawDeg: number): boolean {
        return this.setTarget({ body_yaw: degToRad(yawDeg) });
    }

    playSound(file: string): boolean {
        return this._sendCommand({ type: 'play_sound', file });
    }

    clearIncomingAudio(): boolean {
        return this._sendCommand({ type: 'clear_incoming_audio' });
    }

    startHeadTracking(weight = 1.0): boolean {
        if (!Number.isFinite(weight)) {
            throw new TypeError(`startHeadTracking: weight must be a finite number; got ${weight}`);
        }
        const clampedWeight = Math.min(Math.max(weight, 0), 1);
        return this._sendCommand({
            type: 'set_head_tracking',
            enabled: true,
            weight: clampedWeight,
        });
    }

    stopHeadTracking(): boolean {
        return this._sendCommand({ type: 'set_head_tracking', enabled: false });
    }

    getTrackedFace(): Promise<FaceTarget | null> {
        return this._slotRoundtrip(
            () => this._trackedFaceResolve,
            (next) => { this._trackedFaceResolve = next; },
            { type: 'get_tracked_face' },
        );
    }

    /**
     * Trigger a PyPI update of the daemon over the data channel. Remote
     * counterpart of `POST /update/start`. The daemon acks then restarts
     * itself once the install finishes, which tears this session down -
     * the caller is expected to reconnect afterwards.
     *
     * Pass `onProgress` to receive `update_progress` events (one per log
     * line of the update job). A *successful* update restarts the daemon
     * before a `done` event can arrive, so treat the session teardown +
     * a successful reconnect as the success signal; `onProgress` will fire
     * with `status: 'failed'` if the install errors before the restart.
     *
     * Returns `false` if the data channel isn't open.
     */
    startDaemonUpdate(
        { preRelease = false, onProgress }: { preRelease?: boolean; onProgress?: UpdateProgressCallback } = {},
    ): boolean {
        if (onProgress) this._updateProgressSubscribers.add(onProgress);
        return this._sendCommand({ type: 'start_update', pre_release: preRelease });
    }

    setMotorMode(mode: 'enabled' | 'disabled' | 'gravity_compensation'): boolean {
        return this._sendCommand({ type: 'set_motor_mode', mode });
    }

    setMotorTorque(on: boolean, ids: string[] | null = null): boolean {
        return this._sendCommand({ type: 'set_torque', on, ids });
    }

    wakeUp({ timeoutMs = 8000 }: MotionAwaitOptions = {}): Promise<void> {
        this._sendCommand({ type: 'set_motor_mode', mode: 'enabled' });
        return this._sendCommandAwaitCompletion('wake_up', timeoutMs);
    }

    gotoSleep({ timeoutMs = 8000 }: MotionAwaitOptions = {}): Promise<void> {
        return this._sendCommandAwaitCompletion('goto_sleep', timeoutMs);
    }

    private _sendCommandAwaitCompletion(
        command: 'wake_up' | 'goto_sleep',
        timeoutMs: number,
    ): Promise<void> {
        if (!this._sendCommand({ type: command })) {
            return Promise.reject(new Error(`${command}: data channel not open`));
        }
        return new Promise<void>((resolve, reject) => {
            const entry: PendingMotion = {
                resolve,
                reject,
                timer: setTimeout(() => {
                    const queue = this._pendingMotionCompletions[command];
                    const idx = queue.indexOf(entry);
                    if (idx !== -1) queue.splice(idx, 1);
                    reject(new Error(`${command} timed out after ${timeoutMs}ms`));
                }, timeoutMs),
            };
            this._pendingMotionCompletions[command].push(entry);
        });
    }

    private _rejectPendingMotionCompletions(error: Error): void {
        for (const command of Object.keys(this._pendingMotionCompletions) as Array<'wake_up' | 'goto_sleep'>) {
            const queue = this._pendingMotionCompletions[command];
            while (queue.length) {
                const entry = queue.shift()!;
                clearTimeout(entry.timer);
                entry.reject(error);
            }
        }
    }

    isAwake(): boolean {
        const mode = this._robotState?.motor_mode;
        return mode === 'enabled' || mode === 'gravity_compensation';
    }

    async ensureAwake(timeoutMs = 1000): Promise<boolean> {
        if (this._robotState?.motor_mode === undefined) {
            await new Promise<void>((resolve) => {
                const done = (): void => {
                    this.removeEventListener('state', done);
                    clearTimeout(timer);
                    resolve();
                };
                const timer = setTimeout(done, timeoutMs);
                this.addEventListener('state', done);
                this.requestState();
            });
        }
        if (this.isAwake()) return true;
        this.wakeUp().catch(() => { /* swallow: caller may have torn down */ });
        return true;
    }

    getVersion(): Promise<string | null> {
        return new Promise<string | null>((resolve, reject) => {
            if (!this._dc || this._dc.readyState !== 'open') {
                reject(new Error('Data channel not open'));
                return;
            }
            if (this._versionResolve) {
                this._versionResolve(null);
            }
            this._versionResolve = resolve;
            this._sendCommand({ type: 'get_version' });
        });
    }

    getHardwareId(): Promise<string | null> {
        return new Promise<string | null>((resolve, reject) => {
            if (!this._dc || this._dc.readyState !== 'open') {
                reject(new Error('Data channel not open'));
                return;
            }
            if (this._hardwareIdResolve) {
                this._hardwareIdResolve(null);
            }
            this._hardwareIdResolve = resolve;
            this._sendCommand({ type: 'get_hardware_id' });
        });
    }

    getVolume(): Promise<number | null> {
        return this._slotRoundtrip(
            () => this._volumeResolve,
            (next) => { this._volumeResolve = next; },
            { type: 'get_volume' },
        );
    }

    setVolume(volume: number): Promise<number | null> {
        return this._slotRoundtrip(
            () => this._volumeResolve,
            (next) => { this._volumeResolve = next; },
            { type: 'set_volume', volume: clampVolume(volume) },
        );
    }

    getMicrophoneVolume(): Promise<number | null> {
        return this._slotRoundtrip(
            () => this._micVolumeResolve,
            (next) => { this._micVolumeResolve = next; },
            { type: 'get_microphone_volume' },
        );
    }

    setMicrophoneVolume(volume: number): Promise<number | null> {
        return this._slotRoundtrip(
            () => this._micVolumeResolve,
            (next) => { this._micVolumeResolve = next; },
            { type: 'set_microphone_volume', volume: clampVolume(volume) },
        );
    }

    applyAudioConfig(
        config: AudioConfigEntry[],
        { verify = true }: ApplyAudioConfigOptions = {},
    ): Promise<boolean> {
        return this._slotRoundtrip(
            () => this._applyAudioConfigResolve,
            (next) => { this._applyAudioConfigResolve = next; },
            { type: 'apply_audio_config', config, verify },
        ).then((v) => v === true);
    }

    readAudioParameter(name: string): Promise<number[] | null> {
        return this._slotRoundtrip(
            () => this._readAudioParameterResolve,
            (next) => { this._readAudioParameterResolve = next; },
            { type: 'read_audio_parameter', name },
        );
    }

    /**
     * Internal: send a command and await the matching daemon response in a
     * named single-resolver slot. Used by the volume helpers and the
     * XVF3800 audio-config helpers — every one of them has a strict
     * request/response shape where a single in-flight call per slot is
     * sufficient. If a previous request on the same slot is still
     * pending when a new one comes in, the older promise is resolved to
     * `null` so its caller doesn't hang forever.
     *
     * Slot access is passed in as getter/setter closures rather than a
     * key into `this`: that keeps the helper fully generic-checked (T is
     * inferred from the slot's resolver type at each call site) with no
     * indexed-property casts.
     */
    private _slotRoundtrip<T>(
        getSlot: () => ((v: T | null) => void) | null,
        setSlot: (next: ((v: T | null) => void) | null) => void,
        command: Record<string, unknown>,
    ): Promise<T | null> {
        return new Promise<T | null>((resolve, reject) => {
            if (!this._dc || this._dc.readyState !== 'open') {
                reject(new Error('Data channel not open'));
                return;
            }
            const prev = getSlot();
            if (prev) prev(null);
            setSlot(resolve);
            this._sendCommand(command);
        });
    }

    sendRaw(data: unknown): boolean {
        return this._sendCommand(data);
    }

    subscribeLogs({ onLine, onError }: SubscribeLogsOptions): () => void {
        if (typeof onLine !== 'function') {
            throw new TypeError('subscribeLogs: onLine callback is required');
        }
        const sub: LogSubscriber = { onLine, onError };
        const wasEmpty = this._logSubscribers.size === 0;
        this._logSubscribers.add(sub);
        if (wasEmpty) this._sendCommand({ type: 'subscribe_logs' });

        let detached = false;
        return () => {
            if (detached) return;
            detached = true;
            this._logSubscribers.delete(sub);
            if (this._logSubscribers.size === 0) {
                this._sendCommand({ type: 'unsubscribe_logs' });
            }
        };
    }

    requestState(): boolean {
        return this._sendCommand({ type: 'get_state' });
    }

    // ─── Audio ───────────────────────────────────────────────────────────

    setAudioMuted(muted: boolean): void {
        this._audioMuted = muted;
        if (this._videoElement) this._videoElement.muted = muted;
    }

    setMicMuted(muted: boolean): void {
        this._micMuted = muted;
        if (this._micStream) {
            this._micStream.getAudioTracks().forEach((t) => { t.enabled = !muted; });
        }
    }

    // ─── Video helper ────────────────────────────────────────────────────

    attachVideo(videoElement: HTMLVideoElement): () => void {
        this._videoElement = videoElement;
        videoElement.muted = this._audioMuted;

        const onVideoTrack = (e: Event): void => {
            const ev = e as CustomEvent<{ track: MediaStreamTrack; stream: MediaStream }>;
            videoElement.srcObject = ev.detail.stream;
            videoElement.playsInline = true;
            if ('requestVideoFrameCallback' in videoElement) {
                this._startLatencyMonitor(videoElement);
            }
        };

        const onSessionStopped = (): void => { videoElement.srcObject = null; };

        this.addEventListener('videoTrack', onVideoTrack);
        this.addEventListener('sessionStopped', onSessionStopped);

        return () => {
            this.removeEventListener('videoTrack', onVideoTrack);
            this.removeEventListener('sessionStopped', onSessionStopped);
            if (this._latencyMonitorId) { clearInterval(this._latencyMonitorId); this._latencyMonitorId = null; }
            videoElement.srcObject = null;
            this._videoElement = null;
        };
    }

    // ─── Daemon-side recorded-move playback ──────────────────────────────

    async playMove(
        motion: MoveData,
        {
            audioBlob = null,
            audioLeadMs = -100,
            description = 'move',
            encoding = 'gzip+base64',
            playFrequency = 100,
            initialGotoDuration = 0,
            startTimeoutMs = 8000,
            onProgress = () => { /* no-op */ },
            onStarted = () => { /* no-op */ },
        }: PlayMoveOptions = {},
    ): Promise<PlayMoveResult> {
        if (!this._dc || this._dc.readyState !== 'open') {
            throw new Error('data channel not open');
        }
        if (!motion?.time?.length || !motion?.set_target_data?.length) {
            throw new Error('playMove: motion must have time + set_target_data');
        }
        const uploadId = makeUploadId();
        this._activeMoveUploadId = uploadId;

        const moveDict = {
            description,
            time: motion.time,
            set_target_data: motion.set_target_data,
        };
        const jsonStr = JSON.stringify(moveDict);
        let payload: string;
        let effectiveEncoding: 'gzip+base64' | 'json';
        if (encoding === 'gzip+base64' && hasCompressionStream()) {
            payload = await gzipBase64(jsonStr);
            effectiveEncoding = 'gzip+base64';
        } else {
            payload = jsonStr;
            effectiveEncoding = 'json';
        }
        const totalChunks = Math.ceil(payload.length / UPLOAD_CHUNK_SIZE) || 1;

        onProgress({
            phase: 'starting',
            sent: 0,
            total: totalChunks,
            bytes: payload.length,
            encoding: effectiveEncoding,
        });

        this._sendCommand({
            type: 'upload_move_start',
            upload_id: uploadId,
            total_chunks: totalChunks,
            description,
            encoding: effectiveEncoding,
        });
        for (let i = 0; i < totalChunks; i++) {
            if (this._dc.bufferedAmount > UPLOAD_BUFFERED_HIGH_WATER) {
                await this._awaitDataChannelDrain();
            }
            const start = i * UPLOAD_CHUNK_SIZE;
            this._sendCommand({
                type: 'upload_move_chunk',
                upload_id: uploadId,
                chunk_index: i,
                chunk: payload.slice(start, start + UPLOAD_CHUNK_SIZE),
            });
            onProgress({ phase: 'upload', sent: i + 1, total: totalChunks });
        }
        this._sendCommand({ type: 'upload_move_finish', upload_id: uploadId });
        onProgress({ phase: 'uploaded', sent: totalChunks, total: totalChunks });

        if (audioBlob) {
            const rawBytes = new Uint8Array(await audioBlob.arrayBuffer());
            const audioB64 = bytesToBase64(rawBytes);
            const audioTotal = Math.ceil(audioB64.length / UPLOAD_CHUNK_SIZE) || 1;
            onProgress({
                phase: 'audio-starting',
                sent: 0,
                total: audioTotal,
                bytes: audioB64.length,
            });
            this._sendCommand({
                type: 'upload_audio_start',
                upload_id: uploadId,
                total_chunks: audioTotal,
                encoding: audioUploadEncoding(audioBlob),
                description,
            });
            for (let i = 0; i < audioTotal; i++) {
                if (this._dc.bufferedAmount > UPLOAD_BUFFERED_HIGH_WATER) {
                    await this._awaitDataChannelDrain();
                }
                const start = i * UPLOAD_CHUNK_SIZE;
                this._sendCommand({
                    type: 'upload_audio_chunk',
                    upload_id: uploadId,
                    chunk_index: i,
                    chunk: audioB64.slice(start, start + UPLOAD_CHUNK_SIZE),
                });
                onProgress({ phase: 'audio-upload', sent: i + 1, total: audioTotal });
            }
            this._sendCommand({ type: 'upload_audio_finish', upload_id: uploadId });
            onProgress({ phase: 'audio-uploaded', sent: audioTotal, total: audioTotal });
        }

        this._sendCommand({
            type: 'play_uploaded_move',
            upload_id: uploadId,
            play_frequency: playFrequency,
            initial_goto_duration: initialGotoDuration,
            audio_lead_ms: audioLeadMs,
        });
        let startedAck: Record<string, unknown>;
        try {
            startedAck = await this._waitForBroadcast(
                (m) =>
                    m?.type === 'play_uploaded_move'
                    && m?.upload_id === uploadId
                    && (m.started === true || typeof m.error === 'string'),
                { timeoutMs: startTimeoutMs, debugLabel: 'play_uploaded_move started' },
            );
        } catch (e) {
            throw new Error(
                'Daemon did not respond to play_uploaded_move '
                + '(requires the reachy_mini daemon with feature/daemon-side-move-upload). '
                + `Underlying: ${(e as Error).message}`,
            );
        }
        if (typeof startedAck.error === 'string') {
            throw new Error(`play_uploaded_move: ${startedAck.error}`);
        }
        try {
            onStarted({
                duration_s: startedAck.duration_s as number,
                has_audio: startedAck.has_audio === true,
            });
        } catch (e) {
            console.warn('playMove.onStarted threw:', e);
        }
        onProgress({ phase: 'playing', duration_s: startedAck.duration_s as number });

        const final = await this._waitForBroadcast(
            (m) =>
                m?.type === 'play_uploaded_move'
                && m?.upload_id === uploadId
                && (m.finished === true
                    || m.cancelled === true
                    || typeof m.error === 'string'),
            {
                timeoutMs: ((startedAck.duration_s as number) + 30) * 1000,
                debugLabel: 'play_uploaded_move final',
            },
        );
        if (this._activeMoveUploadId === uploadId) {
            this._activeMoveUploadId = null;
        }
        return final as PlayMoveResult;
    }

    cancelMove(uploadId: string | null = null): boolean {
        const id = uploadId ?? this._activeMoveUploadId;
        if (!id) return false;
        return this._sendCommand({ type: 'cancel_move', upload_id: id });
    }

    async uploadAudio(
        audioBlob: Blob,
        { description = 'audio', onProgress = () => { /* no-op */ } }: UploadAudioOptions = {},
    ): Promise<string> {
        if (!this._dc || this._dc.readyState !== 'open') {
            throw new Error('data channel not open');
        }
        if (!(audioBlob instanceof Blob)) {
            throw new TypeError('uploadAudio: expected a Blob');
        }
        const uploadId = makeUploadId();
        const rawBytes = new Uint8Array(await audioBlob.arrayBuffer());
        const audioB64 = bytesToBase64(rawBytes);
        const total = Math.ceil(audioB64.length / UPLOAD_CHUNK_SIZE) || 1;
        onProgress({ phase: 'audio-starting', sent: 0, total, bytes: audioB64.length });
        this._sendCommand({
            type: 'upload_audio_start',
            upload_id: uploadId,
            total_chunks: total,
            encoding: audioUploadEncoding(audioBlob),
            description,
        });
        for (let i = 0; i < total; i++) {
            if (this._dc.bufferedAmount > UPLOAD_BUFFERED_HIGH_WATER) {
                await this._awaitDataChannelDrain();
            }
            const start = i * UPLOAD_CHUNK_SIZE;
            this._sendCommand({
                type: 'upload_audio_chunk',
                upload_id: uploadId,
                chunk_index: i,
                chunk: audioB64.slice(start, start + UPLOAD_CHUNK_SIZE),
            });
            onProgress({ phase: 'audio-upload', sent: i + 1, total });
        }
        this._sendCommand({ type: 'upload_audio_finish', upload_id: uploadId });
        onProgress({ phase: 'audio-uploaded', sent: total, total });
        return uploadId;
    }

    async playUploadedAudio(
        uploadId: string,
        { timeoutMs = 8000 }: PlayUploadedAudioOptions = {},
    ): Promise<{ started: true }> {
        if (!this._dc || this._dc.readyState !== 'open') {
            throw new Error('data channel not open');
        }
        const waiter = this._waitForBroadcast(
            (m) =>
                m?.type === 'play_uploaded_audio'
                && m?.upload_id === uploadId
                && (m.started === true || typeof m.error === 'string'),
            { timeoutMs, debugLabel: 'play_uploaded_audio started' },
        );
        this._sendCommand({ type: 'play_uploaded_audio', upload_id: uploadId });
        const ack = await waiter;
        if (typeof ack.error === 'string') throw new Error(ack.error);
        this._activeAudioUploadId = uploadId;
        return ack as { started: true };
    }

    cancelAudio(uploadId: string | null = null): boolean {
        const id = uploadId ?? this._activeAudioUploadId;
        if (!id) return false;
        if (this._activeAudioUploadId === id) {
            this._activeAudioUploadId = null;
        }
        return this._sendCommand({ type: 'cancel_audio', upload_id: id });
    }

    // ─── Private ─────────────────────────────────────────────────────────

    private _emit<K extends keyof ReachyMiniEventMap>(
        name: K,
        detail: ReachyMiniEventMap[K] extends CustomEvent<infer D> ? D : never,
    ): void {
        this.dispatchEvent(new CustomEvent(name, { detail }));
    }

    private _waitForBroadcast(
        predicate: (m: Record<string, unknown>) => boolean,
        { timeoutMs = 5000, debugLabel = '' }: { timeoutMs?: number; debugLabel?: string } = {},
    ): Promise<Record<string, unknown>> {
        return new Promise<Record<string, unknown>>((resolve, reject) => {
            const slot: BroadcastWaiter = {
                predicate,
                resolve,
                timer: setTimeout(() => {
                    const i = this._broadcastWaiters.indexOf(slot);
                    if (i !== -1) this._broadcastWaiters.splice(i, 1);
                    reject(new Error(`broadcast timeout (${timeoutMs} ms): ${debugLabel}`));
                }, timeoutMs),
            };
            this._broadcastWaiters.push(slot);
        });
    }

    private async _awaitDataChannelDrain(): Promise<void> {
        while (this._dc && this._dc.bufferedAmount > UPLOAD_BUFFERED_LOW_WATER) {
            await new Promise<void>((r) => setTimeout(r, 30));
            if (!this._dc || this._dc.readyState !== 'open') {
                throw new Error('data channel closed mid-upload');
            }
        }
    }

    private async _sendToServer(
        message: Record<string, unknown>,
    ): Promise<SignalingMessage | null> {
        if (!this._token) throw new Error('No token — authenticate() first');
        try {
            const res = await fetch(`${this._signalingUrl}/send`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this._token}`,
                },
                body: JSON.stringify(message),
            });
            if (!res.ok) {
                let body = '';
                try { body = await res.text(); } catch { /* ignore */ }
                console.warn(
                    `[reachy-mini] /send rejected (${res.status}) for type=${(message as { type?: string })?.type}; body=${body || '<empty>'}`,
                );
                return null;
            }
            return await res.json() as SignalingMessage;
        } catch (e) {
            console.error('Send error:', e);
            return null;
        }
    }

    private _sendCommand(cmd: unknown): boolean {
        if (!this._dc || this._dc.readyState !== 'open') return false;
        this._dc.send(JSON.stringify(cmd));
        return true;
    }

    /**
     * Call a JSON-RPC method on the robot/app over the DataChannel and await
     * its result. This is the one way to drive an on-robot app (start/stop it
     * via `apps.*`, or drive a running app via its own namespace, e.g.
     * `conversation.say`). Rejects on the JSON-RPC error, a closed channel, or
     * timeout.
     */
    rpcCall<T = unknown>(
        method: string,
        params: Record<string, unknown> = {},
        opts: { timeoutMs?: number } = {},
    ): Promise<T> {
        const timeoutMs = opts.timeoutMs ?? 20000;
        const id = `rpc-${++this._rpcCounter}`;
        if (!this._sendCommand({ jsonrpc: '2.0', id, method, params })) {
            return Promise.reject(new Error(`rpcCall(${method}): data channel not open`));
        }
        return new Promise<T>((resolve, reject) => {
            const timer = setTimeout(() => {
                this._pendingRpc.delete(id);
                reject(new Error(`rpcCall(${method}) timed out after ${timeoutMs}ms`));
            }, timeoutMs);
            this._pendingRpc.set(id, {
                resolve: (v) => resolve(v as T),
                reject,
                timer,
            });
        });
    }

    /**
     * Subscribe to a JSON-RPC notification (one-way event) pushed by the
     * robot/app, e.g. `conversation.turn`. Returns an unsubscribe function.
     */
    onNotification(
        method: string,
        cb: (params: Record<string, unknown>) => void,
    ): () => void {
        let set = this._rpcListeners.get(method);
        if (!set) {
            set = new Set();
            this._rpcListeners.set(method, set);
        }
        set.add(cb);
        return () => {
            this._rpcListeners.get(method)?.delete(cb);
        };
    }

    private _rejectPendingRpc(err: Error): void {
        for (const pending of this._pendingRpc.values()) {
            clearTimeout(pending.timer);
            pending.reject(err);
        }
        this._pendingRpc.clear();
    }

    private _handleRpcMessage(data: Record<string, unknown>): void {
        // Response to an rpcCall (correlated by id)...
        if ('id' in data && data.id != null && ('result' in data || 'error' in data)) {
            const pending = this._pendingRpc.get(data.id as string);
            if (!pending) return;
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
            return;
        }
        // ...or a one-way notification (event): dispatch to listeners.
        if (typeof data.method === 'string') {
            const params = (data.params as Record<string, unknown> | undefined) ?? {};
            for (const cb of this._rpcListeners.get(data.method) ?? []) {
                try {
                    cb(params);
                } catch (e) {
                    console.error(`onNotification(${data.method}) threw:`, e);
                }
            }
        }
    }

    private _checkSessionReady(): void {
        if (this._iceConnected && this._dcOpen && this._sessionResolve) {
            this._state = 'streaming';
            this.requestState();
            this._stateRefreshInterval = setInterval(() => this.requestState(), 500);
            this._emit('streaming', { sessionId: this._sessionId!, robotId: this._selectedRobotId! });
            this._sessionResolve();
            this._sessionResolve = null;
            this._sessionReject = null;
        }
    }

    private async _handleSignalingMessage(msg: SignalingMessage): Promise<void> {
        switch (msg.type) {
            case 'welcome':
                break;
            case 'list':
                this._robots = msg.producers || [];
                this._emit('robotsChanged', { robots: this._robots });
                this._maybeAutoStart();
                break;
            case 'peerStatusChanged': {
                const list = await this._sendToServer({ type: 'list' });
                if (list?.producers) {
                    this._robots = list.producers;
                    this._emit('robotsChanged', { robots: this._robots });
                    this._maybeAutoStart();
                }
                break;
            }
            case 'sessionStarted':
                this._sessionId = msg.sessionId ?? null;
                break;
            case 'sessionRejected':
                this._failSessionRejected(msg);
                break;
            case 'endSession':
                this._handleEndSession(msg);
                break;
            case 'peer':
                this._handlePeerMessage(msg);
                break;
        }
    }

    private _handleEndSession(msg: SignalingMessage): void {
        const reason = msg.reason;
        const friendly = reason === 'robot_busy_local_app'
            ? 'Robot is busy: a local Python app is running'
            : reason === 'local_app_started'
                ? 'Disconnected: a local Python app started on the robot'
                : reason === 'robot_busy_local'
                    ? 'Robot is busy: another session is already active'
                    : null;

        if (this._sessionReject) {
            const err = new Error(
                friendly || `Session ended before it could start: ${reason || 'unknown reason'}`,
            ) as SessionRejectError;
            err.reason = reason ?? null;
            this._emit('sessionRejected', { reason, activeApp: null });
            // Resilience teardown alongside the PC close path.
            this._clearIceGrace();
            this._uninstallNetworkListeners();
            if (this._pc) { this._pc.close(); this._pc = null; }
            if (this._micStream) { this._micStream.getTracks().forEach((t) => t.stop()); this._micStream = null; }
            this._iceConnected = false;
            this._dcOpen = false;
            this._micMuted = true;
            this._micSupported = false;
            const reject = this._sessionReject;
            this._sessionResolve = null;
            this._sessionReject = null;
            reject(err);
            return;
        }

        if (this._state === 'streaming') {
            this._emit('sessionStopped', {
                reason: reason || 'remote_end',
                message: friendly,
            });
            this.stopSession().catch(() => { /* swallow */ });
        }
    }

    private async _handlePeerMessage(msg: SignalingMessage): Promise<void> {
        if (!this._pc) return;
        try {
            if (msg.sdp) {
                const sdp = msg.sdp;
                if (sdp.type === 'offer') {
                    const supportsMic = sdpHasAudioSendRecv(sdp.sdp);
                    this._micSupported = supportsMic;
                    this._emit('micSupported', { supported: supportsMic });

                    if (supportsMic && this._micStream) {
                        for (const track of this._micStream.getAudioTracks()) {
                            this._pc.addTrack(track, this._micStream);
                        }
                    }

                    await this._pc.setRemoteDescription(new RTCSessionDescription(sdp));
                    const answer = await this._pc.createAnswer();
                    await this._pc.setLocalDescription(answer);
                    await this._sendToServer({
                        type: 'peer',
                        sessionId: this._sessionId,
                        sdp: { type: 'answer', sdp: answer.sdp },
                    });
                } else {
                    await this._pc.setRemoteDescription(new RTCSessionDescription(sdp));
                }
                const pending = this._pendingRemoteIce;
                if (pending && pending.length) {
                    this._pendingRemoteIce = [];
                    for (const ice of pending) {
                        try {
                            await this._pc.addIceCandidate(new RTCIceCandidate(ice));
                        } catch (err) {
                            console.warn('[reachy-mini] buffered ICE candidate rejected:', err);
                        }
                    }
                }
            }
            if (msg.ice) {
                // Safari (and the iOS WKWebView Tauri ships on) rejects
                // empty candidate strings with `OperationError: Expect
                // line: candidate:<candidate-str>`. The signaling
                // server uses an empty string as the end-of-candidates
                // marker (legal per the WebRTC spec but optional).
                if (!msg.ice.candidate) return;
                if (this._pc.remoteDescription) {
                    await this._pc.addIceCandidate(new RTCIceCandidate(msg.ice));
                } else {
                    if (!this._pendingRemoteIce) this._pendingRemoteIce = [];
                    this._pendingRemoteIce.push(msg.ice);
                }
            }
        } catch (e) {
            console.error('WebRTC error:', e);
            this._emit('error', { source: 'webrtc', error: e as Error });
        }
    }

    private _handleRobotMessage(data: Record<string, unknown>): void {
        // JSON-RPC frames (app control surface) are handled separately from
        // the legacy {command|type} robot messages that share this channel.
        if (data.jsonrpc === '2.0') {
            this._handleRpcMessage(data);
            return;
        }
        if ('version' in data && this._versionResolve) {
            this._versionResolve(data.version as string | null);
            this._versionResolve = null;
            return;
        }
        if ('hardware_id' in data && this._hardwareIdResolve) {
            this._hardwareIdResolve(data.hardware_id as string | null);
            this._hardwareIdResolve = null;
            return;
        }
        if (data.command === 'get_volume' || data.command === 'set_volume') {
            if (this._volumeResolve) {
                this._volumeResolve(data.status === 'error' ? null : (data.volume as number));
                this._volumeResolve = null;
            }
            return;
        }
        if (data.command === 'get_microphone_volume' || data.command === 'set_microphone_volume') {
            if (this._micVolumeResolve) {
                this._micVolumeResolve(data.status === 'error' ? null : (data.volume as number));
                this._micVolumeResolve = null;
            }
            return;
        }
        if (data.command === 'apply_audio_config') {
            if (this._applyAudioConfigResolve) {
                this._applyAudioConfigResolve(data.error ? false : !!data.applied);
                this._applyAudioConfigResolve = null;
            }
            return;
        }
        if (data.command === 'read_audio_parameter') {
            if (this._readAudioParameterResolve) {
                this._readAudioParameterResolve(
                    data.error ? null : ((data.values as number[] | undefined) ?? null),
                );
                this._readAudioParameterResolve = null;
            }
            return;
        }
        if (data.command === 'get_tracked_face') {
            if (this._trackedFaceResolve) {
                this._trackedFaceResolve(
                    (data.face_target as FaceTarget | undefined) ?? null,
                );
                this._trackedFaceResolve = null;
            }
            return;
        }
        if (
            (data.command === 'wake_up' || data.command === 'goto_sleep')
            && this._pendingMotionCompletions
            && this._pendingMotionCompletions[data.command as 'wake_up' | 'goto_sleep']
        ) {
            const queue = this._pendingMotionCompletions[data.command as 'wake_up' | 'goto_sleep'];
            if (data.completed === true && queue.length > 0) {
                const entry = queue.shift()!;
                clearTimeout(entry.timer);
                entry.resolve();
                return;
            }
            if (data.error && queue.length > 0) {
                const entry = queue.shift()!;
                clearTimeout(entry.timer);
                entry.reject(new Error(`${data.command}: ${data.error}`));
                return;
            }
        }
        if (data.type === 'log_line') {
            for (const sub of this._logSubscribers) {
                try {
                    sub.onLine({ timestamp: data.timestamp as string, line: data.line as string });
                } catch (e) {
                    console.error('subscribeLogs onLine threw:', e);
                }
            }
            return;
        }
        if (data.type === 'log_stream_error') {
            for (const sub of this._logSubscribers) {
                if (typeof sub.onError === 'function') {
                    try { sub.onError(data.error as string); }
                    catch (e) { console.error('subscribeLogs onError threw:', e); }
                }
            }
            return;
        }
        if (data.command === 'start_update') {
            // Refusal ack (non-wireless robot, no update available, or one
            // already running): the daemon never spawned the job, so surface
            // it to `onProgress` as a terminal `failed` event - there will be
            // no transport teardown to infer success from.
            if (typeof data.error === 'string') {
                const event: UpdateProgressEvent = { status: 'failed', error: data.error };
                for (const cb of this._updateProgressSubscribers) {
                    try { cb(event); }
                    catch (e) { console.error('startDaemonUpdate onProgress threw:', e); }
                }
            }
            return;
        }
        if (data.type === 'update_progress') {
            const event: UpdateProgressEvent = {
                status: data.status as UpdateProgressEvent['status'],
                line: typeof data.line === 'string' ? data.line : undefined,
                error: typeof data.error === 'string' ? data.error : undefined,
            };
            for (const cb of this._updateProgressSubscribers) {
                try { cb(event); }
                catch (e) { console.error('startDaemonUpdate onProgress threw:', e); }
            }
            return;
        }
        if (data.state) {
            const s = data.state as {
                head_pose?: number[][];
                antennas?: [number, number];
                body_yaw?: number;
                motor_mode?: 'enabled' | 'disabled' | 'gravity_compensation';
                is_move_running?: boolean;
                face_target?: FaceTarget;
            };
            if (s.head_pose) this._robotState.head = s.head_pose.flat();
            if (s.antennas) this._robotState.antennas = [s.antennas[0], s.antennas[1]];
            if (typeof s.body_yaw === 'number') this._robotState.body_yaw = s.body_yaw;
            if (s.motor_mode) this._robotState.motor_mode = s.motor_mode;
            if (typeof s.is_move_running === 'boolean') this._robotState.is_move_running = s.is_move_running;
            if (s.face_target) this._robotState.face_target = s.face_target;
            this._emit('state', { ...this._robotState });
        }
        if (data.error) {
            this._emit('error', { source: 'robot', error: data.error as string });
        }
        if (this._broadcastWaiters.length > 0) {
            for (let i = this._broadcastWaiters.length - 1; i >= 0; i--) {
                const slot = this._broadcastWaiters[i]!;
                if (slot.predicate(data)) {
                    this._broadcastWaiters.splice(i, 1);
                    clearTimeout(slot.timer);
                    slot.resolve(data);
                    return;
                }
            }
        }
    }

    /** Snap video playback to live edge if buffered lag exceeds 0.5 s. */
    private _startLatencyMonitor(video: HTMLVideoElement): void {
        if (this._latencyMonitorId) clearInterval(this._latencyMonitorId);
        this._latencyMonitorId = setInterval(() => {
            if (!video.srcObject || video.paused) return;
            const buf = video.buffered;
            if (buf.length > 0) {
                const end = buf.end(buf.length - 1);
                const lag = end - video.currentTime;
                if (lag > 0.5) {
                    console.log(`Latency correction: was ${lag.toFixed(2)}s behind`);
                    video.currentTime = end - 0.1;
                }
            }
        }, 2000);
    }
}
