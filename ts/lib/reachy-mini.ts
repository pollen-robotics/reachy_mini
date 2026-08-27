/**
 * ReachyMini — Browser SDK for controlling a Reachy Mini robot over WebRTC.
 * See `../reachy-mini-sdk.ts` for the package's barrel and the README for
 * a quick-start guide.
 */

import {
    oauthHandleRedirectIfPresent,
    oauthLoginUrl,
} from '@huggingface/hub';

import { createLogger } from './logger.js';
import { degToRad, rpyToMatrix } from './math.js';
import { BroadcastTimeoutError, PendingReplies, SLOT_ROUNDTRIP_TIMEOUT_MS } from './pending-replies.js';
import type { MotionCommand, ReplySlotKey, ReplySlotValues } from './pending-replies.js';
import { SessionSupervisor } from './session-supervisor.js';
import { SDK_VERSION } from './version.js';
import {
    consumeFragmentCredentials,
    readPreselectedRobotIdFromUrl,
    sdpHasAudioSendRecv,
} from './url-helpers.js';
import {
    clearStoredToken,
    consumeOAuthErrorParams,
    readUsableToken,
    writeStoredToken,
} from './token-store.js';
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
    ImuData,
    LoginOptions,
    MotionAwaitOptions,
    MoveData,
    PlayMoveOptions,
    PlayMoveResult,
    PlayUploadedAudioOptions,
    ReachyMiniEventMap,
    ReachyMiniInstance,
    ReachyMiniOptions,
    RequestOptions,
    RobotInfo,
    RobotState,
    SessionRejectError,
    StartDaemonUpdateOptions,
    SubscribeLogsOptions,
    UpdateProgressEvent,
    UploadAudioOptions,
} from './types.js';

const log = createLogger('sdk');

interface LogSubscriber {
    onLine: (entry: { timestamp: string; line: string }) => void;
    onError?: (error: string) => void;
}

export type { UpdateProgressEvent };

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
// Resilience tunables (ICE grace windows, re-dial backoff, DC-silence
// watchdog thresholds) live in `session-supervisor.ts`.

/**
 * How long a pushed pose frame keeps the periodic `get_state` poll on hold -
 * both on the way out (no request is sent) and on the way back (a reply that
 * arrives anyway doesn't touch the state mirror).
 *
 * Poll replies carry no `seq`, so they slip past the stale-frame guard: a
 * reply that crosses a fresher pushed frame would rewind the very mirror the
 * stream exists to smooth. While frames flow at ~30 Hz there is nothing left
 * for the poll to add, so it stands down. Keying that on frame arrival rather
 * than on `_poseSubRefs` keeps the poll running against a daemon that doesn't
 * know `subscribe_pose`, and brings it back on its own if the stream stalls.
 * A little over one 500 ms poll period, so a couple of dropped frames don't
 * flip it back and forth.
 */
const POSE_STREAM_FRESH_MS = 750;

/**
 * Upper bound on how long `ensureAwake()` waits for the wake trajectory to
 * complete. The emote itself takes ~2-3 s on hardware; the extra headroom
 * covers a cold trajectory player. Deliberately shorter than `wakeUp()`'s
 * own 8 s default: `ensureAwake()` gates app boot, and a daemon that hasn't
 * confirmed within 5 s isn't going to - better to let the app in degraded
 * than trap it on the splash.
 */
const WAKE_TRAJECTORY_BUDGET_MS = 5000;

/**
 * Wire payload for `play_recorded_move`, shared by the fire-and-forget and
 * the awaited variant so the two can't drift apart.
 */
function recordedMoveCmd(
    moveName: string,
    { dataset, initialGotoDuration }: { dataset?: string; initialGotoDuration?: number },
): { type: string } & Record<string, unknown> {
    return {
        type: 'play_recorded_move',
        move_name: moveName,
        ...(dataset ? { dataset_name: dataset } : {}),
        ...(initialGotoDuration && initialGotoDuration > 0
            ? { initial_goto_duration: initialGotoDuration }
            : {}),
    };
}

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
    // Highest `seq` seen on the unordered `pose` channel. Frames that arrive
    // out of order (older seq) are dropped so a late packet can't rewind the
    // live mirror. `null` until the first pose frame.
    private _lastPoseSeq: number | null = null;
    // When the last pushed pose frame was applied, used to park the periodic
    // `get_state` poll while the stream is live (see POSE_STREAM_FRESH_MS).
    private _lastPoseFrameAt = 0;
    // Local refcount of pose-stream consumers (the 3D mirror, the wizard's
    // move-end watcher, ...). The daemon's subscription is a per-peer boolean
    // (not refcounted), so we only send `unsubscribe_pose` once the LAST local
    // consumer releases - otherwise one consumer's cleanup would kill the
    // stream for the others.
    private _poseSubRefs = 0;
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

    // ─── Pending replies (reply slots / JSON-RPC / motion / broadcast) ───
    // All request/response waiters on the data channel live in one
    // ledger so every teardown path settles them in a single call.
    private readonly _pending = new PendingReplies();

    // ─── Log subscribers ─────────────────────────────────────────────────
    private readonly _logSubscribers: Set<LogSubscriber> = new Set();
    private readonly _updateProgressSubscribers: Set<UpdateProgressCallback> = new Set();

    // ─── JSON-RPC notification listeners (one-way events, no id) ─────────
    // onNotification() subscribes to events the robot/app pushes
    // (conversation.phase/turn/transcript, ...).
    private readonly _rpcListeners = new Map<string, Set<(params: Record<string, unknown>) => void>>();

    // ─── Active upload ids for no-arg cancels ────────────────────────────
    private _activeMoveUploadId: string | null = null;
    private _activeAudioUploadId: string | null = null;

    // ─── Session promise plumbing ────────────────────────────────────────
    private _sessionResolve: (() => void) | null = null;
    private _sessionReject: ((err: Error) => void) | null = null;
    private _iceConnected = false;
    private _dcOpen = false;

    // ─── Resilience ──────────────────────────────────────────────────────
    // ICE-blip debounce, network awareness, automatic session re-dial and
    // the data-channel silence watchdog all live in the supervisor; the
    // class forwards pc/browser events to it and implements its deps as
    // closures over the private state below (see the constructor).
    private readonly _supervisor: SessionSupervisor;

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
        // Arrow closures so every dep call reads the CURRENT class state.
        this._supervisor = new SessionSupervisor({
            iceState: () => this._pc?.iceConnectionState ?? null,
            hasPc: () => !!this._pc,
            isStreaming: () => this._state === 'streaming',
            isMidSetup: () => !!(this._sessionResolve || this._sessionReject),
            isSignalingDown: () => this._state === 'disconnected',
            selectedRobotId: () => this._selectedRobotId,
            rejectPendingSession: (err) => {
                if (!this._sessionReject) return false;
                const reject = this._sessionReject;
                this._sessionResolve = null;
                this._sessionReject = null;
                reject(err);
                return true;
            },
            reconnectSignaling: () => this.connect(),
            // The private dial skips the public startSession()'s
            // cancelRedial, so the loop's own attempts don't cancel it.
            dial: (robotId) => this._startSessionInternal(robotId),
            teardownForRedial: () => this._teardownForRedial(),
            nudgeState: () => { this.requestState(); },
            emit: (name, detail) => this._emit(name, detail),
        }, { autoReconnect: options.autoReconnect !== false });
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
     * Live RTCPeerConnection, or `null` between sessions. Read-only escape
     * hatch for stats sampling (`getStats()`); mutating it is unsupported.
     * Auto-reconnect re-dials REPLACE this object, so re-read it on every
     * use — never capture it across ticks.
     */
    get peerConnection(): RTCPeerConnection | null { return this._pc; }

    /**
     * Build version of this SDK (npm package `version`), injected from
     * package.json at build time. This is the JS SDK's OWN version and is
     * distinct from `getVersion()`, which asks the DAEMON its version over
     * the data channel. `0.0.0-managed-by-ci` means an unreleased/branch
     * build (npm releases carry a real semver).
     */
    get sdkVersion(): string { return SDK_VERSION; }

    /** Same value as the instance `sdkVersion`, reachable without an
     *  instance: `ReachyMini.version`. */
    static get version(): string { return SDK_VERSION; }

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
                log.warn('autoStartFromUrl: startSession rejected:', err);
            });
        }, 0);
    }

    // ─── Auth ────────────────────────────────────────────────────────────

    async authenticate(): Promise<boolean> {
        try {
            consumeFragmentCredentials();

            // A failed silent login (`login({ prompt: 'none' })`) returns as
            // `?error=login_required` / `consent_required` query params.
            // Strip them so they don't linger in the URL; the fall-through
            // to the cached-token check below then reports "not signed in".
            const silentError = consumeOAuthErrorParams();
            if (silentError) {
                log.info('silent sign-in declined:', silentError);
            }

            const result = (await oauthHandleRedirectIfPresent()) as OAuthRedirectResult | false | null;
            if (result) {
                this._username = result.userInfo.preferred_username || result.userInfo.name || null;
                this._token = result.accessToken;
                this._tokenExpires = result.accessTokenExpiresAt;
                writeStoredToken({
                    token: this._token,
                    username: this._username ?? '',
                    expires:
                        typeof this._tokenExpires === 'string'
                            ? this._tokenExpires
                            : this._tokenExpires.toISOString(),
                });
                return true;
            }

            // Cached-token path. `readUsableToken` enforces both the OAuth
            // expiry and a sliding idle window (see token-store.ts).
            const stored = readUsableToken();
            if (stored && stored.username) {
                this._token = stored.token;
                this._username = stored.username;
                this._tokenExpires = stored.expires;
                return true;
            }
            return false;
        } catch (e) {
            log.error('authenticate failed:', e);
            return false;
        }
    }

    async login(options?: LoginOptions): Promise<void> {
        const opts: { clientId?: string } = {};
        if (this._clientId) opts.clientId = this._clientId;
        let url = await oauthLoginUrl(opts);
        // OIDC prompt param. `oauthLoginUrl` doesn't expose it, but the HF
        // authorize endpoint honours it (verified empirically): with
        // `prompt=none` an already-authorized user comes straight back with
        // a code and no screen, anyone else comes back with `?error=...`
        // instead of landing on the HF login page.
        if (options?.prompt) {
            const u = new URL(url);
            u.searchParams.set('prompt', options.prompt);
            url = u.toString();
        }
        window.location.href = url;
    }

    logout(): void {
        clearStoredToken();
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
                catch (e) { log.warn('autoConnect: ensureAwake failed:', e); }
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
            log.warn('/api/robot-status unavailable, using SSE list:', e);
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
        // An explicit dial from the app supersedes any in-flight
        // auto-reconnect (possibly towards a different robot). The
        // redial loop dials through `_startSessionInternal` directly,
        // so its own attempts never trip this.
        this._supervisor.cancelRedial();
        return this._startSessionInternal(robotId);
    }

    private async _startSessionInternal(robotId: string): Promise<void> {
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
            log.warn('audio sender placeholder setup failed:', e);
            this._micStream = null;
        }

        this._pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] satisfies RTCIceServer[],
        });

        // Scope `networkOnline` / `networkOffline` / `networkChange`
        // event forwarding to the lifetime of this session.
        this._supervisor.installNetworkListeners();

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
                    this._supervisor.onIceHealed();
                    this._iceConnected = true;
                    this._checkSessionReady();
                    return;
                }
                if (s === 'disconnected') {
                    this._supervisor.onIceDisconnected();
                    return;
                }
                if (s === 'failed') {
                    this._supervisor.onIceFailed();
                    return;
                }
            };

            this._pc!.ondatachannel = (e) => {
                const ch = e.channel;
                // On `subscribe_pose` the daemon opens a second,
                // unreliable/unordered channel labelled "pose" that *pushes*
                // the robot state at ~30 Hz (see media_server
                // `_setup_pose_channel`), so this can fire mid-session. It
                // carries the same `{state:{...}}` envelope as a get_state
                // reply, so we route it through the same handler - but it
                // must NOT gate session readiness (that's the reliable
                // control channel's job) nor become `_dc` (commands must
                // never ride the lossy channel).
                if (ch.label === 'pose') {
                    // Fresh channel (new session or daemon restart): the
                    // daemon's seq counter may have reset, so forget the old
                    // high-water mark or we'd drop every new frame.
                    this._lastPoseSeq = null;
                    ch.onmessage = (ev) => {
                        const msg = JSON.parse(ev.data);
                        // Drop stale/reordered frames (unordered channel).
                        if (typeof msg.seq === 'number') {
                            if (this._lastPoseSeq !== null && msg.seq <= this._lastPoseSeq) return;
                            this._lastPoseSeq = msg.seq;
                        }
                        this._lastPoseFrameAt = Date.now();
                        this._handleRobotMessage(msg, true);
                    };
                    return;
                }
                this._dc = ch;
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

    /**
     * Common transport teardown shared by every session-ending path
     * (stopSession, disconnect, the auto-redial, a session rejection):
     * settle the pending-reply ledger and the startSession() resolvers
     * with `settleErr`, stand the resilience plumbing down, stop the
     * session-scoped timers, release the mic placeholder and close the
     * pc/dc. Returns the session id (already cleared on the instance)
     * so each caller decides whether and how to send `endSession`.
     * What else differs — events, subscriber wipes, where `_state`
     * lands — stays with the callers.
     */
    private _closeTransport(settleErr: Error): string | null {
        this._pending.settleAll(settleErr);
        if (this._sessionReject) {
            const reject = this._sessionReject;
            this._sessionResolve = null;
            this._sessionReject = null;
            reject(settleErr);
        }
        this._sessionResolve = null;
        // Resilience teardown BEFORE closing `_pc` so a queued grace
        // callback can't dereference a dead handle.
        this._supervisor.clearIceGrace();
        this._supervisor.uninstallNetworkListeners();
        this._supervisor.stopDcWatchdog();
        if (this._stateRefreshInterval) { clearInterval(this._stateRefreshInterval); this._stateRefreshInterval = null; }
        if (this._latencyMonitorId) { clearInterval(this._latencyMonitorId); this._latencyMonitorId = null; }
        if (this._micStream) { this._micStream.getTracks().forEach((t) => t.stop()); this._micStream = null; }
        this._micMuted = true;
        this._micSupported = false;
        if (this._pc) { this._pc.close(); this._pc = null; }
        if (this._dc) { this._dc.close(); this._dc = null; }
        this._iceConnected = false;
        this._dcOpen = false;
        const sessionId = this._sessionId;
        this._sessionId = null;
        return sessionId;
    }

    private _failSessionRejected(msg: SignalingMessage): void {
        const err = new Error(
            msg.reason === 'robot_busy'
                ? `Robot is busy: "${msg.activeApp || 'another app'}" is already connected`
                : `Session rejected: ${msg.reason || 'unknown reason'}`,
        ) as SessionRejectError;
        err.reason = msg.reason ?? null;
        err.activeApp = msg.activeApp ?? null;

        // No endSession: the robot side never granted this session.
        this._closeTransport(err);

        // During an auto-reconnect the rejection is expected noise (the
        // robot side may still hold the dead session for a few seconds) —
        // the loop retries, the app only sees `sessionReconnecting`.
        if (!this._supervisor.redialing) {
            this._emit('sessionRejected', { reason: msg.reason, activeApp: msg.activeApp });
        }
    }

    async stopSession(): Promise<void> {
        // A deliberate stop always wins over a pending auto-reconnect:
        // the in-flight dial attempt (if any) is rejected inside
        // `_closeTransport`, and the loop exits on the cleared flag.
        this._supervisor.cancelRedial();
        this._logSubscribers.clear();
        this._updateProgressSubscribers.clear();

        const sessionId = this._closeTransport(new Error('Session stopped'));
        if (sessionId) {
            await this._sendToServer({ type: 'endSession', sessionId });
        }

        const wasStreaming = this._state === 'streaming';
        if (wasStreaming) {
            this._state = 'connected';
            this._emit('sessionStopped', { reason: 'user' });
        }
    }

    disconnect(): void {
        this._supervisor.cancelRedial();
        if (this._sseAbortController) { this._sseAbortController.abort(); this._sseAbortController = null; }
        this._logSubscribers.clear();
        this._updateProgressSubscribers.clear();

        const sessionId = this._closeTransport(new Error('Disconnected'));
        if (sessionId && this._token) {
            void this._sendToServer({ type: 'endSession', sessionId });
        }

        this._robots = [];
        this._state = 'disconnected';
        this._emit('disconnected', { reason: 'user' });
    }

    // ─── Resilience: supervisor delegation ───────────────────────────────
    // The policy lives in `session-supervisor.ts`; only the transport
    // teardown below stays here because it manipulates the class's own
    // private state.

    /**
     * Enable/disable the automatic session re-dial at runtime.
     * Disabling also aborts any re-dial already in flight. Used by
     * flows that EXPECT the transport to die — e.g. the daemon
     * self-update, whose `systemctl restart` teardown must surface as
     * `sessionStopped` immediately (it's the "install done, rebooting"
     * signal), not get absorbed by ~22 s of doomed reconnect attempts
     * against a robot that is rebooting anyway.
     *
     * Sharp edge: cancelling an in-flight re-dial leaves the session
     * torn down without emitting `sessionStopped` or `error` — the
     * caller owns the terminal event (see `ReachyMiniInstance`).
     */
    setAutoReconnect(enabled: boolean): void {
        this._supervisor.setAutoReconnect(enabled);
    }

    /**
     * Transport-only teardown: everything stopSession() does EXCEPT the
     * user-facing state flip (`sessionStopped` is not emitted, `_state`
     * falls back to 'connected' so the re-dial can start a session).
     * Pending promises that ride the dead channel are settled just like
     * stopSession() settles them — callers see the same failure shape a
     * manual stop would produce; that includes a timed-out dial
     * attempt's own startSession() promise (its resolvers were left
     * armed, and a late signaling reply could otherwise settle them
     * against a closed pc).
     */
    private _teardownForRedial(): void {
        const sessionId = this._closeTransport(new Error('Session reconnecting'));
        // Free the robot-side slot: the relay refuses a second session
        // while it still tracks the dead one, and only endSession (or its
        // own consent-freshness timeout, ~30 s) clears it. Fire and
        // forget — _sendToServer never throws.
        if (sessionId && this._token) {
            void this._sendToServer({ type: 'endSession', sessionId });
        }
        if (this._state === 'streaming') this._state = 'connected';
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

    playRecordedMove(
        moveName: string,
        opts: { dataset?: string; initialGotoDuration?: number } = {},
    ): boolean {
        return this._sendCommand(recordedMoveCmd(moveName, opts));
    }

    /**
     * Like `playRecordedMove()`, but resolves once the daemon acks the
     * dispatch: `true` when the move was loaded and handed to the player,
     * `false` when the daemon could not load it (unknown name, missing
     * dataset), `null` on the fail-open timeout. Rejects when the data
     * channel isn't open.
     *
     * Use this over the fire-and-forget variant whenever the dataset may be
     * cold: the daemon loads the move *before* acking, downloading the
     * dataset on the spot if it isn't cached, so the ack is the only honest
     * "it got that far" signal. Watching `is_move_running` with a short
     * deadline instead makes a slow download look like a missing move.
     *
     * Hence the generous default timeout: it has to cover a multi-MB
     * download on the robot's Wi-Fi, not just a data-channel round trip.
     * Warm the cache with `preloadDatasetAndWait()` first if you'd rather
     * surface the download as its own step.
     *
     * Two things `true` does *not* promise, both daemon-side:
     * - the robot is moving. The daemon acks before starting the player,
     *   which then drops the move if another one is already running. Call
     *   `stopMove()` first when a new move must win.
     * - the move ran to completion. There is no end-of-playback broadcast
     *   for recorded moves (unlike `playMove()`), so a caller that needs
     *   the end still watches `is_move_running` fall - from this ack
     *   rather than from its own call, which is the point.
     */
    playRecordedMoveAndWait(
        moveName: string,
        opts: { dataset?: string; initialGotoDuration?: number; timeoutMs?: number } = {},
    ): Promise<boolean | null> {
        // Key the reply on the move name, not the bare `command` echo:
        // broadcast matching hands a message to the newest matching waiter,
        // so with two `play_recorded_move` commands in flight a bare-echo
        // predicate lets concurrent calls swap results. Error acks also
        // match on status alone because older daemons omit `move_name`
        // there (newer ones include it) - a fully-strict name match would
        // miss those failures and fall through to the timeout.
        return this.request(recordedMoveCmd(moveName, opts), {
            timeoutMs: opts.timeoutMs ?? 120000,
            match: (m) =>
                m.command === 'play_recorded_move' &&
                (m.status === 'error' || m.move_name === moveName),
        }).then((reply) => (reply == null ? null : reply.status === 'ok'));
    }

    /**
     * Stop whatever move is currently playing on the daemon (recorded move,
     * uploaded move, goto). Fire-and-forget and idempotent: the daemon acks
     * ok with `stopped: false` when nothing was running. Returns `false` if
     * the data channel is not open.
     */
    stopMove(): boolean {
        return this._sendCommand({ type: 'stop_move' });
    }

    preloadDataset(dataset: string): boolean {
        return this._sendCommand({ type: 'preload_dataset', dataset_name: dataset });
    }

    /**
     * Like `preloadDataset()`, but resolves once the daemon acks the preload
     * (`{command: "preload_dataset", ...}` on the data channel), i.e. when the
     * dataset is actually in the local HF cache. Resolves `true` on success,
     * `false` when the daemon reports a download failure, and `null` on the
     * fail-open timeout (download slower than `timeoutMs`, or a daemon that
     * predates the command and never replies) - callers should proceed in all
     * three cases, `playRecordedMove` still downloads on demand. Rejects when
     * the data channel isn't open or the session tears down mid-flight.
     */
    preloadDatasetAndWait(
        dataset: string,
        { timeoutMs = 120000 }: { timeoutMs?: number } = {},
    ): Promise<boolean | null> {
        // Send before registering the waiter (same rationale as `request()`):
        // a channel closed mid-flight rejects instead of hanging a waiter to
        // its timeout.
        if (!this.preloadDataset(dataset)) {
            return Promise.reject(new Error('Data channel not open'));
        }
        return this._pending
            .awaitBroadcast(
                (m) => m.command === 'preload_dataset' && m.dataset_name === dataset,
                { timeoutMs, debugLabel: `preload_dataset(${dataset})` },
            )
            .then((m) => m.status === 'ok')
            .catch((err: unknown): null => {
                if (err instanceof BroadcastTimeoutError) return null;
                throw err;
            });
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
        return this._slotRoundtrip('tracked_face', { type: 'get_tracked_face' });
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
        { preRelease = false, onProgress }: StartDaemonUpdateOptions = {},
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
        command: MotionCommand,
        timeoutMs: number,
    ): Promise<void> {
        if (!this._sendCommand({ type: command })) {
            return Promise.reject(new Error(`${command}: data channel not open`));
        }
        return this._pending.awaitMotion(command, timeoutMs);
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
        // Gravity compensation counts as awake - the robot is standing - but
        // it runs under current control, where the daemon ignores position
        // targets outright. An app inheriting that state from its predecessor
        // (a fast handoff cancels the daemon's idle reset, so the mode
        // survives) would see every goto silently do nothing. Flip back to
        // position control without replaying the emote: the robot is already
        // up, and the daemon pins targets to the measured pose on the way in,
        // so nothing snaps.
        if (this._robotState?.motor_mode === 'gravity_compensation') {
            log.info('ensureAwake: gravity compensation inherited, flipping to enabled (no emote)');
            this.setMotorMode('enabled');
            // Refresh the cache so a caller reading isAwake() right after us
            // doesn't see the mode we just left.
            this.requestState();
            return true;
        }
        if (this.isAwake()) {
            log.info('ensureAwake: already awake, nothing to do');
            return true;
        }
        // Await the trajectory: callers treat resolution as "robot ready for
        // position targets", and the emote keeps moving the head for ~2 s
        // after the command is acked - an app that starts commanding poses
        // under it fights it. Failures are swallowed: a torn-down session or
        // a daemon that never acks must not take the whole boot down with it.
        log.info(`ensureAwake: robot asleep (motor_mode=${this._robotState?.motor_mode ?? 'unknown'}), playing wake-up`);
        const wakeStartedAt = Date.now();
        try {
            await this.wakeUp({ timeoutMs: WAKE_TRAJECTORY_BUDGET_MS });
            log.info(`ensureAwake: wake-up trajectory done in ${Date.now() - wakeStartedAt}ms`);
        } catch (e) {
            /* timed out, or the session went away under us */
            log.warn(`ensureAwake: wake-up not confirmed after ${Date.now() - wakeStartedAt}ms (continuing):`, e);
        }
        return true;
    }

    /**
     * Query the daemon version. Resolves `null` when the daemon predates
     * `get_version` (fail-open on the shared slot timeout) or the reply is
     * superseded by session teardown.
     */
    getVersion(): Promise<string | null> {
        return this._slotRoundtrip('version', { type: 'get_version' });
    }

    getHardwareId(): Promise<string | null> {
        return this._slotRoundtrip('hardware_id', { type: 'get_hardware_id' });
    }

    /**
     * One-shot IMU reading (BMI088, wireless version only). Resolves `null`
     * when the robot has no IMU (Lite, simulation) or the daemon predates
     * the `get_imu` command (fail-open on the shared slot timeout).
     */
    getImu(): Promise<ImuData | null> {
        return this._slotRoundtrip('imu', { type: 'get_imu' });
    }

    getVolume(): Promise<number | null> {
        return this._slotRoundtrip('volume', { type: 'get_volume' });
    }

    setVolume(volume: number): Promise<number | null> {
        return this._slotRoundtrip('volume', { type: 'set_volume', volume: clampVolume(volume) });
    }

    getMicrophoneVolume(): Promise<number | null> {
        return this._slotRoundtrip('mic_volume', { type: 'get_microphone_volume' });
    }

    setMicrophoneVolume(volume: number): Promise<number | null> {
        return this._slotRoundtrip('mic_volume', { type: 'set_microphone_volume', volume: clampVolume(volume) });
    }

    /**
     * Query whether the first wake-up setup wizard has been completed.
     * Robot-wide, persisted on the robot. Resolves `false` when pending,
     * `true` when done, or `null` when the channel isn't open / the daemon
     * predates the `get_first_wake_up` command (callers should fail-open
     * and skip the wizard on `null`).
     */
    getFirstWakeUp(): Promise<boolean | null> {
        // Fail-open, so this never rejects: the wizard gate runs right after
        // connect, which is exactly when the channel may not be open yet.
        if (!this._dc || this._dc.readyState !== 'open') return Promise.resolve(null);
        return this._slotRoundtrip('first_wake_up', { type: 'get_first_wake_up' });
    }

    /**
     * Persist the first wake-up wizard completion flag on the robot.
     * Resolves with the stored value (or `null` on channel-closed).
     */
    setFirstWakeUp(isCompleted: boolean): Promise<boolean | null> {
        if (!this._dc || this._dc.readyState !== 'open') return Promise.resolve(null);
        return this._slotRoundtrip('first_wake_up', { type: 'set_first_wake_up', is_completed: isCompleted });
    }

    /**
     * Query the persisted robot display name. Resolves the stored name,
     * `null` when none is set / the channel isn't open / the daemon predates
     * the `get_robot_name` command.
     */
    getRobotName(): Promise<string | null> {
        return this._slotRoundtrip('robot_name', { type: 'get_robot_name' });
    }

    /**
     * Set and persist the robot display name on the robot. Resolves with the
     * stored (trimmed) name, or `null` on error / channel-closed. Applied live
     * by the daemon (status + central relay + mDNS), so it takes effect right
     * away without a restart; the persisted name also overrides --robot-name
     * on the next start.
     */
    setRobotName(name: string): Promise<string | null> {
        return this._slotRoundtrip('robot_name', { type: 'set_robot_name', name });
    }

    /**
     * Sign this robot out of Hugging Face: asks the daemon to delete its
     * stored HF token, which de-registers the robot from the central
     * signaling relay (it disappears from its owner's robot list until it
     * is set up again). Works over the WebRTC data channel, so it reaches
     * the robot remotely (no LAN HTTP path required).
     *
     * Resolves `true` when the daemon acked success, `false` on a daemon
     * error, or `null` when no ack arrives before the timeout (e.g. a
     * daemon that predates the `delete_hf_token` command silently drops
     * it). Rejects if the data channel isn't open. Note the sign-out
     * drops the central relay, so the session may tear down right after
     * the ack - callers should treat a post-call session drop as expected,
     * and a successful sign-out may surface as `null` if teardown races
     * ahead of the ack.
     */
    signOut(): Promise<boolean | null> {
        return this._slotRoundtrip('delete_hf_token', { type: 'delete_hf_token' });
    }

    applyAudioConfig(
        config: AudioConfigEntry[],
        { verify = true }: ApplyAudioConfigOptions = {},
    ): Promise<boolean> {
        return this._slotRoundtrip('apply_audio_config', { type: 'apply_audio_config', config, verify })
            .then((v) => v === true);
    }

    readAudioParameter(name: string): Promise<number[] | null> {
        return this._slotRoundtrip('read_audio_parameter', { type: 'read_audio_parameter', name });
    }

    /**
     * Internal: send a command and await the matching daemon response in a
     * named single-resolver slot (see `PendingReplies.slotRoundtrip`).
     * Every caller has a strict request/response shape where a single
     * in-flight call per slot is sufficient.
     */
    private _slotRoundtrip<K extends ReplySlotKey>(
        slot: K,
        command: Record<string, unknown>,
    ): Promise<ReplySlotValues[K] | null> {
        if (!this._dc || this._dc.readyState !== 'open') {
            return Promise.reject(new Error('Data channel not open'));
        }
        return this._pending.slotRoundtrip(slot, () => { this._sendCommand(command); });
    }

    sendRaw(data: unknown): boolean {
        return this._sendCommand(data);
    }

    /**
     * Generic command round-trip for daemon commands the SDK has no typed
     * wrapper for (yet). Escape hatch so an app can use a newer daemon
     * feature without waiting for an SDK release: sends `command` and
     * resolves with the first robot message whose `command` field equals
     * the sent `type` - the daemon's reply convention - or with `null` on
     * the fail-open timeout (daemon predates the command, or the command
     * is fire-and-forget and never replies).
     *
     * Rejects when the data channel isn't open or the session tears down
     * mid-flight, mirroring the typed wrappers.
     *
     * Replies the SDK already consumes internally (`get_imu`,
     * `get_volume`, ...) are swallowed by their own handlers and never
     * reach this matcher - use the typed wrappers for those. Pass `match`
     * for replies that don't follow the `command` echo convention.
     */
    request(
        command: { type: string } & Record<string, unknown>,
        { timeoutMs = SLOT_ROUNDTRIP_TIMEOUT_MS, match }: RequestOptions = {},
    ): Promise<Record<string, unknown> | null> {
        // Send before registering the waiter (safe: replies arrive on a
        // later task), so a channel closed mid-flight rejects instead of
        // hanging a waiter to its timeout and resolving `null`.
        if (!this._sendCommand(command)) {
            return Promise.reject(new Error('Data channel not open'));
        }
        const predicate = match
            ?? ((m: Record<string, unknown>): boolean => m.command === command.type);
        return this._pending
            .awaitBroadcast(predicate, { timeoutMs, debugLabel: `request(${command.type})` })
            .catch((err: unknown): null => {
                // Fail-open on the waiter timeout only; teardown rejections
                // (settleAll) propagate to the caller like every other
                // in-flight round-trip.
                if (err instanceof BroadcastTimeoutError) return null;
                throw err;
            });
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

    /**
     * Ask the daemon to *push* the robot state (~30 Hz) over the dedicated
     * unreliable/unordered `pose` data channel instead of polling get_state.
     * Fires `state` events as frames arrive. No-op against an older daemon (no
     * pose channel) - fall back to `requestState()` polling there.
     *
     * Refcounted: pair every `subscribePose()` with exactly one
     * `unsubscribePose()`. Multiple consumers share a single daemon-side
     * subscription; the daemon only stops pushing once the last one releases.
     * If the channel isn't open yet (or the session later reconnects), the
     * subscription is (re-)asserted from `_checkSessionReady`.
     */
    subscribePose(): boolean {
        this._poseSubRefs++;
        return this._sendCommand({ type: 'subscribe_pose' });
    }

    /** Release one pose-stream consumer; sends `unsubscribe_pose` on the last. */
    unsubscribePose(): boolean {
        if (this._poseSubRefs > 0) this._poseSubRefs--;
        if (this._poseSubRefs > 0) return true; // still wanted by another consumer
        return this._sendCommand({ type: 'unsubscribe_pose' });
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
            log.warn('playMove.onStarted threw:', e);
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
        detail: ReachyMiniEventMap[K]['detail'],
    ): void {
        this.dispatchEvent(new CustomEvent(name, { detail }));
    }

    private _waitForBroadcast(
        predicate: (m: Record<string, unknown>) => boolean,
        opts: { timeoutMs?: number; debugLabel?: string } = {},
    ): Promise<Record<string, unknown>> {
        return this._pending.awaitBroadcast(predicate, opts);
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
                log.warn(
                    `/send rejected (${res.status}) for type=${(message as { type?: string })?.type}; body=${body || '<empty>'}`,
                );
                return null;
            }
            return await res.json() as SignalingMessage;
        } catch (e) {
            log.error('signaling /send failed:', e);
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
        return this._pending.rpcRoundtrip(method, timeoutMs, (id) =>
            this._sendCommand({ jsonrpc: '2.0', id, method, params }),
        ) as Promise<T>;
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

    private _handleRpcMessage(data: Record<string, unknown>): void {
        // Response to an rpcCall (correlated by id)...
        if (this._pending.settleRpcResponse(data)) return;
        // ...or a one-way notification (event): dispatch to listeners.
        if (typeof data.method === 'string') {
            const params = (data.params as Record<string, unknown> | undefined) ?? {};
            for (const cb of this._rpcListeners.get(data.method) ?? []) {
                try {
                    cb(params);
                } catch (e) {
                    log.error(`onNotification(${data.method}) threw:`, e);
                }
            }
        }
    }

    private _checkSessionReady(): void {
        if (this._iceConnected && this._dcOpen && this._sessionResolve) {
            this._state = 'streaming';
            this.requestState();
            // Re-assert a pose subscription that was requested before the data
            // channel was open, or lost on reconnect (a fresh peer starts
            // unsubscribed on the daemon). Sent raw so it doesn't touch the
            // local refcount, which already reflects the live consumer count.
            if (this._poseSubRefs > 0) this._sendCommand({ type: 'subscribe_pose' });
            // A fresh session has received no pose frame yet, so a timestamp
            // left over from the previous one must not hold off the poll.
            this._lastPoseFrameAt = 0;
            this._stateRefreshInterval = setInterval(() => {
                // Skip while the pose stream is already feeding state; see
                // POSE_STREAM_FRESH_MS.
                if (Date.now() - this._lastPoseFrameAt < POSE_STREAM_FRESH_MS) return;
                this.requestState();
            }, 500);
            this._supervisor.startDcWatchdog();
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
            // Same suppression as _failSessionRejected: retries during an
            // auto-reconnect must not surface as `sessionRejected`.
            if (!this._supervisor.redialing) {
                this._emit('sessionRejected', { reason, activeApp: null });
            }
            // Resilience teardown alongside the PC close path.
            this._supervisor.clearIceGrace();
            this._supervisor.uninstallNetworkListeners();
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
                            log.warn('buffered ICE candidate rejected:', err);
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
            log.error('webrtc signaling failed:', e);
            this._emit('error', { source: 'webrtc', error: e as Error });
        }
    }

    /**
     * @param fromPoseStream Frame came from the pushed `pose` channel rather
     *   than the reliable control channel. Only those may refresh the state
     *   mirror while the stream is live (see the `data.state` branch).
     */
    private _handleRobotMessage(data: Record<string, unknown>, fromPoseStream = false): void {
        // Liveness stamp for the data-channel silence watchdog: every
        // inbound message — control replies, broadcasts, pose frames —
        // proves the transport is alive.
        this._supervisor.stampDcInbound();
        // JSON-RPC frames (app control surface) are handled separately from
        // the legacy {command|type} robot messages that share this channel.
        if (data.jsonrpc === '2.0') {
            this._handleRpcMessage(data);
            return;
        }
        // Bare `version` / `hardware_id` replies carry no `command` key:
        // only swallow them when a waiter is actually pending, otherwise
        // let the message fall through to the branches below.
        if ('version' in data
            && this._pending.settleReplySlot('version', (data.version as string | null) ?? null)) {
            return;
        }
        if ('hardware_id' in data
            && this._pending.settleReplySlot('hardware_id', (data.hardware_id as string | null) ?? null)) {
            return;
        }
        if (data.command === 'get_volume' || data.command === 'set_volume') {
            this._pending.settleReplySlot('volume', data.status === 'error' ? null : (data.volume as number));
            return;
        }
        if (data.command === 'get_microphone_volume' || data.command === 'set_microphone_volume') {
            this._pending.settleReplySlot('mic_volume', data.status === 'error' ? null : (data.volume as number));
            return;
        }
        if (data.command === 'get_first_wake_up' || data.command === 'set_first_wake_up') {
            this._pending.settleReplySlot(
                'first_wake_up',
                data.status === 'error' ? null : !!data.is_completed,
            );
            return;
        }
        if (data.command === 'get_robot_name' || data.command === 'set_robot_name') {
            this._pending.settleReplySlot(
                'robot_name',
                data.status === 'error' ? null : ((data.name as string | null) ?? null),
            );
            return;
        }
        if (data.command === 'delete_hf_token') {
            this._pending.settleReplySlot('delete_hf_token', data.status !== 'error');
            return;
        }
        if (data.command === 'apply_audio_config') {
            this._pending.settleReplySlot('apply_audio_config', data.error ? false : !!data.applied);
            return;
        }
        if (data.command === 'read_audio_parameter') {
            this._pending.settleReplySlot(
                'read_audio_parameter',
                data.error ? null : ((data.values as number[] | undefined) ?? null),
            );
            return;
        }
        if (data.command === 'get_tracked_face') {
            this._pending.settleReplySlot('tracked_face', (data.face_target as FaceTarget | undefined) ?? null);
            return;
        }
        if (data.command === 'get_imu') {
            this._pending.settleReplySlot('imu', (data.imu as ImuData | null | undefined) ?? null);
            return;
        }
        if (
            (data.command === 'wake_up' || data.command === 'goto_sleep')
            && this._pending.settleMotion(data.command as MotionCommand, data)
        ) {
            return;
        }
        if (data.type === 'log_line') {
            for (const sub of this._logSubscribers) {
                try {
                    sub.onLine({ timestamp: data.timestamp as string, line: data.line as string });
                } catch (e) {
                    log.error('subscribeLogs onLine threw:', e);
                }
            }
            return;
        }
        if (data.type === 'log_stream_error') {
            for (const sub of this._logSubscribers) {
                if (typeof sub.onError === 'function') {
                    try { sub.onError(data.error as string); }
                    catch (e) { log.error('subscribeLogs onError threw:', e); }
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
                    catch (e) { log.error('startDaemonUpdate onProgress threw:', e); }
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
                catch (e) { log.error('startDaemonUpdate onProgress threw:', e); }
            }
            return;
        }
        // Only the stream may write the mirror while the stream is live. The
        // poll stands down in that case (see POSE_STREAM_FRESH_MS), but it
        // can't unsend a request already in flight: `get_state` rides the
        // reliable channel, so its reply queues behind whatever else is on it
        // - a `upload_move_*` burst, typically - and can land hundreds of ms
        // after the snapshot it carries. Having no `seq`, it slips past the
        // stale-frame guard and rewinds every consumer to a pose from before
        // the upload, until the next pushed frame puts them back. That reads
        // as a one-frame flick to the pre-move pose right as an animation
        // starts. Nothing is lost by dropping it: pushed frames carry the same
        // fields (daemon-side `build_state_dict` feeds both).
        if (data.state && (fromPoseStream || Date.now() - this._lastPoseFrameAt >= POSE_STREAM_FRESH_MS)) {
            const s = data.state as {
                head_pose?: number[][];
                antennas?: [number, number];
                head_joint_positions?: number[];
                body_yaw?: number;
                motor_mode?: 'enabled' | 'disabled' | 'gravity_compensation';
                is_move_running?: boolean;
                face_target?: FaceTarget;
                doa?: { angle: number; speech_detected: boolean } | null;
                imu?: ImuData | null;
            };
            if (s.head_pose) this._robotState.head = s.head_pose.flat();
            if (s.antennas) this._robotState.antennas = [s.antennas[0], s.antennas[1]];
            if (s.head_joint_positions) this._robotState.head_joint_positions = s.head_joint_positions;
            if (typeof s.body_yaw === 'number') this._robotState.body_yaw = s.body_yaw;
            if (s.motor_mode) this._robotState.motor_mode = s.motor_mode;
            if (typeof s.is_move_running === 'boolean') this._robotState.is_move_running = s.is_move_running;
            if (s.face_target) this._robotState.face_target = s.face_target;
            // DoA is null when there's no mic array / no reading yet - reflect
            // that by clearing our mirror so stale angles don't linger.
            if ('doa' in s) this._robotState.doa = s.doa ?? undefined;
            // Same for the IMU: the daemon sends null once its 0.5 s cache
            // goes stale, so a frozen last reading must not linger either.
            if ('imu' in s) this._robotState.imu = s.imu ?? undefined;
            this._emit('state', { ...this._robotState });
        }
        if (data.error) {
            this._emit('error', { source: 'robot', error: data.error as string });
        }
        if (this._pending.matchBroadcast(data)) return;
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
                    log.debug(`video latency correction: was ${lag.toFixed(2)}s behind`);
                    video.currentTime = end - 0.1;
                }
            }
        }, 2000);
    }
}
