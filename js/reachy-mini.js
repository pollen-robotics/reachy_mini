/**
 * reachy-mini.js — Browser SDK for controlling a Reachy Mini robot over WebRTC.
 * https://github.com/pollen-robotics/reachy-mini
 *
 * QUICK START
 * ───────────
 *   import { ReachyMini } from "./reachy-mini.js";
 *   const robot = new ReachyMini();
 *
 *   // 1. Auth (HuggingFace OAuth — required for the signaling server)
 *   if (!await robot.authenticate()) { robot.login(); return; }
 *
 *   // 2. Connect to signaling server (SSE)
 *   await robot.connect();
 *
 *   // 3. Pick a robot once the list arrives
 *   robot.addEventListener("robotsChanged", (e) => {
 *       const robots = e.detail.robots;  // [{ id, meta: { name } }, ...]
 *   });
 *
 *   // 4. Start a WebRTC session (resolves when video + data channel ready)
 *   const detach = robot.attachVideo(document.querySelector("video"));
 *   await robot.startSession(robotId);
 *
 *   // 5. Send commands — degree-friendly helpers, all built on setTarget()
 *   robot.setHeadRpyDeg(0, 10, -5);   // roll, pitch, yaw in degrees
 *   robot.setAntennasDeg(30, -30);    // right, left in degrees
 *   robot.setBodyYawDeg(15);          // body yaw in degrees
 *
 *   // …or compose an atomic update in raw wire units (full SE(3); no XYZ loss):
 *   robot.setTarget({
 *       head: rpyToMatrix(0, 10, -5).flat(),  // number[16] flat row-major 4×4
 *       antennas: [degToRad(30), degToRad(-30)],
 *       body_yaw: degToRad(15),
 *   });
 *   robot.playSound("wake_up.wav");   // filename on robot
 *   const ver = await robot.getVersion(); // e.g. "1.5.1"
 *
 *   // 6. Receive live state (emitted every ~500 ms while streaming; call
 *   //    robot.requestState() yourself for higher rates — see its JSDoc).
 *   //    State payload is the daemon's raw wire shape — use the math
 *   //    utilities exported from this module for degree conversions.
 *   robot.addEventListener("state", (e) => {
 *       const { head, antennas, body_yaw, motor_mode, is_move_running } = e.detail;
 *       // head:            number[16]   — flat row-major 4×4 (full SE(3))
 *       // antennas:        [rightRad, leftRad]
 *       // body_yaw:        number       — radians
 *       // motor_mode:      "enabled" | "disabled" | "gravity_compensation"
 *       // is_move_running: boolean
 *       // For human-friendly head RPY:
 *       //   const rpy = matrixToRpy(head);   // { roll, pitch, yaw } in degrees
 *   });
 *
 *   // 7. Audio controls
 *   robot.setAudioMuted(false);   // unmute robot speaker (muted by default)
 *   robot.setMicMuted(false);     // unmute your mic → robot speaker (if supported)
 *
 *   // 8. Cleanup
 *   detach();                      // remove video binding
 *   await robot.stopSession();     // back to 'connected'
 *   robot.disconnect();            // back to 'disconnected' (keeps auth)
 *   robot.logout();                // clear HF credentials too
 *
 *
 * STATE MACHINE
 * ─────────────
 *   'disconnected' ──connect()──▸ 'connected' ──startSession()──▸ 'streaming'
 *        ▴ disconnect()                ▴ stopSession()
 *        └─────────────────────────────┘
 *
 *
 * CONSTRUCTOR OPTIONS
 * ───────────────────
 *   new ReachyMini({
 *     signalingUrl:              string,   // default: "https://cduss-reachy-mini-central.hf.space"
 *     enableMicrophone:          boolean,  // default: true  — acquire mic for bidirectional audio
 *     videoJitterBufferTargetMs: number,   // default: 0     — receiver-side jitter buffer hint, ms
 *                                          //                  0 = "render ASAP" (teleop). Spec range [0, 4000].
 *                                          //                  Raise (100–400) on flaky links to trade latency for resilience.
 *   })
 *
 *
 * READ-ONLY PROPERTIES
 * ────────────────────
 *   .state            "disconnected" | "connected" | "streaming"
 *   .robots           Array<{ id: string, meta: { name: string } }>
 *   .robotState       Mirror of the latest "state" event detail —
 *                     { head: number[16], antennas: [rightRad, leftRad],
 *                       body_yaw, motor_mode, is_move_running }
 *                     (fields only present once the daemon sends them;
 *                      see EVENTS below)
 *   .username         string | null     — HF username after authenticate()
 *   .isAuthenticated  boolean           — true if a valid HF token is available
 *   .micSupported     boolean           — true if robot offers bidirectional audio
 *   .micMuted         boolean           — your microphone mute state
 *   .audioMuted       boolean           — robot speaker mute state (local)
 *
 *
 * EVENTS  (EventTarget — use addEventListener)
 * ──────────────────────────────────────────────
 *   "connected"       { peerId: string }
 *   "disconnected"    { reason: string }
 *   "robotsChanged"   { robots: Array<{ id, meta }> }
 *   "streaming"       { sessionId: string, robotId: string }
 *   "sessionStopped"  { reason: string }
 *   "state"           { head: number[16],                    // flat row-major 4×4, when daemon sends head_pose
 *                       antennas: [rightRad, leftRad],       // when daemon sends antennas
 *                       body_yaw: number,                    // radians, when daemon sends body_yaw
 *                       motor_mode: string,                  // when daemon sends motor_mode
 *                       is_move_running: boolean }           // when daemon sends is_move_running
 *   "videoTrack"      { track: MediaStreamTrack, stream: MediaStream }
 *   "micSupported"    { supported: boolean }
 *   "error"           { source: "signaling"|"webrtc"|"robot", error: Error|string }
 *
 *
 * EXPORTS
 * ───────
 *   export default ReachyMini;
 *   export { ReachyMini, rpyToMatrix, matrixToRpy, degToRad, radToDeg };
 */

import {
    oauthHandleRedirectIfPresent,
    oauthLoginUrl,
} from "https://cdn.jsdelivr.net/npm/@huggingface/hub@0.15.2/+esm";

// ─── Math utilities ──────────────────────────────────────────────────────────

/** @param {number} deg @returns {number} */
export function degToRad(deg) { return deg * Math.PI / 180; }

/** @param {number} rad @returns {number} */
export function radToDeg(rad) { return rad * 180 / Math.PI; }

/**
 * Roll/pitch/yaw (degrees) → 4×4 rotation matrix (ZYX convention).
 * This is the wire format for the robot's `set_target` command.
 * @param {number} rollDeg  @param {number} pitchDeg  @param {number} yawDeg
 * @returns {number[][]} 4×4 matrix
 */
export function rpyToMatrix(rollDeg, pitchDeg, yawDeg) {
    const r = degToRad(rollDeg), p = degToRad(pitchDeg), y = degToRad(yawDeg);
    const cy = Math.cos(y), sy = Math.sin(y);
    const cp = Math.cos(p), sp = Math.sin(p);
    const cr = Math.cos(r), sr = Math.sin(r);
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0],
        [-sp, cp * sr, cp * cr, 0],
        [0, 0, 0, 1],
    ];
}

/**
 * Rotation matrix (3×3 or 4×4) → { roll, pitch, yaw } in degrees.
 * @param {number[][]} m  @returns {{ roll: number, pitch: number, yaw: number }}
 */
export function matrixToRpy(m) {
    return {
        roll: radToDeg(Math.atan2(m[2][1], m[2][2])),
        pitch: radToDeg(Math.asin(-m[2][0])),
        yaw: radToDeg(Math.atan2(m[1][0], m[0][0])),
    };
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/** Clamp a volume to [0, 100] and round to integer — mirrors the server-side
 *  Field(..., ge=0, le=100) validator so calling setVolume(150) doesn't 400. */
function clampVolume(v) {
    const n = Math.round(Number(v) || 0);
    return Math.max(0, Math.min(100, n));
}

/** Check if the audio m= section of an SDP has a=sendrecv (bidirectional audio). */
function sdpHasAudioSendRecv(sdp) {
    const lines = sdp.split('\r\n');
    let inAudio = false;
    for (const line of lines) {
        if (line.startsWith('m=audio')) inAudio = true;
        else if (line.startsWith('m=')) inAudio = false;
        if (inAudio && line === 'a=sendrecv') return true;
    }
    return false;
}

// ─── ReachyMini class ────────────────────────────────────────────────────────

export class ReachyMini extends EventTarget {

    /** @param {{ signalingUrl?: string, enableMicrophone?: boolean, clientId?: string, appName?: string, videoJitterBufferTargetMs?: number }} [options] */
    constructor(options = {}) {
        super();
        this._signalingUrl = options.signalingUrl || 'https://cduss-reachy-mini-central.hf.space';
        this._enableMicrophone = options.enableMicrophone !== false;
        this._clientId = options.clientId || null;
        this._appName = options.appName || 'unknown';
        // Hint to the receiver's WebRTC jitter buffer (ms). 0 = "render ASAP",
        // appropriate for teleop. Spec range [0, 4000]. Browsers that don't
        // implement RTCRtpReceiver.jitterBufferTarget fall back to default
        // buffering (~150-200 ms).
        this._videoJitterBufferTargetMs = options.videoJitterBufferTargetMs ?? 0;

        this._state = 'disconnected';                 // 'disconnected' | 'connected' | 'streaming'
        this._robots = [];                             // latest robot list from signaling
        this._robotState = {};                         // populated from daemon state events (wire shape)

        // Auth
        this._token = null;
        this._username = null;
        this._tokenExpires = null;

        // Signaling
        this._peerId = null;
        this._sseAbortController = null;

        // WebRTC
        this._pc = null;           // RTCPeerConnection
        this._dc = null;           // RTCDataChannel (robot commands)
        this._sessionId = null;
        this._selectedRobotId = null;

        // Audio
        this._micStream = null;    // MediaStream from getUserMedia
        this._micMuted = true;
        this._audioMuted = true;
        this._micSupported = false; // set after SDP negotiation

        // Timers
        this._latencyMonitorId = null;
        this._stateRefreshInterval = null;

        // getVersion() promise plumbing
        this._versionResolve = null;

        // Volume getter/setter promise plumbing (get_volume / set_volume).
        // Speaker and microphone are tracked separately so two in-flight
        // requests can't collide on the same slot.
        this._volumeResolve = null;
        this._micVolumeResolve = null;

        // startSession() promise plumbing
        this._sessionResolve = null;
        this._sessionReject = null;
        this._iceConnected = false;
        this._dcOpen = false;

        // Set by attachVideo()
        this._videoElement = null;
    }

    // ─── Read-only properties ────────────────────────────────────────────

    /** @returns {"disconnected"|"connected"|"streaming"} */
    get state() { return this._state; }

    /** @returns {Array<{id: string, meta: {name: string}}>} */
    get robots() { return this._robots; }

    /**
     * Latest robot state (same shape as the "state" event detail).
     * Mirrors the daemon's wire format — fields appear only once the
     * daemon has sent the corresponding source field. Use the exported
     * math utilities (``matrixToRpy``, ``radToDeg``) for human units.
     * @returns {{
     *   head?: number[],
     *   antennas?: number[],
     *   body_yaw?: number,
     *   motor_mode?: "enabled"|"disabled"|"gravity_compensation",
     *   is_move_running?: boolean,
     * }}
     */
    get robotState() { return this._robotState; }

    /** @returns {string|null} HuggingFace username, set after authenticate(). */
    get username() { return this._username; }

    /** @returns {boolean} True if a valid HF token is available. */
    get isAuthenticated() { return !!this._token; }

    /** @returns {boolean} True if the robot's SDP offered bidirectional audio. */
    get micSupported() { return this._micSupported; }

    /** @returns {boolean} */
    get micMuted() { return this._micMuted; }

    /** @returns {boolean} */
    get audioMuted() { return this._audioMuted; }

    // ─── Auth ────────────────────────────────────────────────────────────

    /**
     * Check for a valid HuggingFace token.
     * Tries the OAuth redirect callback first, then falls back to sessionStorage.
     * @returns {Promise<boolean>} true → token ready, false → call login()
     */
    async authenticate() {
        try {
            const result = await oauthHandleRedirectIfPresent();
            if (result) {
                this._username = result.userInfo.name || result.userInfo.preferred_username;
                this._token = result.accessToken;
                this._tokenExpires = result.accessTokenExpiresAt;
                sessionStorage.setItem('hf_token', this._token);
                sessionStorage.setItem('hf_username', this._username);
                sessionStorage.setItem('hf_token_expires', this._tokenExpires);
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

    /** Redirect the browser to the HuggingFace OAuth login page. */
    async login() {
        const opts = {};
        if (this._clientId) opts.clientId = this._clientId;
        window.location.href = await oauthLoginUrl(opts);
    }

    /** Clear stored HF credentials and disconnect everything. */
    logout() {
        sessionStorage.removeItem('hf_token');
        sessionStorage.removeItem('hf_username');
        sessionStorage.removeItem('hf_token_expires');
        this._username = null;
        this._tokenExpires = null;
        this.disconnect();
    }

    // ─── Lifecycle ───────────────────────────────────────────────────────

    /**
     * Open SSE signaling connection.  Resolves once the server sends `welcome`.
     * Emits "robotsChanged" as robots come and go.
     * @param {string} [token] — HF access token.  Omit to use the one from authenticate().
     * @returns {Promise<void>}
     */
    async connect(token) {
        if (this._state !== 'disconnected') throw new Error('Already connected');
        if (token) this._token = token;
        if (!this._token) throw new Error('No token — call authenticate() first or pass a token');
        this._sseAbortController = new AbortController();

        let res;
        try {
            // Token goes in the Authorization header, not the URL —
            // keeps it out of DevTools Network tab, browser history,
            // Referer, and any server/proxy access log. We use fetch
            // + manual stream reader (below) rather than EventSource
            // specifically to allow custom headers.
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

        return new Promise((resolve, reject) => {
            let welcomed = false;
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            const readLoop = async () => {
                try {
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop();
                        for (const line of lines) {
                            if (!line.startsWith('data:')) continue;
                            try {
                                const msg = JSON.parse(line.slice(5).trim());
                                if (!welcomed && msg.type === 'welcome') {
                                    welcomed = true;
                                    this._peerId = msg.peerId;
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
                            } catch (_) { /* malformed JSON — skip */ }
                        }
                    }
                } catch (e) {
                    if (e.name !== 'AbortError') {
                        this._emit('error', { source: 'signaling', error: e });
                    }
                    if (!welcomed) { reject(e); return; }
                }
                // SSE stream ended (server closed or network drop)
                if (this._state !== 'disconnected') {
                    this._state = 'disconnected';
                    this._emit('disconnected', { reason: 'SSE closed' });
                }
                if (!welcomed) reject(new Error('Connection closed before welcome'));
            };

            readLoop();
        });
    }

    /**
     * Start a WebRTC session with the given robot.
     * Acquires the microphone (if enabled), negotiates SDP, and waits for
     * both ICE connection and data channel to be ready before resolving.
     * Emits "videoTrack" when the robot's camera stream arrives.
     * Emits "micSupported" once SDP negotiation reveals whether the robot
     * accepts bidirectional audio.
     * @param {string} robotId — one of the ids from the robots list
     * @returns {Promise<void>}
     */
    async startSession(robotId) {
        if (this._state !== 'connected') throw new Error('Not connected');
        this._selectedRobotId = robotId;
        this._iceConnected = false;
        this._dcOpen = false;
        this._micSupported = false;

        // Acquire mic eagerly so the browser permission prompt appears now,
        // but tracks stay disabled (muted) until the user explicitly unmutes.
        if (this._enableMicrophone) {
            try {
                this._micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                this._micStream.getAudioTracks().forEach(t => { t.enabled = false; });
                this._micMuted = true;
            } catch (e) {
                console.warn('Microphone not available:', e);
                this._micStream = null;
            }
        }

        this._pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
        });

        return new Promise((resolve, reject) => {
            this._sessionResolve = resolve;
            this._sessionReject = reject;

            this._pc.ontrack = (e) => {
                if (e.track.kind === 'video') {
                    // Tell the receiver's jitter buffer to minimise its hold
                    // time. Both properties target the same internal buffer;
                    // browsers ignore whichever they don't implement.
                    const ms = this._videoJitterBufferTargetMs;
                    try { e.receiver.jitterBufferTarget = ms; } catch (_) {}
                    try { e.receiver.playoutDelayHint = ms / 1000; } catch (_) {}
                    this._emit('videoTrack', { track: e.track, stream: e.streams[0] });
                }
            };

            this._pc.onicecandidate = async (e) => {
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

            this._pc.oniceconnectionstatechange = () => {
                const s = this._pc?.iceConnectionState;
                if (!s) return;
                if (s === 'connected' || s === 'completed') {
                    this._iceConnected = true;
                    this._checkSessionReady();
                } else if (s === 'failed') {
                    const err = new Error('ICE connection failed');
                    if (this._sessionReject) {
                        this._sessionReject(err);
                        this._sessionResolve = null;
                        this._sessionReject = null;
                    }
                    this._emit('error', { source: 'webrtc', error: err });
                } else if (s === 'disconnected') {
                    this._emit('error', { source: 'webrtc', error: new Error('ICE disconnected') });
                }
            };

            this._pc.ondatachannel = (e) => {
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

    /**
     * Internal: handle a sessionRejected response from central.
     * Releases resources allocated by startSession() (RTCPeerConnection,
     * microphone stream) and rejects the pending startSession() promise
     * with an Error carrying `.reason` and `.activeApp`.
     *
     * Called from both the POST-response path (primary) and the SSE
     * handler (defensive, in case the server changes).
     */
    _failSessionRejected(msg) {
        const err = new Error(
            msg.reason === 'robot_busy'
                ? `Robot is busy: "${msg.activeApp || 'another app'}" is already connected`
                : `Session rejected: ${msg.reason || 'unknown reason'}`
        );
        err.reason = msg.reason;
        err.activeApp = msg.activeApp;

        // Release resources allocated optimistically in startSession()
        // before we knew the server would refuse.
        if (this._pc) { this._pc.close(); this._pc = null; }
        if (this._micStream) { this._micStream.getTracks().forEach(t => t.stop()); this._micStream = null; }
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

    /**
     * End the WebRTC session.  Returns to "connected" state so you can
     * startSession() again with the same or a different robot.
     * @returns {Promise<void>}
     */
    async stopSession() {
        if (this._versionResolve) { this._versionResolve(null); this._versionResolve = null; }
        if (this._volumeResolve) { this._volumeResolve(null); this._volumeResolve = null; }
        if (this._micVolumeResolve) { this._micVolumeResolve(null); this._micVolumeResolve = null; }
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

        if (this._micStream) { this._micStream.getTracks().forEach(t => t.stop()); this._micStream = null; }
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

    /**
     * Full teardown — abort SSE, close WebRTC.
     * Auth state is preserved (call logout() to also clear credentials).
     */
    disconnect() {
        if (this._sseAbortController) { this._sseAbortController.abort(); this._sseAbortController = null; }

        if (this._versionResolve) { this._versionResolve(null); this._versionResolve = null; }
        if (this._volumeResolve) { this._volumeResolve(null); this._volumeResolve = null; }
        if (this._micVolumeResolve) { this._micVolumeResolve(null); this._micVolumeResolve = null; }
        if (this._sessionReject) {
            this._sessionReject(new Error('Disconnected'));
            this._sessionResolve = null;
            this._sessionReject = null;
        }

        if (this._stateRefreshInterval) { clearInterval(this._stateRefreshInterval); this._stateRefreshInterval = null; }
        if (this._latencyMonitorId) { clearInterval(this._latencyMonitorId); this._latencyMonitorId = null; }

        if (this._sessionId && this._token) {
            this._sendToServer({ type: 'endSession', sessionId: this._sessionId }); // fire-and-forget
        }

        if (this._micStream) { this._micStream.getTracks().forEach(t => t.stop()); this._micStream = null; }
        if (this._pc) { this._pc.close(); this._pc = null; }
        if (this._dc) { this._dc.close(); this._dc = null; }

        this._sessionId = null;
        this._micMuted = true;
        this._micSupported = false;
        this._iceConnected = false;
        this._dcOpen = false;
        this._peerId = null;
        this._robots = [];
        this._state = 'disconnected';
        this._emit('disconnected', { reason: 'user' });
    }

    // ─── Commands ────────────────────────────────────────────────────────
    // All return false if the data channel is not open, true if sent.

    /**
     * Send a target pose to the robot. Wire-shape, raw units only —
     * single source of truth for motion commands. Every field is
     * optional; omitted fields leave the daemon's previous target
     * unchanged, so partial updates compose naturally.
     *
     * For human units (degrees), use the ``setHeadRpyDeg`` /
     * ``setAntennasDeg`` / ``setBodyYawDeg`` thin wrappers below.
     *
     * @param {object} [target]
     * @param {number[]} [target.head] 16-element flat row-major 4×4
     *   matrix (full SE(3); preserves translation, no XYZ loss).
     * @param {number[]} [target.antennas] ``[rightRad, leftRad]``.
     * @param {number} [target.body_yaw] Body yaw in radians.
     * @returns {boolean} false if the data channel is not open.
     * @throws {TypeError} if any provided field has the wrong shape or
     *   contains a non-finite value (NaN, Infinity). Validation runs at
     *   the JS boundary so caller mistakes surface with a stack trace
     *   pointing to the call site, not as a confusing daemon-side error.
     */
    setTarget({ head, antennas, body_yaw } = {}) {
        const cmd = { type: "set_full_target" };
        if (head !== undefined) {
            if (!Array.isArray(head) || head.length !== 16
                || !head.every((n) => Number.isFinite(n))) {
                throw new TypeError(
                    'setTarget: head must be a 16-element flat row-major 4×4 matrix '
                    + `of finite numbers; got ${Array.isArray(head) ? `Array(${head.length})` : typeof head}`
                );
            }
            cmd.head = head;
        }
        if (antennas !== undefined) {
            if (!Array.isArray(antennas) || antennas.length !== 2
                || !antennas.every((n) => Number.isFinite(n))) {
                throw new TypeError(
                    'setTarget: antennas must be [rightRad, leftRad] (2 finite numbers); '
                    + `got ${Array.isArray(antennas) ? `Array(${antennas.length})` : typeof antennas}`
                );
            }
            cmd.antennas = antennas;
        }
        if (body_yaw !== undefined) {
            if (!Number.isFinite(body_yaw)) {
                throw new TypeError(
                    `setTarget: body_yaw must be a finite number (radians); got ${body_yaw}`
                );
            }
            cmd.body_yaw = body_yaw;
        }
        return this._sendCommand(cmd);
    }

    /**
     * Set head orientation from roll/pitch/yaw in degrees.
     * Convenience wrapper over ``setTarget``.
     * @param {number} rollDeg @param {number} pitchDeg @param {number} yawDeg
     * @returns {boolean}
     */
    setHeadRpyDeg(rollDeg, pitchDeg, yawDeg) {
        return this.setTarget({ head: rpyToMatrix(rollDeg, pitchDeg, yawDeg).flat() });
    }

    /**
     * Set antenna positions from degrees.
     * Convenience wrapper over ``setTarget``.
     * @param {number} rightDeg @param {number} leftDeg
     * @returns {boolean}
     */
    setAntennasDeg(rightDeg, leftDeg) {
        return this.setTarget({ antennas: [degToRad(rightDeg), degToRad(leftDeg)] });
    }

    /**
     * Set body yaw from degrees.
     * Convenience wrapper over ``setTarget``.
     * @param {number} yawDeg
     * @returns {boolean}
     */
    setBodyYawDeg(yawDeg) {
        return this.setTarget({ body_yaw: degToRad(yawDeg) });
    }

    /**
     * Play a sound file on the robot.
     * @param {string} file — filename available on the robot (e.g. "wake_up.wav")
     * @returns {boolean}
     */
    playSound(file) {
        return this._sendCommand({ type: "play_sound", file });
    }

    /**
     * Set the motor control mode.
     *
     * @param {"enabled"|"disabled"|"gravity_compensation"} mode
     *   - "enabled"              torque on, position-controlled.
     *   - "disabled"             torque off; the robot is backdrivable
     *                            and will not hold any pose.
     *   - "gravity_compensation" torque on in current-control mode;
     *                            motors actively cancel gravity so the
     *                            robot is easy to move by hand.
     * @returns {boolean} false if the data channel is not open.
     */
    setMotorMode(mode) {
        return this._sendCommand({ type: "set_motor_mode", mode });
    }

    /**
     * Play the wake-up animation (full head/antennas trajectory on the
     * robot, ~2 s). Fire-and-forget — poll ``requestState()`` and watch
     * ``is_move_running`` if you need to know when it finishes.
     *
     * This helper sends a ``set_motor_mode: "enabled"`` command *before*
     * the ``wake_up`` command so the animation actually moves the motors.
     * The robot's ``wake_up`` handler does not touch motor mode itself;
     * if torque is off when the trajectory runs, the commanded positions
     * are silently ignored and the robot stays limp. Both commands
     * travel over the same data channel so ordering at the backend is
     * preserved.
     *
     * Semantics match the REST endpoint ``POST /api/move/play/wake_up``
     * plus the LAN convention of enabling motors before playing motion
     * trajectories.
     *
     * @returns {boolean} false if the data channel is not open.
     */
    wakeUp() {
        this._sendCommand({ type: "set_motor_mode", mode: "enabled" });
        return this._sendCommand({ type: "wake_up" });
    }

    /**
     * Play the goto-sleep animation. Fire-and-forget; see ``wakeUp`` for
     * progress-polling notes.
     *
     * Does NOT touch motor mode: the daemon's ``goto_sleep`` handler
     * manages the transition out of torque on its own (motors must stay
     * powered during the trajectory to move into the sleep pose, then
     * are typically disabled by the daemon once the pose is reached).
     *
     * Semantics match ``POST /api/move/play/goto_sleep`` and the
     * ``"goto_sleep"`` WebRTC command.
     *
     * @returns {boolean} false if the data channel is not open.
     */
    gotoSleep() {
        return this._sendCommand({ type: "goto_sleep" });
    }

    /**
     * Whether the robot's motors are currently powered (the "awake" state).
     *
     * Reads ``motor_mode`` from the last state event. Both ``"enabled"``
     * and ``"gravity_compensation"`` count as awake: in gravity-comp the
     * motors are actively holding the arm against gravity, so the robot
     * is *not* limp and playing wake_up on top would fight the user.
     * Only ``"disabled"`` (true sleep) is considered not-awake.
     *
     * Returns ``false`` before the first state event arrives (typical
     * right after ``startSession()``). Use ``ensureAwake()`` if you want
     * to wait for the first state before deciding.
     *
     * @returns {boolean}
     */
    isAwake() {
        const mode = this._robotState?.motor_mode;
        return mode === "enabled" || mode === "gravity_compensation";
    }

    /**
     * Wake the robot up if it is currently asleep, otherwise no-op.
     *
     * Intended as the first line of any app after ``startSession()``
     * resolves — robots are often left in the sleep pose (torque off,
     * head resting on the base) and commanded positions are silently
     * ignored in that state.
     *
     * If no state event has arrived yet, waits up to ``timeoutMs`` for
     * one before deciding. If still no state, falls back to sending
     * ``wakeUp()`` (safe: the daemon's wake_up handler is idempotent
     * at the motion level — it moves to the awake pose from wherever
     * the head currently is).
     *
     * @param {number} [timeoutMs=1000] how long to wait for the first
     *   state event before falling through to wakeUp().
     * @returns {Promise<boolean>} true if the robot is awake afterwards.
     */
    async ensureAwake(timeoutMs = 1000) {
        if (this._robotState?.motor_mode === undefined) {
            await new Promise((resolve) => {
                const done = () => {
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
        this.wakeUp();
        return true;
    }

    /**
     * Request the daemon version.
     * Resolves with the version string (or null if unavailable).
     * @returns {Promise<string|null>}
     */
    getVersion() {
        return new Promise((resolve, reject) => {
            if (!this._dc || this._dc.readyState !== 'open') {
                reject(new Error('Data channel not open'));
                return;
            }
            if (this._versionResolve) {
                this._versionResolve(null);
            }
            this._versionResolve = resolve;
            this._sendCommand({ type: "get_version" });
        });
    }

    /**
     * Query the current speaker volume (0-100).
     * Resolves with null if volume control is unavailable (platform unsupported
     * or audio stack down).
     * @returns {Promise<number|null>}
     */
    getVolume() {
        return this._volumeRoundtrip({ type: "get_volume" }, "_volumeResolve");
    }

    /**
     * Set the speaker volume (0-100). Persists for the next connection
     * (same semantics as the REST /api/volume/set endpoint).
     * Resolves with the applied volume, or null on failure.
     * @param {number} volume 0-100
     * @returns {Promise<number|null>}
     */
    setVolume(volume) {
        return this._volumeRoundtrip(
            { type: "set_volume", volume: clampVolume(volume) },
            "_volumeResolve",
        );
    }

    /**
     * Query the current microphone input volume (0-100).
     * @returns {Promise<number|null>}
     */
    getMicrophoneVolume() {
        return this._volumeRoundtrip(
            { type: "get_microphone_volume" },
            "_micVolumeResolve",
        );
    }

    /**
     * Set the microphone input volume (0-100). Persists across sessions.
     * @param {number} volume 0-100
     * @returns {Promise<number|null>}
     */
    setMicrophoneVolume(volume) {
        return this._volumeRoundtrip(
            { type: "set_microphone_volume", volume: clampVolume(volume) },
            "_micVolumeResolve",
        );
    }

    /**
     * Internal: send a volume command and await the single-slot response.
     * The slot name selects which resolver (speaker vs mic) owns the
     * pending request so the two can be in-flight concurrently without
     * collision.
     */
    _volumeRoundtrip(command, slot) {
        return new Promise((resolve, reject) => {
            if (!this._dc || this._dc.readyState !== 'open') {
                reject(new Error('Data channel not open'));
                return;
            }
            // If a previous request on the same slot is still pending,
            // resolve it to null so its caller doesn't hang forever.
            if (this[slot]) this[slot](null);
            this[slot] = resolve;
            this._sendCommand(command);
        });
    }

    /**
     * Send an arbitrary JSON command over the data channel.
     * @param {object} data  @returns {boolean}
     */
    sendRaw(data) {
        return this._sendCommand(data);
    }

    /**
     * Request a state snapshot.  The response arrives as a "state" event.
     * Called automatically every 500 ms while streaming.
     *
     * Safe to call at a higher rate if you need faster telemetry: e.g.
     * ``setInterval(() => robot.requestState(), 20)`` for ~50 Hz, or drive
     * it from a ``requestAnimationFrame`` loop for display-rate updates.
     * On LAN the daemon can sustain ~90-100 Hz round-trips over the
     * datachannel; over the internet expect the WebRTC path's RTT to
     * dominate. The built-in 500 ms poll keeps running in parallel — it
     * is harmless, as state responses are idempotent.
     *
     * @returns {boolean}
     */
    requestState() {
        return this._sendCommand({ type: "get_state" });
    }

    // ─── Audio ───────────────────────────────────────────────────────────

    /**
     * Mute/unmute the robot's audio playback (speaker) locally.
     * Audio is muted by default — browsers require a user gesture to unmute.
     * @param {boolean} muted
     */
    setAudioMuted(muted) {
        this._audioMuted = muted;
        if (this._videoElement) this._videoElement.muted = muted;
    }

    /**
     * Mute/unmute your microphone.  Only works if micSupported is true.
     * Mic is muted by default even after acquisition.
     * @param {boolean} muted
     */
    setMicMuted(muted) {
        this._micMuted = muted;
        if (this._micStream) {
            this._micStream.getAudioTracks().forEach(t => { t.enabled = !muted; });
        }
    }

    // ─── Video helper ────────────────────────────────────────────────────

    /**
     * Bind a `<video>` element to this robot's stream.
     * Call before startSession().  Sets srcObject when the video track arrives,
     * applies audio mute state, and runs a latency monitor that snaps to the
     * live edge if the buffer grows > 0.5 s.
     *
     * @param {HTMLVideoElement} videoElement
     * @returns {() => void} cleanup function — call to detach video and stop monitoring
     */
    attachVideo(videoElement) {
        this._videoElement = videoElement;
        videoElement.muted = this._audioMuted;

        const onVideoTrack = (e) => {
            videoElement.srcObject = e.detail.stream;
            videoElement.playsInline = true;
            if ('requestVideoFrameCallback' in videoElement) {
                this._startLatencyMonitor(videoElement);
            }
        };

        const onSessionStopped = () => { videoElement.srcObject = null; };

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

    // ─── Private ─────────────────────────────────────────────────────────

    _emit(name, detail) {
        this.dispatchEvent(new CustomEvent(name, { detail }));
    }

    async _sendToServer(message) {
        // Mirrors connect()'s guard — a missing token between connect and
        // send (e.g. logout mid-session) would otherwise produce
        // "Authorization: Bearer undefined", which central correctly 401s
        // but silently returns null to the caller, hiding the real cause.
        if (!this._token) throw new Error('No token — authenticate() first');
        try {
            // Token in Authorization header, not URL — same reasoning as
            // connect()'s SSE fetch: never put secrets in URLs.
            const res = await fetch(`${this._signalingUrl}/send`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this._token}`,
                },
                body: JSON.stringify(message),
            });
            return await res.json();
        } catch (e) {
            console.error('Send error:', e);
            return null;
        }
    }

    _sendCommand(cmd) {
        if (!this._dc || this._dc.readyState !== 'open') return false;
        this._dc.send(JSON.stringify(cmd));
        return true;
    }

    /** Resolves the startSession() promise once both ICE and datachannel are ready. */
    _checkSessionReady() {
        if (this._iceConnected && this._dcOpen && this._sessionResolve) {
            this._state = 'streaming';
            this.requestState();
            this._stateRefreshInterval = setInterval(() => this.requestState(), 500);
            this._emit('streaming', { sessionId: this._sessionId, robotId: this._selectedRobotId });
            this._sessionResolve();
            this._sessionResolve = null;
            this._sessionReject = null;
        }
    }

    async _handleSignalingMessage(msg) {
        switch (msg.type) {
            case 'welcome':
                break; // handled in connect()
            case 'list':
                this._robots = msg.producers || [];
                this._emit('robotsChanged', { robots: this._robots });
                break;
            case 'peerStatusChanged': {
                const list = await this._sendToServer({ type: 'list' });
                if (list?.producers) {
                    this._robots = list.producers;
                    this._emit('robotsChanged', { robots: this._robots });
                }
                break;
            }
            case 'sessionStarted':
                this._sessionId = msg.sessionId;
                break;
            case 'sessionRejected':
                // Defensive: the current central server returns sessionRejected
                // as the direct POST response (handled at the startSession() call
                // site). This branch is a safety net in case the server ever
                // pushes the rejection over SSE instead.
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

    /**
     * Internal: handle an endSession pushed from central.
     *
     * Two user-visible scenarios converge here:
     *
     * 1. Pending startSession — central accepted our request but the
     *    robot-side relay then refused (e.g. a local Python app holds
     *    the daemon's robot lock). Without this handler, the client's
     *    startSession() promise would hang forever because ICE/datachannel
     *    never come up.
     * 2. Streaming was evicted — a local Python app started on the robot
     *    while we were connected, so the relay tore down our session.
     *
     * The ``reason`` is forwarded verbatim from the relay through central:
     * - "robot_busy_local_app": daemon lock held by a local Python app
     *   (refused before any media negotiation).
     * - "local_app_started": a local Python app started mid-session and
     *   evicted us.
     * - "robot_busy_local": relay's safety-net for stale/concurrent
     *   sessions (should be rare).
     */
    _handleEndSession(msg) {
        const reason = msg.reason;
        const friendly = reason === 'robot_busy_local_app'
            ? 'Robot is busy: a local Python app is running'
            : reason === 'local_app_started'
                ? 'Disconnected: a local Python app started on the robot'
                : reason === 'robot_busy_local'
                    ? 'Robot is busy: another session is already active'
                    : null;

        // Case 1: a startSession() is still pending — reject its promise
        // with the same error shape as sessionRejected so app code can
        // treat both paths identically.
        if (this._sessionReject) {
            const err = new Error(
                friendly || `Session ended before it could start: ${reason || 'unknown reason'}`
            );
            err.reason = reason;
            this._emit('sessionRejected', { reason, activeApp: null });
            // Release resources allocated optimistically by startSession().
            if (this._pc) { this._pc.close(); this._pc = null; }
            if (this._micStream) { this._micStream.getTracks().forEach(t => t.stop()); this._micStream = null; }
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

        // Case 2: we were streaming and got kicked. Leave cleanup of _pc,
        // _dc, mic and timers to stopSession() so the event listener path
        // matches the user-initiated stop path exactly.
        if (this._state === 'streaming') {
            this._emit('sessionStopped', {
                reason: reason || 'remote_end',
                message: friendly,
            });
            // Fire-and-forget: we just react to what the server already
            // decided. stopSession() sends its own endSession back but
            // central has already dropped the session, so the echo is
            // harmless.
            this.stopSession().catch(() => { });
        }
    }

    async _handlePeerMessage(msg) {
        if (!this._pc) return;
        try {
            if (msg.sdp) {
                const sdp = msg.sdp;
                if (sdp.type === 'offer') {
                    const supportsMic = sdpHasAudioSendRecv(sdp.sdp);
                    this._micSupported = supportsMic;
                    this._emit('micSupported', { supported: supportsMic });

                    // Mic track must be added BEFORE setRemoteDescription so the
                    // generated answer naturally includes sendrecv for audio.
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
            }
            if (msg.ice) {
                await this._pc.addIceCandidate(new RTCIceCandidate(msg.ice));
            }
        } catch (e) {
            console.error('WebRTC error:', e);
            this._emit('error', { source: 'webrtc', error: e });
        }
    }

    /** Parse robot messages and dispatch. */
    _handleRobotMessage(data) {
        if ('version' in data && this._versionResolve) {
            this._versionResolve(data.version);
            this._versionResolve = null;
            return;
        }
        // Volume responses. Backend tags each response with `command` so we
        // know which pending resolver (speaker vs mic) to fulfil. If a
        // response arrives with no matching pending request (e.g. stale
        // after reconnect), just ignore it — the data channel protocol
        // has no multiplexing, so unmatched replies are expected.
        if (data.command === 'get_volume' || data.command === 'set_volume') {
            if (this._volumeResolve) {
                this._volumeResolve(data.status === 'error' ? null : data.volume);
                this._volumeResolve = null;
            }
            return;
        }
        if (data.command === 'get_microphone_volume' || data.command === 'set_microphone_volume') {
            if (this._micVolumeResolve) {
                this._micVolumeResolve(data.status === 'error' ? null : data.volume);
                this._micVolumeResolve = null;
            }
            return;
        }
        if (data.state) {
            const s = data.state;
            // Wire-shape pass-through. The daemon ships the head pose as a
            // nested 4×4 (numpy tolist()); we flatten to 16 numbers so
            // consumers can hand it straight to WebGL / Three.js / trajectory
            // logs. Everything else is forwarded as-is.
            if (s.head_pose) this._robotState.head = s.head_pose.flat();
            if (s.antennas) this._robotState.antennas = [s.antennas[0], s.antennas[1]];
            if (typeof s.body_yaw === 'number') this._robotState.body_yaw = s.body_yaw;
            if (s.motor_mode) this._robotState.motor_mode = s.motor_mode;
            if (typeof s.is_move_running === 'boolean') this._robotState.is_move_running = s.is_move_running;
            this._emit('state', { ...this._robotState });
        }
        if (data.error) {
            this._emit('error', { source: 'robot', error: data.error });
        }
    }

    /** Snap video playback to live edge if buffered lag exceeds 0.5 s. */
    _startLatencyMonitor(video) {
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

export default ReachyMini;
