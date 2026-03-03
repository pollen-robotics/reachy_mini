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
 *   // 5. Send commands
 *   robot.setHeadPose(0, 10, -5);    // roll, pitch, yaw in degrees
 *   robot.setAntennas(30, -30);       // right, left in degrees
 *   robot.playSound("wake_up.wav");   // filename on robot
 *
 *   // 6. Receive live state (emitted every ~500 ms while streaming)
 *   robot.addEventListener("state", (e) => {
 *       const { head, antennas } = e.detail;
 *       // head:     { roll, pitch, yaw }   — degrees
 *       // antennas: { right, left }        — degrees
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
 *     signalingUrl:     string,   // default: "https://cduss-reachy-mini-central.hf.space"
 *     enableMicrophone: boolean,  // default: true — acquire mic for bidirectional audio
 *   })
 *
 *
 * READ-ONLY PROPERTIES
 * ────────────────────
 *   .state            "disconnected" | "connected" | "streaming"
 *   .robots           Array<{ id: string, meta: { name: string } }>
 *   .robotState       { head: { roll, pitch, yaw }, antennas: { right, left } }  (degrees)
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
 *   "state"           { head: { roll, pitch, yaw }, antennas: { right, left } }
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
    oauthLoginUrl,
    oauthHandleRedirectIfPresent,
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
        [-sp,     cp * sr,                cp * cr,                0],
        [0,       0,                      0,                      1],
    ];
}

/**
 * Rotation matrix (3×3 or 4×4) → { roll, pitch, yaw } in degrees.
 * @param {number[][]} m  @returns {{ roll: number, pitch: number, yaw: number }}
 */
export function matrixToRpy(m) {
    return {
        roll:  radToDeg(Math.atan2(m[2][1], m[2][2])),
        pitch: radToDeg(Math.asin(-m[2][0])),
        yaw:   radToDeg(Math.atan2(m[1][0], m[0][0])),
    };
}

// ─── Internal helpers ────────────────────────────────────────────────────────

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

    /** @param {{ signalingUrl?: string, enableMicrophone?: boolean }} [options] */
    constructor(options = {}) {
        super();
        this._signalingUrl = options.signalingUrl || 'https://cduss-reachy-mini-central.hf.space';
        this._enableMicrophone = options.enableMicrophone !== false;

        this._state = 'disconnected';                 // 'disconnected' | 'connected' | 'streaming'
        this._robots = [];                             // latest robot list from signaling
        this._robotState = {                           // updated every ~500 ms while streaming
            head: { roll: 0, pitch: 0, yaw: 0 },
            antennas: { right: 0, left: 0 },
        };

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

    /** @returns {{head: {roll:number,pitch:number,yaw:number}, antennas: {right:number,left:number}}} */
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
        window.location.href = await oauthLoginUrl();
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
            res = await fetch(
                `${this._signalingUrl}/events?token=${encodeURIComponent(this._token)}`,
                { signal: this._sseAbortController.signal },
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
                                        meta: { name: 'Telepresence' },
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
                if (r?.sessionId) this._sessionId = r.sessionId;
            });
        });
    }

    /**
     * End the WebRTC session.  Returns to "connected" state so you can
     * startSession() again with the same or a different robot.
     * @returns {Promise<void>}
     */
    async stopSession() {
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
     * Set the head orientation.
     * @param {number} roll  — degrees  @param {number} pitch — degrees  @param {number} yaw — degrees
     * @returns {boolean}
     */
    setHeadPose(roll, pitch, yaw) {
        return this._sendCommand({ type: "set_target", head: rpyToMatrix(roll, pitch, yaw).flat() });
    }

    /**
     * Set antenna positions.
     * @param {number} rightDeg  @param {number} leftDeg
     * @returns {boolean}
     */
    setAntennas(rightDeg, leftDeg) {
        return this._sendCommand({ type: "set_antennas", antennas: [degToRad(rightDeg), degToRad(leftDeg)] });
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
     * Send an arbitrary JSON command over the data channel.
     * @param {object} data  @returns {boolean}
     */
    sendRaw(data) {
        return this._sendCommand(data);
    }

    /**
     * Request a state snapshot.  The response arrives as a "state" event.
     * Called automatically every 500 ms while streaming.
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
        try {
            const res = await fetch(`${this._signalingUrl}/send?token=${encodeURIComponent(this._token)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
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
            case 'peer':
                this._handlePeerMessage(msg);
                break;
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

    /** Parse robot state (rotation matrix + radians) into degrees and emit. */
    _handleRobotMessage(data) {
        if (data.state) {
            const s = data.state;
            if (s.head_pose) this._robotState.head = matrixToRpy(s.head_pose);
            if (s.antennas) {
                this._robotState.antennas = {
                    right: radToDeg(s.antennas[0]),
                    left:  radToDeg(s.antennas[1]),
                };
            }
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
