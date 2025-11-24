/**
 * WebRTC client example for Reachy Mini
 *
 * This example shows how to connect to the Reachy Mini daemon
 * via WebRTC DataChannel from a JavaScript frontend.
 *
 * Usage:
 *   const client = new ReachyWebRTCClient('http://192.168.1.X:8000');
 *   await client.connect();
 *   const state = await client.getState();
 *   await client.goto({ target: [...], duration: 1.0 });
 */

class ReachyWebRTCClient {
    constructor(daemonUrl) {
        this.daemonUrl = daemonUrl.replace(/\/$/, '');
        this.pc = null;
        this.dataChannel = null;
        this.peerId = null;
        this.connected = false;
        this.pendingRequests = new Map();
        this.requestCounter = 0;
        this.onStateUpdate = null;
    }

    /**
     * Connect to the Reachy Mini daemon via WebRTC
     */
    async connect() {
        // Create peer connection
        this.pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });

        // Create data channel
        this.dataChannel = this.pc.createDataChannel('reachy-control', {
            ordered: true
        });

        this._setupDataChannel();
        this._setupPeerConnection();

        // Create and send offer
        const offer = await this.pc.createOffer();
        await this.pc.setLocalDescription(offer);

        // Wait for ICE gathering to complete
        await this._waitForIceGathering();

        // Send offer to signaling server
        const response = await fetch(`${this.daemonUrl}/api/webrtc/offer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sdp: this.pc.localDescription.sdp,
                type: this.pc.localDescription.type
            })
        });

        if (!response.ok) {
            throw new Error(`Signaling failed: ${response.statusText}`);
        }

        const answer = await response.json();
        this.peerId = answer.peer_id;

        // Set remote description
        await this.pc.setRemoteDescription({
            sdp: answer.sdp,
            type: answer.type
        });

        // Wait for data channel to open
        await this._waitForDataChannel();

        console.log(`Connected to Reachy Mini (peer: ${this.peerId})`);
        return this.peerId;
    }

    _setupDataChannel() {
        this.dataChannel.onopen = () => {
            console.log('Data channel opened');
            this.connected = true;
        };

        this.dataChannel.onclose = () => {
            console.log('Data channel closed');
            this.connected = false;
        };

        this.dataChannel.onmessage = (event) => {
            const message = JSON.parse(event.data);

            if (message.id && this.pendingRequests.has(message.id)) {
                // Response to a request
                const { resolve, reject } = this.pendingRequests.get(message.id);
                this.pendingRequests.delete(message.id);

                if (message.type === 'error') {
                    reject(new Error(message.error));
                } else {
                    resolve(message.data);
                }
            } else if (message.type === 'state' && this.onStateUpdate) {
                // Unsolicited state update
                this.onStateUpdate(message.data);
            }
        };

        this.dataChannel.onerror = (error) => {
            console.error('Data channel error:', error);
        };
    }

    _setupPeerConnection() {
        this.pc.oniceconnectionstatechange = () => {
            console.log('ICE connection state:', this.pc.iceConnectionState);
        };

        this.pc.onconnectionstatechange = () => {
            console.log('Connection state:', this.pc.connectionState);
            if (this.pc.connectionState === 'failed' ||
                this.pc.connectionState === 'closed') {
                this.connected = false;
            }
        };
    }

    _waitForIceGathering() {
        return new Promise((resolve) => {
            if (this.pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                this.pc.onicegatheringstatechange = () => {
                    if (this.pc.iceGatheringState === 'complete') {
                        resolve();
                    }
                };
            }
        });
    }

    _waitForDataChannel() {
        return new Promise((resolve, reject) => {
            if (this.dataChannel.readyState === 'open') {
                resolve();
            } else {
                const timeout = setTimeout(() => {
                    reject(new Error('Data channel open timeout'));
                }, 10000);

                this.dataChannel.onopen = () => {
                    clearTimeout(timeout);
                    this.connected = true;
                    resolve();
                };
            }
        });
    }

    /**
     * Send a command and wait for response
     */
    async send(action, params = {}) {
        if (!this.connected) {
            throw new Error('Not connected');
        }

        const id = ++this.requestCounter;
        const message = { action, params, id };

        return new Promise((resolve, reject) => {
            this.pendingRequests.set(id, { resolve, reject });

            // Timeout after 10 seconds
            setTimeout(() => {
                if (this.pendingRequests.has(id)) {
                    this.pendingRequests.delete(id);
                    reject(new Error(`Request timeout: ${action}`));
                }
            }, 10000);

            this.dataChannel.send(JSON.stringify(message));
        });
    }

    /**
     * Close the connection
     */
    async disconnect() {
        if (this.dataChannel) {
            this.dataChannel.close();
        }
        if (this.pc) {
            this.pc.close();
        }
        this.connected = false;
        console.log('Disconnected from Reachy Mini');
    }

    // ===== High-level API methods =====

    /**
     * Ping the robot
     */
    async ping() {
        return this.send('ping');
    }

    /**
     * Get full robot state
     */
    async getState() {
        return this.send('state/full');
    }

    /**
     * Get head pose
     */
    async getHeadPose() {
        return this.send('state/head_pose');
    }

    /**
     * Get body yaw angle
     */
    async getBodyYaw() {
        return this.send('state/body_yaw');
    }

    /**
     * Move to target position
     * @param {Object} options - Movement options
     * @param {Array} options.target - Target pose (4x4 matrix as flat array or nested)
     * @param {number} options.duration - Duration in seconds
     * @param {string} options.mode - Interpolation mode ('linear', 'minjerk', etc.)
     */
    async goto(options) {
        return this.send('move/goto', {
            target: options.target,
            duration: options.duration || 1.0,
            mode: options.mode || 'linear'
        });
    }

    /**
     * Set target directly (for continuous control)
     * @param {Array} target - Target pose
     */
    async setTarget(target) {
        return this.send('move/set_target', { target });
    }

    /**
     * Wake up the robot
     */
    async wakeUp() {
        return this.send('move/wake_up');
    }

    /**
     * Put robot to sleep
     */
    async gotoSleep() {
        return this.send('move/goto_sleep');
    }

    /**
     * Stop current movement
     * @param {string} taskId - Optional task ID to stop
     */
    async stopMove(taskId = null) {
        return this.send('move/stop', { task_id: taskId });
    }

    /**
     * Get motor status
     */
    async getMotorStatus() {
        return this.send('motors/status');
    }

    /**
     * Set motor mode
     * @param {string} mode - 'enabled', 'disabled', or 'gravity_compensation'
     */
    async setMotorMode(mode) {
        return this.send('motors/set_mode', { mode });
    }
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ReachyWebRTCClient };
}
if (typeof window !== 'undefined') {
    window.ReachyWebRTCClient = ReachyWebRTCClient;
}
