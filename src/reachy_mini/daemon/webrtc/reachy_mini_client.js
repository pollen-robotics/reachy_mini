/**
 * Reachy Mini WebRTC Client Library
 *
 * A complete client library for controlling Reachy Mini robot via WebRTC.
 * Designed to work from HTTPS frontends connecting to HTTP robot daemon.
 *
 * @version 0.3.0
 * @license MIT
 *
 * @example
 * // Basic usage
 * const reachy = new ReachyMini('http://192.168.1.100:8000');
 * await reachy.connect();
 *
 * // Get robot state
 * const state = await reachy.state.getFull();
 * console.log('Head pose:', state.head_pose);
 *
 * // Control the robot
 * await reachy.motors.enable();
 * await reachy.move.wakeUp();
 * await reachy.move.goto({ head_pose: {x:0, y:0, z:0, roll:0, pitch:0.2, yaw:0}, duration: 1.5 });
 *
 * // Disconnect when done
 * await reachy.disconnect();
 *
 * // Check version
 * console.log(ReachyMini.VERSION);  // "0.2.0"
 */

(function(root, factory) {
    if (typeof define === 'function' && define.amd) {
        define([], factory);
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.ReachyMini = factory();
    }
}(typeof self !== 'undefined' ? self : this, function() {
    'use strict';

    /**
     * Library version - UPDATE THIS ON EVERY CHANGE
     */
    const VERSION = '0.3.0';

    /**
     * Connection states
     */
    const ConnectionState = {
        DISCONNECTED: 'disconnected',
        CONNECTING: 'connecting',
        CONNECTED: 'connected',
        FAILED: 'failed'
    };

    /**
     * Motor modes
     */
    const MotorMode = {
        ENABLED: 'enabled',
        DISABLED: 'disabled',
        GRAVITY_COMPENSATION: 'gravity_compensation'
    };

    /**
     * Interpolation modes for movements
     */
    const InterpolationMode = {
        LINEAR: 'linear',
        MINJERK: 'minjerk',
        EASE: 'ease',
        CARTOON: 'cartoon'
    };

    /**
     * Main Reachy Mini client class
     */
    class ReachyMini {
        /**
         * Create a new Reachy Mini client
         * @param {string} daemonUrl - URL of the Reachy Mini daemon (e.g., 'http://192.168.1.100:8000')
         * @param {Object} options - Configuration options
         * @param {Object} options.iceServers - Custom ICE servers for WebRTC
         * @param {number} options.timeout - Request timeout in milliseconds (default: 10000)
         * @param {boolean} options.debug - Enable debug logging (default: false)
         */
        constructor(daemonUrl, options = {}) {
            this._daemonUrl = daemonUrl.replace(/\/$/, '');
            this._options = {
                iceServers: options.iceServers || [
                    { urls: 'stun:stun.l.google.com:19302' }
                ],
                timeout: options.timeout || 10000,
                debug: options.debug || false
            };

            this._pc = null;
            this._dataChannel = null;
            this._peerId = null;
            this._state = ConnectionState.DISCONNECTED;
            this._pendingRequests = new Map();
            this._requestCounter = 0;

            // Event handlers
            this._eventHandlers = {
                stateUpdate: [],
                connectionChange: [],
                error: []
            };

            // Initialize sub-APIs
            this.state = new StateAPI(this);
            this.move = new MoveAPI(this);
            this.motors = new MotorsAPI(this);
            this.joints = new JointsAPI(this);
        }

        // ===== Connection Management =====

        /**
         * Get current connection state
         * @returns {string} Current connection state
         */
        get connectionState() {
            return this._state;
        }

        /**
         * Get peer ID
         * @returns {string|null} Peer ID if connected
         */
        get peerId() {
            return this._peerId;
        }

        /**
         * Check if connected
         * @returns {boolean} True if connected
         */
        get isConnected() {
            return this._state === ConnectionState.CONNECTED;
        }

        /**
         * Connect to the Reachy Mini daemon
         * @returns {Promise<string>} Peer ID on successful connection
         * @throws {Error} If connection fails
         */
        async connect() {
            if (this._state === ConnectionState.CONNECTED) {
                return this._peerId;
            }

            this._setState(ConnectionState.CONNECTING);

            try {
                // Create peer connection
                this._pc = new RTCPeerConnection({
                    iceServers: this._options.iceServers
                });

                // Create data channel
                this._dataChannel = this._pc.createDataChannel('reachy-control', {
                    ordered: true
                });

                this._setupDataChannel();
                this._setupPeerConnection();

                // Create and set local description
                const offer = await this._pc.createOffer();
                await this._pc.setLocalDescription(offer);

                // Wait for ICE gathering
                await this._waitForIceGathering();

                // Send offer to signaling server
                const response = await this._fetch('/api/webrtc/offer', {
                    method: 'POST',
                    body: JSON.stringify({
                        sdp: this._pc.localDescription.sdp,
                        type: this._pc.localDescription.type
                    })
                });

                this._peerId = response.peer_id;

                // Set remote description
                await this._pc.setRemoteDescription({
                    sdp: response.sdp,
                    type: response.type
                });

                // Wait for data channel to open
                await this._waitForDataChannel();

                this._setState(ConnectionState.CONNECTED);
                this._log(`Connected to Reachy Mini (peer: ${this._peerId})`);

                return this._peerId;

            } catch (error) {
                this._setState(ConnectionState.FAILED);
                this._emit('error', error);
                throw error;
            }
        }

        /**
         * Disconnect from the Reachy Mini daemon
         */
        async disconnect() {
            if (this._dataChannel) {
                this._dataChannel.close();
                this._dataChannel = null;
            }

            if (this._pc) {
                this._pc.close();
                this._pc = null;
            }

            this._peerId = null;
            this._pendingRequests.clear();
            this._setState(ConnectionState.DISCONNECTED);
            this._log('Disconnected from Reachy Mini');
        }

        // ===== Event Handling =====

        /**
         * Register an event handler
         * @param {string} event - Event name ('stateUpdate', 'connectionChange', 'error')
         * @param {Function} handler - Event handler function
         */
        on(event, handler) {
            if (this._eventHandlers[event]) {
                this._eventHandlers[event].push(handler);
            }
        }

        /**
         * Remove an event handler
         * @param {string} event - Event name
         * @param {Function} handler - Handler to remove
         */
        off(event, handler) {
            if (this._eventHandlers[event]) {
                const index = this._eventHandlers[event].indexOf(handler);
                if (index > -1) {
                    this._eventHandlers[event].splice(index, 1);
                }
            }
        }

        // ===== Core Communication =====

        /**
         * Send a command and wait for response
         * @param {string} action - Action name (e.g., 'state/full')
         * @param {Object} params - Action parameters
         * @returns {Promise<any>} Response data
         */
        async send(action, params = {}) {
            if (!this.isConnected) {
                throw new Error('Not connected to Reachy Mini');
            }

            const id = ++this._requestCounter;
            const message = { action, params, id };

            return new Promise((resolve, reject) => {
                const timeoutId = setTimeout(() => {
                    this._pendingRequests.delete(id);
                    reject(new Error(`Request timeout: ${action}`));
                }, this._options.timeout);

                this._pendingRequests.set(id, {
                    resolve: (data) => {
                        clearTimeout(timeoutId);
                        resolve(data);
                    },
                    reject: (error) => {
                        clearTimeout(timeoutId);
                        reject(error);
                    }
                });

                this._dataChannel.send(JSON.stringify(message));
                this._log(`Sent: ${action}`, params);
            });
        }

        /**
         * Ping the robot to test connection
         * @returns {Promise<Object>} Ping response
         */
        async ping() {
            return this.send('ping');
        }

        // ===== Private Methods =====

        _setupDataChannel() {
            this._dataChannel.onopen = () => {
                this._log('Data channel opened');
            };

            this._dataChannel.onclose = () => {
                this._log('Data channel closed');
                if (this._state === ConnectionState.CONNECTED) {
                    this._setState(ConnectionState.DISCONNECTED);
                }
            };

            this._dataChannel.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this._handleMessage(message);
                } catch (error) {
                    this._log('Failed to parse message:', error);
                }
            };

            this._dataChannel.onerror = (error) => {
                this._log('Data channel error:', error);
                this._emit('error', error);
            };
        }

        _setupPeerConnection() {
            this._pc.oniceconnectionstatechange = () => {
                this._log('ICE state:', this._pc.iceConnectionState);
            };

            this._pc.onconnectionstatechange = () => {
                this._log('Connection state:', this._pc.connectionState);
                if (this._pc.connectionState === 'failed') {
                    this._setState(ConnectionState.FAILED);
                } else if (this._pc.connectionState === 'closed') {
                    this._setState(ConnectionState.DISCONNECTED);
                }
            };
        }

        _handleMessage(message) {
            if (message.id && this._pendingRequests.has(message.id)) {
                const { resolve, reject } = this._pendingRequests.get(message.id);
                this._pendingRequests.delete(message.id);

                if (message.type === 'error') {
                    reject(new Error(message.error));
                } else {
                    this._log(`Received: ${message.action}`, message.data);
                    resolve(message.data);
                }
            } else if (message.type === 'state') {
                this._emit('stateUpdate', message.data);
            }
        }

        _waitForIceGathering() {
            return new Promise((resolve) => {
                if (this._pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    const checkState = () => {
                        if (this._pc.iceGatheringState === 'complete') {
                            this._pc.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    };
                    this._pc.addEventListener('icegatheringstatechange', checkState);
                }
            });
        }

        _waitForDataChannel() {
            return new Promise((resolve, reject) => {
                if (this._dataChannel.readyState === 'open') {
                    resolve();
                    return;
                }

                const timeout = setTimeout(() => {
                    reject(new Error('Data channel connection timeout'));
                }, this._options.timeout);

                const onOpen = () => {
                    clearTimeout(timeout);
                    this._dataChannel.removeEventListener('open', onOpen);
                    resolve();
                };

                this._dataChannel.addEventListener('open', onOpen);
            });
        }

        async _fetch(path, options = {}) {
            const url = `${this._daemonUrl}${path}`;
            const response = await fetch(url, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });

            if (!response.ok) {
                const error = await response.text();
                throw new Error(`HTTP ${response.status}: ${error}`);
            }

            return response.json();
        }

        _setState(state) {
            if (this._state !== state) {
                this._state = state;
                this._emit('connectionChange', state);
            }
        }

        _emit(event, data) {
            if (this._eventHandlers[event]) {
                for (const handler of this._eventHandlers[event]) {
                    try {
                        handler(data);
                    } catch (error) {
                        console.error(`Error in ${event} handler:`, error);
                    }
                }
            }
        }

        _log(...args) {
            if (this._options.debug) {
                console.log('[ReachyMini]', ...args);
            }
        }
    }

    /**
     * State API - Query robot state
     */
    class StateAPI {
        constructor(client) {
            this._client = client;
        }

        /**
         * Get full robot state
         * @returns {Promise<Object>} Full state with head_pose, body_yaw, antenna_positions
         */
        async getFull() {
            return this._client.send('state/full');
        }

        /**
         * Get head pose as 4x4 transformation matrix
         * @returns {Promise<Array>} 4x4 pose matrix
         */
        async getHeadPose() {
            return this._client.send('state/head_pose');
        }

        /**
         * Get body yaw angle in radians
         * @returns {Promise<number>} Body yaw angle
         */
        async getBodyYaw() {
            return this._client.send('state/body_yaw');
        }

        /**
         * Get antenna joint positions
         * @returns {Promise<Array>} Antenna positions [left, right] in radians
         */
        async getAntennaPositions() {
            return this._client.send('state/antenna_positions');
        }
    }

    /**
     * Move API - Control robot movements
     */
    class MoveAPI {
        constructor(client) {
            this._client = client;
        }

        /**
         * Move to target pose
         * @param {Object} options - Movement options
         * @param {Object|Array} [options.head_pose] - Head pose (4x4 matrix array or {x,y,z,roll,pitch,yaw})
         * @param {Array} [options.antennas] - Antenna positions [left, right] in radians
         * @param {number} [options.body_yaw] - Body yaw angle in radians
         * @param {number} [options.duration=1.0] - Duration in seconds
         * @returns {Promise<Object>} Status
         */
        async goto(options) {
            return this._client.send('move/goto', {
                head_pose: options.head_pose,
                antennas: options.antennas,
                body_yaw: options.body_yaw,
                duration: options.duration || 1.0
            });
        }

        /**
         * Set target directly for continuous control
         * @param {Array} target - Target pose
         * @returns {Promise<Object>} Status
         */
        async setTarget(target) {
            return this._client.send('move/set_target', { target });
        }

        /**
         * Wake up the robot
         * @returns {Promise<Object>} Task info
         */
        async wakeUp() {
            return this._client.send('move/wake_up');
        }

        /**
         * Put robot to sleep
         * @returns {Promise<Object>} Task info
         */
        async gotoSleep() {
            return this._client.send('move/goto_sleep');
        }

        /**
         * Stop current movement
         * @param {string} [taskId] - Specific task ID to stop
         * @returns {Promise<Object>} Status
         */
        async stop(taskId = null) {
            return this._client.send('move/stop', { task_id: taskId });
        }
    }

    /**
     * Joints API - Direct joint control
     */
    class JointsAPI {
        constructor(client) {
            this._client = client;
        }

        /**
         * Get current joint positions
         * @returns {Promise<Object>} Joint positions with head_joints and antenna_joints arrays
         */
        async get() {
            return this._client.send('joints/get');
        }

        /**
         * Set joint target positions directly
         * @param {Object} options - Joint targets
         * @param {Array} [options.head_joints] - Head joint positions (7 values) in radians
         * @param {Array} [options.antenna_joints] - Antenna joint positions (2 values) in radians
         * @returns {Promise<Object>} Status
         */
        async setTarget(options) {
            return this._client.send('joints/set_target', {
                head_joints: options.head_joints,
                antenna_joints: options.antenna_joints
            });
        }
    }

    /**
     * Motors API - Control motor modes
     */
    class MotorsAPI {
        constructor(client) {
            this._client = client;
        }

        /**
         * Get current motor status
         * @returns {Promise<Object>} Motor status with mode
         */
        async getStatus() {
            return this._client.send('motors/status');
        }

        /**
         * Set motor mode
         * @param {string} mode - Motor mode ('enabled', 'disabled', 'gravity_compensation')
         * @returns {Promise<Object>} Status
         */
        async setMode(mode) {
            return this._client.send('motors/set_mode', { mode });
        }

        /**
         * Enable motors (torque on)
         * @returns {Promise<Object>} Status
         */
        async enable() {
            return this.setMode(MotorMode.ENABLED);
        }

        /**
         * Disable motors (torque off)
         * @returns {Promise<Object>} Status
         */
        async disable() {
            return this.setMode(MotorMode.DISABLED);
        }

        /**
         * Enable gravity compensation mode
         * @returns {Promise<Object>} Status
         */
        async gravityCompensation() {
            return this.setMode(MotorMode.GRAVITY_COMPENSATION);
        }
    }

    // Export constants alongside the main class
    ReachyMini.VERSION = VERSION;
    ReachyMini.ConnectionState = ConnectionState;
    ReachyMini.MotorMode = MotorMode;
    ReachyMini.InterpolationMode = InterpolationMode;

    return ReachyMini;
}));
