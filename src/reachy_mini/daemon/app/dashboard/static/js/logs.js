const daemonLogs = {
    ws: null,
    isConnected: false,

    connectWebSocket: () => {
        const logsDiv = document.getElementById('daemon-logs-content');
        const statusDiv = document.getElementById('logs-status');

        if (!logsDiv || !statusDiv) {
            console.error('Logs elements not found');
            return;
        }

        // Create WebSocket connection
        daemonLogs.ws = new WebSocket(`ws://${location.host}/logs/ws/daemon`);

        daemonLogs.ws.onopen = () => {
            daemonLogs.isConnected = true;
            statusDiv.textContent = 'Connected';
            statusDiv.className = 'text-sm text-green-600 font-semibold';
        };

        daemonLogs.ws.onmessage = (event) => {
            // Ignore empty keepalive messages
            if (event.data && event.data.trim()) {
                daemonLogs.appendLog(event.data);
            }
        };

        daemonLogs.ws.onclose = () => {
            daemonLogs.isConnected = false;
            statusDiv.textContent = 'Disconnected';
            statusDiv.className = 'text-sm text-red-600 font-semibold';

            // Attempt reconnection after 2 seconds
            setTimeout(() => {
                if (!daemonLogs.isConnected) {
                    console.log('Attempting to reconnect to logs WebSocket...');
                    daemonLogs.connectWebSocket();
                }
            }, 2000);
        };

        daemonLogs.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            statusDiv.textContent = 'Connection Error';
            statusDiv.className = 'text-sm text-red-600 font-semibold';
        };
    },

    appendLog: (line) => {
        const logsDiv = document.getElementById('daemon-logs-content');

        if (!logsDiv) {
            return;
        }

        // Check if user is scrolled to bottom before adding new log
        const isScrolledToBottom = logsDiv.scrollHeight - logsDiv.scrollTop <= logsDiv.clientHeight + 50;

        // Create new log line element
        const logLine = document.createElement('div');
        logLine.className = 'text-gray-300 leading-tight';

        // Highlight error and warning lines
        if (line.includes('ERROR') || line.includes('Error') || line.includes('error')) {
            logLine.className = 'text-red-400 font-semibold leading-tight';
        } else if (line.includes('WARNING') || line.includes('Warning') || line.includes('warning')) {
            logLine.className = 'text-yellow-400 leading-tight';
        } else if (line.includes('INFO')) {
            logLine.className = 'text-green-400 leading-tight';
        }

        logLine.textContent = line;
        logsDiv.appendChild(logLine);

        // Only auto-scroll if user was already at the bottom
        if (isScrolledToBottom) {
            requestAnimationFrame(() => {
                logsDiv.scrollTop = logsDiv.scrollHeight;
            });
        }
    },

    clearLogs: () => {
        const logsDiv = document.getElementById('daemon-logs-content');

        if (logsDiv) {
            logsDiv.innerHTML = '';
        }
    },

    copyLogs: () => {
        const logsDiv = document.getElementById('daemon-logs-content');
        const buttonText = document.getElementById('copy-button-text');

        if (!logsDiv) {
            return;
        }

        // Get all log text
        const logText = logsDiv.innerText;

        // Create temporary textarea for copying (works without HTTPS)
        const textarea = document.createElement('textarea');
        textarea.value = logText;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);

        try {
            // Select and copy
            textarea.select();
            textarea.setSelectionRange(0, 99999); // For mobile devices
            const successful = document.execCommand('copy');

            if (successful && buttonText) {
                buttonText.textContent = 'Copied!';
                setTimeout(() => {
                    buttonText.textContent = 'Copy';
                }, 2000);
            } else if (buttonText) {
                buttonText.textContent = 'Failed';
                setTimeout(() => {
                    buttonText.textContent = 'Copy';
                }, 2000);
            }
        } catch (err) {
            console.error('Failed to copy logs:', err);
            if (buttonText) {
                buttonText.textContent = 'Failed';
                setTimeout(() => {
                    buttonText.textContent = 'Copy';
                }, 2000);
            }
        } finally {
            document.body.removeChild(textarea);
        }
    },

    disconnect: () => {
        if (daemonLogs.ws) {
            daemonLogs.ws.close();
            daemonLogs.ws = null;
            daemonLogs.isConnected = false;
        }
    }
};
