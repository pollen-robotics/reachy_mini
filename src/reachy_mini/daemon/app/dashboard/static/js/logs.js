const daemonLogs = {
    ws: null,
    isConnected: false,
    modalOpen: false,
    maxLines: 1000,

    openLogsModal: () => {
        const modal = document.getElementById('logs-modal');
        const logsDiv = document.getElementById('daemon-logs-content');

        if (!modal || !logsDiv) {
            console.error('Logs modal elements not found');
            return;
        }

        // Clear previous logs
        logsDiv.innerHTML = '';

        // Show modal
        modal.classList.remove('hidden');
        daemonLogs.modalOpen = true;

        // Connect to WebSocket
        daemonLogs.connectWebSocket();
    },

    closeLogsModal: () => {
        const modal = document.getElementById('logs-modal');

        if (!modal) {
            return;
        }

        // Hide modal
        modal.classList.add('hidden');
        daemonLogs.modalOpen = false;

        // Disconnect WebSocket
        daemonLogs.disconnect();
    },

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
            daemonLogs.appendLog(event.data);
        };

        daemonLogs.ws.onclose = () => {
            daemonLogs.isConnected = false;
            statusDiv.textContent = 'Disconnected';
            statusDiv.className = 'text-sm text-red-600 font-semibold';

            // Attempt reconnection if modal is still open
            if (daemonLogs.modalOpen) {
                setTimeout(() => {
                    if (daemonLogs.modalOpen && !daemonLogs.isConnected) {
                        console.log('Attempting to reconnect to logs WebSocket...');
                        daemonLogs.connectWebSocket();
                    }
                }, 2000);
            }
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

        // Create new log line element
        const logLine = document.createElement('div');
        logLine.className = 'font-mono text-xs text-gray-700 leading-relaxed';

        // Highlight error lines
        if (line.includes('ERROR') || line.includes('Error') || line.includes('error')) {
            logLine.className = 'font-mono text-xs text-red-700 font-semibold leading-relaxed';
        } else if (line.includes('WARNING') || line.includes('Warning') || line.includes('warning')) {
            logLine.className = 'font-mono text-xs text-orange-600 leading-relaxed';
        }

        logLine.textContent = line;
        logsDiv.appendChild(logLine);

        // Limit buffer size
        while (logsDiv.children.length > daemonLogs.maxLines) {
            logsDiv.removeChild(logsDiv.firstChild);
        }

        // Auto-scroll to bottom
        logsDiv.scrollTop = logsDiv.scrollHeight;
    },

    clearLogs: () => {
        const logsDiv = document.getElementById('daemon-logs-content');

        if (logsDiv) {
            logsDiv.innerHTML = '';
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

// Close modal when clicking outside of it
window.addEventListener('load', () => {
    const modal = document.getElementById('logs-modal');
    if (modal) {
        modal.addEventListener('click', (event) => {
            // Close if clicking on the overlay (not the modal content)
            if (event.target === modal) {
                daemonLogs.closeLogsModal();
            }
        });
    }
});
