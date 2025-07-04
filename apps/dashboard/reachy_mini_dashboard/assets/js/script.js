
let ws = null;
let activeOperations = new Set();
let currentLogProcessId = null;
let autoScroll = true;
let logBuffer = new Map();
let simulationEnabled = false;
let daemonStatus = null;
let spacesStore = null;

// Initialize spaces store on page load
function initializeSpacesStore() {
    spacesStore = new SpacesStore();
    spacesStore.init();
}

// Simulation functions
async function toggleSimulation() {
    const toggle = document.getElementById('simulation-toggle');
    const originalState = toggle.checked;

    try {
        toggle.disabled = true;
        const response = await fetch('/api/simulation/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const result = await response.json();
        if (response.ok) {
            simulationEnabled = result.simulation_enabled;
            updateSimulationUI();

            const indicator = document.getElementById('simulation-mode-indicator');
            const originalText = indicator.textContent;
            indicator.textContent = `🎮 SIMULATION ${simulationEnabled ? 'ENABLED' : 'DISABLED'}`;

            setTimeout(() => {
                if (simulationEnabled) {
                    indicator.textContent = originalText;
                }
            }, 2000);

            console.log(`Simulation mode ${simulationEnabled ? 'enabled' : 'disabled'}`);
        } else {
            throw new Error(result.detail || 'Failed to toggle simulation');
        }
    } catch (error) {
        console.error('Failed to toggle simulation:', error);
        toggle.checked = !originalState;
        simulationEnabled = !simulationEnabled;
        updateSimulationUI();
        alert(`Failed to toggle simulation: ${error.message}`);
    } finally {
        toggle.disabled = false;
    }
}

async function fetchSimulationStatus() {
    try {
        const response = await fetch('/api/simulation/status');
        const data = await response.json();
        simulationEnabled = data.simulation_enabled;
        updateSimulationUI();
    } catch (error) {
        console.error('Failed to fetch simulation status:', error);
    }
}

function updateSimulationUI() {
    const toggle = document.getElementById('simulation-toggle');
    const indicator = document.getElementById('simulation-mode-indicator');
    toggle.checked = simulationEnabled;
    indicator.style.display = simulationEnabled ? 'block' : 'none';
}

// Daemon functions
async function fetchDaemonStatus() {
    try {
        const response = await fetch('/daemon_status');
        const data = await response.json();
        daemonStatus = data;
        updateDaemonStatusUI(data);
    } catch (error) {
        console.error('Failed to fetch daemon status:', error);
        updateDaemonStatusUI({
            status: 'error',
            message: 'Failed to connect to daemon',
            running: false
        });
    }
}

function updateDaemonStatusUI(status) {
    const indicator = document.getElementById('daemon-status-indicator');
    const statusText = document.getElementById('daemon-status-text');
    const robotInfo = document.getElementById('robot-info');
    const robotDetails = document.getElementById('robot-details');
    const startBtn = document.getElementById('daemon-start-btn');
    const stopBtn = document.getElementById('daemon-stop-btn');
    const restartBtn = document.getElementById('daemon-restart-btn');

    indicator.className = 'status-indicator';

    if (status.running === true) {
        indicator.classList.add('status-running');
        statusText.textContent = status.message || 'Daemon is running';
        startBtn.disabled = true;
        stopBtn.disabled = false;
        restartBtn.disabled = false;
    } else if (status.running === false) {
        indicator.classList.add('status-stopped');
        statusText.textContent = status.message || 'Daemon is stopped';
        startBtn.disabled = false;
        stopBtn.disabled = true;
        restartBtn.disabled = true;
    } else {
        indicator.classList.add('status-unknown');
        statusText.textContent = status.message || 'Status unknown';
        startBtn.disabled = false;
        stopBtn.disabled = false;
        restartBtn.disabled = false;
    }

    if (status.robot_info) {
        robotDetails.innerHTML = `
          <div>Model: ${status.robot_info.model || 'Unknown'}</div>
          <div>Version: ${status.robot_info.version || 'Unknown'}</div>
          ${status.robot_info.uptime ? `<div>Uptime: ${status.robot_info.uptime}</div>` : ''}
        `;
        robotInfo.style.display = 'block';
    } else {
        robotInfo.style.display = 'none';
    }
}

async function controlDaemon(action) {
    const button = document.getElementById(`daemon-${action}-btn`);
    const originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '⏳ Working...';

    try {
        const response = await fetch(`/daemon_${action}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const result = await response.json();
        if (response.ok) {
            setTimeout(() => fetchDaemonStatus(), 1000);
            button.innerHTML = '✅ Done';
            setTimeout(() => button.innerHTML = originalText, 2000);
        } else {
            throw new Error(result.detail || `Failed to ${action} daemon`);
        }
    } catch (error) {
        console.error(`Failed to ${action} daemon:`, error);
        alert(`Failed to ${action} daemon: ${error.message}`);
        button.innerHTML = '❌ Error';
        setTimeout(() => button.innerHTML = originalText, 2000);
    } finally {
        setTimeout(() => {
            button.disabled = false;
            if (button.innerHTML === '⏳ Working...' || button.innerHTML === '✅ Done' || button.innerHTML === '❌ Error') {
                button.innerHTML = originalText;
            }
        }, 2000);
    }
}

// WebSocket and logging functions
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    ws = new WebSocket(wsUrl);

    ws.onopen = function () {
        console.log('WebSocket connected');
        if (currentLogProcessId) {
            ws.send(JSON.stringify({
                type: 'request_logs',
                process_id: currentLogProcessId
            }));
        }
    };

    ws.onmessage = function (event) {
        const data = JSON.parse(event.data);
        if (data.type === 'installation_update') {
            updateInstallationProgress(data.installation_id, data.status);
        } else if (data.type === 'log_update') {
            handleLogUpdate(data.process_id, data.log_entry);
        }
    };

    ws.onclose = function () {
        console.log('WebSocket disconnected, retrying in 3 seconds...');
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = function (error) {
        console.error('WebSocket error:', error);
    };
}

function handleLogUpdate(processId, logEntry) {
    if (!logBuffer.has(processId)) {
        logBuffer.set(processId, []);
    }
    logBuffer.get(processId).push(logEntry);

    const logs = logBuffer.get(processId);
    if (logs.length > 500) {
        logBuffer.set(processId, logs.slice(-500));
    }

    if (currentLogProcessId === processId) {
        appendLogEntry(logEntry);
    }
}

function showLogViewer(processId, description = '') {
    currentLogProcessId = processId;
    const logSection = document.getElementById('log-viewer-section');
    const processName = document.getElementById('log-process-name');
    const logContent = document.getElementById('log-content');

    processName.textContent = description || processId;
    logContent.innerHTML = '';

    const logs = logBuffer.get(processId) || [];
    logs.forEach(logEntry => {
        appendLogEntry(logEntry, false);
    });

    logSection.style.display = 'block';
    setTimeout(() => scrollToBottom(), 100);

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'request_logs',
            process_id: processId
        }));
    }
}

function hideLogViewer() {
    currentLogProcessId = null;
    document.getElementById('log-viewer-section').style.display = 'none';
}

function appendLogEntry(logEntry, shouldAutoScroll = true) {
    const logContent = document.getElementById('log-content');
    const entry = document.createElement('div');
    entry.className = `log-entry ${logEntry.level} ${logEntry.stream}`;

    const timestamp = new Date(logEntry.timestamp).toLocaleTimeString();
    entry.innerHTML = `
        <span class="log-timestamp">[${timestamp}]</span>
        <span class="log-message">${escapeHtml(logEntry.message)}</span>
      `;

    logContent.appendChild(entry);
    if (autoScroll && shouldAutoScroll) {
        scrollToBottom();
    }
}

function scrollToBottom() {
    const logViewer = document.getElementById('log-viewer');
    logViewer.scrollTop = logViewer.scrollHeight;
}

function clearCurrentLogs() {
    if (currentLogProcessId) {
        logBuffer.delete(currentLogProcessId);
        document.getElementById('log-content').innerHTML = '';
        fetch(`/api/logs/${currentLogProcessId}`, { method: 'DELETE' })
            .catch(err => console.error('Failed to clear logs on server:', err));
    }
}

function toggleAutoScroll() {
    autoScroll = !autoScroll;
    const button = document.getElementById('auto-scroll-toggle');
    button.textContent = `Auto-scroll: ${autoScroll ? 'ON' : 'OFF'}`;
    button.className = autoScroll
        ? 'text-sm px-3 py-1 rounded bg-blue-500 text-white hover:bg-blue-600 transition-colors'
        : 'text-sm px-3 py-1 rounded bg-gray-500 text-white hover:bg-gray-600 transition-colors';
}

function updateInstallationProgress(installationId, status) {
    const container = document.getElementById('installation-list');
    const activeSection = document.getElementById('active-installations');
    let installDiv = document.getElementById(`install-${installationId}`);

    if (status.stage === 'complete' || status.stage === 'error') {
        activeOperations.delete(status.app_name);

        if (installDiv) {
            const progressColor = status.stage === 'error' ? 'bg-red-500' : 'bg-green-500';
            const statusIcon = status.stage === 'error' ? '❌' : '✅';

            installDiv.innerHTML = `
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-lg">${statusIcon}</span>
                <span class="font-semibold">${status.message}</span>
              </div>
            </div>
          `;

            setTimeout(() => {
                installDiv.remove();
                if (container.children.length === 0) {
                    activeSection.style.display = 'none';
                }
                if (status.operation === 'remove' && status.stage === 'complete') {
                    console.log('Removal completed, refreshing status...');
                    setTimeout(fetchStatus, 500);
                }
            }, 3000);
        }

        fetchStatus();
        return;
    }

    activeOperations.add(status.app_name);

    if (!installDiv) {
        activeSection.style.display = 'block';
        installDiv = document.createElement('div');
        installDiv.id = `install-${installationId}`;
        installDiv.className = 'bg-white rounded-lg shadow p-4';
        container.appendChild(installDiv);
    }

    const progressColor = status.stage === 'error' ? 'bg-red-500' :
        status.operation === 'update' ? 'bg-green-500' :
            status.operation === 'remove' ? 'bg-orange-500' : 'bg-blue-500';

    const statusIcon = getStatusIcon(status.stage, status.operation);
    const operationText = status.operation === 'update' ? 'Updating' :
        status.operation === 'remove' ? 'Removing' : 'Installing';

    installDiv.innerHTML = `
        <div class="flex items-center justify-between mb-2">
          <div class="flex items-center space-x-2">
            <span class="text-lg">${statusIcon}</span>
            <span class="font-semibold">${operationText} ${status.app_name}</span>
          </div>
          <div class="flex items-center space-x-2">
            <button onclick="showLogViewer('${status.operation}_${installationId}', '${operationText} ${status.app_name}')" 
                    class="text-sm px-2 py-1 rounded bg-gray-500 text-white hover:bg-gray-600 transition-colors">
              📋 Logs
            </button>
            <span class="text-sm text-gray-500">${status.progress}%</span>
          </div>
        </div>
        <div class="mb-2">
          <div class="bg-gray-200 rounded-full h-2">
            <div class="installation-progress ${progressColor} h-2 rounded-full" style="width: ${status.progress}%"></div>
          </div>
        </div>
        <p class="text-sm text-gray-600">${status.message}</p>
      `;
}

function getStatusIcon(stage, operation = 'install') {
    const icons = {
        'starting': '🚀', 'creating_venv': '🐍', 'installing': '📦', 'dependencies': '🔧',
        'finalizing': '🎯', 'complete': '✅', 'error': '❌', 'backup': '💾',
        'updating_pip': '⬆️', 'updating_app': '🔄', 'stopping': '⏹️',
        'clearing_venv': '🧹', 'removing_files': '🗑️'
    };
    return icons[stage] || (operation === 'remove' ? '🗑️' : '⏳');
}

async function fetchStatus() {
    try {
        if (Date.now() - (window.lastFetchTime || 0) < 1000) {
            console.log('Skipping fetchStatus due to rate limit');
            return;
        }
        window.lastFetchTime = Date.now();

        const res = await fetch('/api/status/full');
        const data = await res.json();
        console.log('Fetched full status:', data);

        renderApps(data.available_apps, data.current, data.venv_apps || [], data.venv_apps_detailed || [], data.app_process_id);
        renderInstallationHistory(data.installation_history || []);

        if (data.simulation_enabled !== undefined) {
            simulationEnabled = data.simulation_enabled;
            updateSimulationUI();
        }

        const activeInstalls = data.active_installations || {};
        const container = document.getElementById('installation-list');
        const activeSection = document.getElementById('active-installations');

        if (Object.keys(activeInstalls).length > 0) {
            activeSection.style.display = 'block';
            for (const [id, status] of Object.entries(activeInstalls)) {
                updateInstallationProgress(id, status);
            }
        } else {
            activeSection.style.display = 'none';
            container.innerHTML = '';
        }

    } catch (error) {
        console.error('Failed to fetch status:', error);
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            renderApps(data.available_apps, data.current, [], [], data.app_process_id);

            if (data.simulation_enabled !== undefined) {
                simulationEnabled = data.simulation_enabled;
                updateSimulationUI();
            }
        } catch (fallbackError) {
            console.error('Failed to fetch basic status:', fallbackError);
        }
    }
}

function renderApps(apps, current, venvApps, venvAppsDetailed = [], appProcessId = null) {
    const container = document.getElementById('app-list');
    container.innerHTML = '';

    if (apps.length === 0) {
        container.innerHTML = '<p class="text-gray-600">No apps found.</p>';
        return;
    }

    const venvDetailsMap = {};
    venvAppsDetailed.forEach(app => {
        venvDetailsMap[app.name] = app;
    });

    apps.forEach(name => {
        const isRunning = name === current;
        const isVenvApp = venvApps.includes(name);
        const venvDetails = venvDetailsMap[name];
        const isBeingRemoved = activeOperations.has(name);

        const item = document.createElement('div');
        item.className = `bg-white rounded-lg shadow p-4 flex items-center justify-between ${isBeingRemoved ? 'removing' : ''}`;
        item.id = `app-item-${name}`;

        const leftSection = document.createElement('div');
        leftSection.className = 'flex items-center space-x-3';

        const labelSection = document.createElement('div');
        labelSection.className = 'flex flex-col';

        const label = document.createElement('span');
        label.className = 'text-lg font-medium';
        label.textContent = name;
        labelSection.appendChild(label);

        if (isVenvApp || venvDetails) {
            const badge = document.createElement('span');
            badge.className = 'bg-purple-100 text-purple-800 text-xs font-semibold px-2 py-1 rounded mt-1 w-fit';
            badge.textContent = 'VENV';
            labelSection.appendChild(badge);

            if (venvDetails && venvDetails.last_updated) {
                const updateInfo = document.createElement('span');
                updateInfo.className = 'text-xs text-gray-500 mt-1';
                updateInfo.textContent = `Updated: ${new Date(venvDetails.last_updated).toLocaleDateString()}`;
                labelSection.appendChild(updateInfo);
            }
        }

        leftSection.appendChild(labelSection);

        const rightSection = document.createElement('div');
        rightSection.className = 'flex items-center space-x-3';

        if (!isBeingRemoved) {
            if (isRunning) {
                const status = document.createElement('span');
                status.className = 'text-green-600 font-semibold mr-4';
                status.textContent = 'Running';
                rightSection.appendChild(status);

                if (appProcessId) {
                    const logsBtn = document.createElement('button');
                    logsBtn.className = 'bg-gray-500 hover:bg-gray-600 text-white px-3 py-2 rounded text-sm transition-colors';
                    logsBtn.textContent = '📋 Logs';
                    logsBtn.onclick = () => showLogViewer(appProcessId, `Running ${name}`);
                    rightSection.appendChild(logsBtn);
                }

                const stopBtn = document.createElement('button');
                stopBtn.className = 'bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded transition-colors';
                stopBtn.textContent = 'Stop';
                stopBtn.onclick = () => stopApp();
                rightSection.appendChild(stopBtn);
            } else {
                const startBtn = document.createElement('button');
                startBtn.className = 'bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded transition-colors';
                startBtn.textContent = 'Start';
                startBtn.onclick = () => startApp(name);
                rightSection.appendChild(startBtn);
            }

            if (isVenvApp || venvDetails || true) {
                const updateBtn = document.createElement('button');
                updateBtn.className = 'bg-green-500 hover:bg-green-600 text-white px-3 py-2 rounded text-sm transition-colors';
                updateBtn.textContent = '🔄 Update';
                updateBtn.onclick = () => updateApp(name);
                rightSection.appendChild(updateBtn);

                const removeBtn = document.createElement('button');
                removeBtn.className = 'bg-red-500 hover:bg-red-600 text-white px-3 py-2 rounded text-sm transition-colors';
                removeBtn.textContent = '🗑️ Remove';
                removeBtn.onclick = () => removeApp(name);
                rightSection.appendChild(removeBtn);
            }
        } else {
            const removingText = document.createElement('span');
            removingText.className = 'text-orange-600 font-semibold';
            removingText.textContent = 'Removing...';
            rightSection.appendChild(removingText);
        }

        item.appendChild(leftSection);
        item.appendChild(rightSection);
        container.appendChild(item);
    });
}

function renderInstallationHistory(history) {
    const container = document.getElementById('installation-history');
    container.innerHTML = '';

    if (history.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-sm">No recent activity.</p>';
        return;
    }

    history.slice(-5).reverse().forEach(install => {
        const item = document.createElement('div');
        item.className = 'bg-white rounded p-3 text-sm';

        let statusIcon = '✅';
        let statusText = install.status;
        let timestamp = new Date(install.installed_at || install.updated_at || install.removed_at || install.failed_at).toLocaleString();

        switch (install.status) {
            case 'completed':
                statusIcon = '✅';
                statusText = 'Installed';
                break;
            case 'updated':
                statusIcon = '🔄';
                statusText = 'Updated';
                break;
            case 'removed':
                statusIcon = '🗑️';
                statusText = 'Removed';
                break;
            case 'failed':
            case 'update_failed':
            case 'removal_failed':
                statusIcon = '❌';
                statusText = 'Failed';
                break;
        }

        item.innerHTML = `
          <div class="flex items-center justify-between">
            <span>${statusIcon} <strong>${install.app_name}</strong> - ${statusText}</span>
            <span class="text-gray-500">${timestamp}</span>
          </div>
          ${install.error ? `<p class="text-red-600 text-xs mt-1">${install.error}</p>` : ''}
        `;

        container.appendChild(item);
    });
}

async function startApp(name) {
    try {
        const response = await fetch(`/start/${name}`, { method: 'POST' });
        const result = await response.json();
        if (response.ok) {
            setTimeout(fetchStatus, 1000);
            if (result.process_id) {
                setTimeout(() => {
                    showLogViewer(result.process_id, `Running ${name}`);
                }, 2000);
            }
        }
    } catch (error) {
        console.error('Failed to start app:', error);
    }
}

async function stopApp() {
    try {
        const response = await fetch('/stop', { method: 'POST' });
        if (response.ok) {
            setTimeout(fetchStatus, 1000);
            hideLogViewer();
        }
    } catch (error) {
        console.error('Failed to stop app:', error);
    }
}

async function updateApp(appName) {
    if (!confirm(`Update "${appName}" to the latest version?`)) {
        return;
    }

    try {
        const response = await fetch(`/api/apps/${appName}/update`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const result = await response.json();

        if (!response.ok) {
            alert(`Failed to start update: ${result.detail}`);
        }
    } catch (error) {
        console.error('Failed to update app:', error);
        alert(`Failed to update app: ${error.message}`);
    }
}

async function removeApp(appName) {
    if (!confirm(`Are you sure you want to remove "${appName}"? This will permanently delete the app and its virtual environment.`)) {
        return;
    }

    activeOperations.add(appName);
    const appItem = document.getElementById(`app-item-${appName}`);
    if (appItem) {
        appItem.classList.add('removing');
    }

    try {
        const response = await fetch(`/api/apps/${appName}`, { method: 'DELETE' });
        const result = await response.json();

        if (!response.ok) {
            alert(`Failed to start removal: ${result.detail}`);
            activeOperations.delete(appName);
            if (appItem) {
                appItem.classList.remove('removing');
            }
        }
    } catch (error) {
        console.error('Failed to remove app:', error);
        alert(`Failed to remove app: ${error.message}`);
        activeOperations.delete(appName);
        if (appItem) {
            appItem.classList.remove('removing');
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// HF Spaces Store functionality
class SpacesStore {
    constructor() {
        this.spaces = [];
        this.filteredSpaces = [];
        this.currentSort = 'likes';
        this.searchTerm = '';
    }

    async init() {
        await this.loadSpaces();
        this.setupEventListeners();
        this.renderSpaces();
    }

    async loadSpaces() {
        try {
            // Search for spaces with the reachy_mini tag
            const response = await fetch('https://huggingface.co/api/spaces?filter=reachy_mini&sort=likes&direction=-1&limit=50');
            const data = await response.json();

            this.spaces = data.map(space => ({
                id: space.id,
                title: space.id.split('/').pop().replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                author: space.author,
                description: space.cardData?.short_description || 'No description available',
                likes: space.likes || 0,
                created: new Date(space.createdAt).getTime(),
                url: `https://huggingface.co/spaces/${space.id}`,
                installUrl: `https://huggingface.co/spaces/${space.id}`,
                tags: space.tags || []
            }));

            this.filteredSpaces = [...this.spaces];
            this.updateStats();
        } catch (error) {
            console.error('Error loading spaces:', error);
            this.showError();
        }
    }

    setupEventListeners() {
        const searchInput = document.getElementById('spaces-search');
        searchInput.addEventListener('input', (e) => {
            this.searchTerm = e.target.value.toLowerCase();
            this.filterSpaces();
        });

        const sortButtons = document.querySelectorAll('.sort-btn');
        sortButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                sortButtons.forEach(b => {
                    b.classList.remove('active', 'bg-blue-500', 'text-white');
                    b.classList.add('bg-gray-200', 'text-gray-700');
                });
                e.target.classList.remove('bg-gray-200', 'text-gray-700');
                e.target.classList.add('active', 'bg-blue-500', 'text-white');
                this.currentSort = e.target.dataset.sort;
                this.sortSpaces();
            });
        });
    }

    filterSpaces() {
        this.filteredSpaces = this.spaces.filter(space =>
            space.title.toLowerCase().includes(this.searchTerm) ||
            space.author.toLowerCase().includes(this.searchTerm) ||
            space.description.toLowerCase().includes(this.searchTerm)
        );
        this.sortSpaces();
    }

    sortSpaces() {
        switch (this.currentSort) {
            case 'likes':
                this.filteredSpaces.sort((a, b) => b.likes - a.likes);
                break;
            case 'created':
                this.filteredSpaces.sort((a, b) => b.created - a.created);
                break;
            case 'name':
                this.filteredSpaces.sort((a, b) => a.title.localeCompare(b.title));
                break;
        }
        this.renderSpaces();
    }

    updateStats() {
        const statsEl = document.getElementById('spaces-stats');
        const total = this.spaces.length;
        const totalLikes = this.spaces.reduce((sum, space) => sum + space.likes, 0);
        statsEl.innerHTML = `Found ${total} spaces with ${totalLikes.toLocaleString()} total likes`;
    }

    renderSpaces() {
        const grid = document.getElementById('spaces-grid');

        if (this.filteredSpaces.length === 0) {
            grid.innerHTML = '<div class="col-span-full text-center text-gray-500 py-8">No spaces found matching your criteria</div>';
            return;
        }

        grid.innerHTML = this.filteredSpaces.map(space => `
          <div class="spaces-card p-4 cursor-pointer">
            <div class="flex items-start space-x-3 mb-3">
              <div class="space-icon">
                ${space.title.charAt(0)}
              </div>
              <div class="flex-1 min-w-0">
                <div class="font-semibold text-gray-800 truncate">${space.title}</div>
                <div class="text-sm text-gray-600">by ${space.author}</div>
              </div>
            </div>
            <div class="text-sm text-gray-700 mb-4 line-clamp-3">${space.description}</div>
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-4 text-sm text-gray-500">
                <div class="flex items-center space-x-1">
                  <span>❤️</span>
                  <span>${space.likes}</span>
                </div>
                <div class="flex items-center space-x-1">
                  <span>📅</span>
                  <span>${this.formatDate(space.created)}</span>
                </div>
              </div>
              <button onclick="installFromSpace('${space.installUrl}', '${space.title}')" 
                      class="bg-purple-500 hover:bg-purple-600 text-white px-3 py-1 rounded text-sm transition-colors">
                📥 Install
              </button>
            </div>
          </div>
        `).join('');
    }

    formatDate(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diffInDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));

        if (diffInDays === 0) return 'Today';
        if (diffInDays === 1) return 'Yesterday';
        if (diffInDays < 30) return `${diffInDays}d ago`;
        if (diffInDays < 365) return `${Math.floor(diffInDays / 30)}mo ago`;
        return `${Math.floor(diffInDays / 365)}y ago`;
    }

    showError() {
        const grid = document.getElementById('spaces-grid');
        const stats = document.getElementById('spaces-stats');

        stats.innerHTML = 'Unable to load spaces';
        grid.innerHTML = `
          <div class="col-span-full text-center text-gray-500 py-8">
            <h3 class="font-semibold mb-2">Unable to load Hugging Face Spaces</h3>
            <p class="text-sm">This might be due to CORS restrictions. In a production environment, you'd use a backend API to fetch the data.</p>
          </div>
        `;
    }
}

function initializeSpacesStore() {
    spacesStore = new SpacesStore();
    spacesStore.init();
}

async function installFromSpace(spaceUrl, spaceName) {
    const sanitizedName = spaceName.replace(/[^a-zA-Z0-9-_]/g, '_').toLowerCase();

    if (!confirm(`Install "${spaceName}" from Hugging Face Space?`)) {
        return;
    }

    try {
        const response = await fetch('/api/install', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url: spaceUrl,
                name: sanitizedName
            })
        });

        const result = await response.json();

        if (response.ok) {
            // Scroll to top to see installation progress
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } else {
            alert(`Failed to start installation: ${result.detail}`);
        }
    } catch (error) {
        console.error('Failed to install from space:', error);
        alert(`Failed to install: ${error.message}`);
    }
}

// Event listeners for log viewer controls
document.getElementById('auto-scroll-toggle').addEventListener('click', toggleAutoScroll);
document.getElementById('clear-logs-btn').addEventListener('click', clearCurrentLogs);
document.getElementById('close-logs-btn').addEventListener('click', hideLogViewer);

// Initialize everything
connectWebSocket();
fetchStatus();
fetchSimulationStatus();
fetchDaemonStatus();
initializeSpacesStore(); // Load spaces on page load
setInterval(fetchStatus, 10000);
setInterval(fetchDaemonStatus, 15000);