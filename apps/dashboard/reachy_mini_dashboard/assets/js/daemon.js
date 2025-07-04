let daemonStatus = null;
let currentState = null;


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
            message: 'Failed to connect to Reachy Mini',
            running: false
        });
    }
}

function updateDaemonStatusUI(status) {
    if (status.state === currentState && status.state !== "error") {
        // No change in state, no need to update UI
        return;
    }

    console.log('Updating daemon status UI:', status);

    const indicator = document.getElementById('daemon-status-indicator');
    indicator.className = 'status-indicator';

    const statusText = document.getElementById('daemon-status-text');
    const statusDetailText = document.getElementById('daemon-status-detail-text');
    statusDetailText.textContent = '';
    // const robotInfo = document.getElementById('robot-info');
    // const robotDetails = document.getElementById('robot-details');
    const startBtn = document.getElementById('daemon-start-btn');
    const stopBtn = document.getElementById('daemon-stop-btn');
    // const restartBtn = document.getElementById('daemon-restart-btn');
    const toggle = document.getElementById('simulation-toggle');

    if (status.state === "running") {
        indicator.classList.add('status-running');
        // statusText.textContent = status.message || 'Reachy Mini is ready';
        statusText.textContent = '';

        startBtn.disabled = true;
        startBtn.classList.remove('starting');
        startBtn.classList.add('start');
        startBtn.innerHTML = "▶️ Start";

        stopBtn.disabled = false;

    } else if (status.state === "not_initialized") {
        indicator.classList.add('status-not-initialized');
        statusText.textContent = status.message || 'Reachy Mini is not initialized.';

        startBtn.disabled = false;
        startBtn.classList.remove('starting', 'restart');
        startBtn.classList.add('start');
        startBtn.innerHTML = "▶️ Start";

        stopBtn.disabled = true;

        toggle.disabled = false;
    }

    else if (status.state === "starting") {
        indicator.classList.add('status-starting');
        // statusText.textContent = status.message || 'Starting Reachy Mini...';

        startBtn.disabled = true;
        startBtn.classList.remove('start');
        startBtn.classList.add('starting');
        startBtn.innerHTML = "⏳ Starting...";

        stopBtn.disabled = true;

    } else if (status.state === "stopping") {
        indicator.classList.add('status-stopping');
        // statusText.textContent = status.message || 'Stopping Reachy Mini...';

        startBtn.disabled = true;

        stopBtn.disabled = true;
        stopBtn.classList.remove('stop');
        stopBtn.classList.add('stopping');
        stopBtn.innerHTML = "⏳ Stopping...";

    } else if (status.state === "stopped") {
        indicator.classList.add('status-stopped');
        // statusText.textContent = status.message || 'Reachy Mini is stopped';
        statusText.textContent = '';

        startBtn.disabled = false;

        stopBtn.disabled = true;
        stopBtn.classList.remove('stopping');
        stopBtn.classList.add('stop');
        stopBtn.innerHTML = "⏹️ Stop";

        toggle.disabled = false;
    } else if (status.state === "error") {
        indicator.classList.add('status-error');
        statusText.textContent = 'Error: Reachy Mini crashed!';
        statusDetailText.textContent = (status.error || 'Unknown error');

        startBtn.classList.remove('start');
        startBtn.classList.add('restart');
        startBtn.innerHTML = "🔄 Restart";
        startBtn.disabled = false;

        toggle.disabled = false;
    }

    else {
        indicator.classList.add('status-unknown');
        statusText.textContent = status.message || 'Status unknown';
    }

    // if (status.robot_info) {
    //     robotDetails.innerHTML = `
    //       <div>Model: ${status.robot_info.model || 'Unknown'}</div>
    //       <div>Version: ${status.robot_info.version || 'Unknown'}</div>
    //       ${status.robot_info.uptime ? `<div>Uptime: ${status.robot_info.uptime}</div>` : ''}
    //     `;
    //     robotInfo.style.display = 'block';
    // } else {
    //     robotInfo.style.display = 'none';
    // }

    currentState = status.state;
}

async function controlDaemon(action) {


    const button = document.getElementById(`daemon-${action}-btn`);
    // const originalText = button.innerHTML;
    button.disabled = true;
    // button.innerHTML = '⏳ Working...';

    const toggle = document.getElementById('simulation-toggle');
    toggle.disabled = true;

    try {
        if (action === 'start') {
            const startBtn = document.getElementById('daemon-start-btn');
            if (startBtn.classList.contains('restart')) {
                action = 'restart';
            }

        }

        const response = await fetch(`/daemon_${action}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const result = await response.json();
        if (response.ok) {
            // setTimeout(() => fetchDaemonStatus(), 1000);
            // button.innerHTML = '✅ Done';
            // setTimeout(() => button.innerHTML = originalText, 2000);
        } else {
            throw new Error(result.detail || `Failed to ${action} daemon`);
        }
    } catch (error) {
        // console.error(`Failed to ${action} daemon:`, error);
        console.log(error);
        // alert(`Failed to ${action} daemon: ${error.message}`);
        // button.innerHTML = '❌ Error';
        // setTimeout(() => button.innerHTML = originalText, 2000);
    } finally {
        // setTimeout(() => {
        //     button.disabled = false;
        //     if (button.innerHTML === '⏳ Working...' || button.innerHTML === '✅ Done' || button.innerHTML === '❌ Error') {
        //         button.innerHTML = originalText;
        //     }
        // }, 2000);
    }
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

            currentState = null;
            const resetResponse = await fetch('/daemon_reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const resetResult = await resetResponse.json();
            if (!resetResponse.ok) {
                throw new Error(resetResult.detail || 'Failed to reset daemon');
            }

            updateSimulationUI();
            fetchSimulationStatus();

            // const indicator = document.getElementById('simulation-mode-indicator');
            // const originalText = indicator.textContent;
            // indicator.textContent = `🎮 SIMULATION ${simulationEnabled ? 'ENABLED' : 'DISABLED'}`;
            // setTimeout(() => {
            //     if (simulationEnabled) {
            //         indicator.textContent = originalText;
            //     }
            // }, 2000);

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
    // const toggle = document.getElementById('simulation-toggle');
    // const indicator = document.getElementById('simulation-mode-indicator');
    // toggle.checked = simulationEnabled;
    // indicator.style.display = simulationEnabled ? 'block' : 'none';
}


fetchDaemonStatus();
setInterval(fetchDaemonStatus, 1000);
fetchSimulationStatus();