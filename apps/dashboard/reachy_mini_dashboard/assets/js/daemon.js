let daemonStatus = null;


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
    indicator.className = 'status-indicator';

    const statusText = document.getElementById('daemon-status-text');
    // const robotInfo = document.getElementById('robot-info');
    // const robotDetails = document.getElementById('robot-details');
    const startBtn = document.getElementById('daemon-start-btn');
    const stopBtn = document.getElementById('daemon-stop-btn');
    // const restartBtn = document.getElementById('daemon-restart-btn');


    if (status.state === "running") {
        indicator.classList.add('status-running');
        statusText.textContent = status.message || 'Daemon is running';

        startBtn.disabled = true;
        startBtn.classList.remove('starting');
        startBtn.classList.add('start');
        startBtn.innerHTML = "▶️ Start";

        stopBtn.disabled = false;

    } else if (status.state === "starting") {
        indicator.classList.add('status-starting');
        statusText.textContent = status.message || 'Starting daemon...';

        startBtn.disabled = true;
        startBtn.classList.remove('start');
        startBtn.classList.add('starting');
        startBtn.innerHTML = "⏳ Starting...";

        stopBtn.disabled = true;

    } else if (status.state === "stopping") {
        indicator.classList.add('status-stopping');
        statusText.textContent = status.message || 'Stopping daemon...';

        startBtn.disabled = true;

        stopBtn.disabled = true;
        stopBtn.classList.remove('stop');
        stopBtn.classList.add('stopping');
        stopBtn.innerHTML = "⏳ Stopping...";

    } else if (status.state === "stopped") {
        indicator.classList.add('status-stopped');
        statusText.textContent = status.message || 'Daemon is stopped';

        startBtn.disabled = false;

        stopBtn.disabled = true;
        stopBtn.classList.remove('stopping');
        stopBtn.classList.add('stop');
        stopBtn.innerHTML = "⏹️ Stop";

    } else if (status.state === "error") {
        indicator.classList.add('status-error');
        statusText.textContent = status.message || 'Error: Daemon is not running';

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
}

async function controlDaemon(action) {
    const button = document.getElementById(`daemon-${action}-btn`);
    // const originalText = button.innerHTML;
    button.disabled = true;
    // button.innerHTML = '⏳ Working...';

    try {
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
        console.error(`Failed to ${action} daemon:`, error);
        alert(`Failed to ${action} daemon: ${error.message}`);
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


fetchDaemonStatus();
setInterval(fetchDaemonStatus, 200);