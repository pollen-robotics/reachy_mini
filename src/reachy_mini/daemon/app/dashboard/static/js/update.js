async function checkUpdates() {
    const statusElem = document.getElementById('status');
    const updateBtn = document.getElementById('update-btn');
    statusElem.textContent = 'Checking for updates...';
    updateBtn.style.display = 'none';

    try {
        const response = await fetch('/update/available');
        const data = await response.json();
        if (data.update.reachy_mini) {
            statusElem.textContent = 'An update is available!';
            updateBtn.style.display = 'inline-block';
        } else {
            statusElem.textContent = 'Your system is up to date.';
        }
    } catch (e) {
        statusElem.textContent = 'Error checking for updates.';
    }
}

async function triggerUpdate() {
    const statusElem = document.getElementById('status');
    statusElem.textContent = 'Updating...';
    try {
        const response = await fetch('/update/start', { method: 'POST' });
        if (response.ok) {
            await response.json().then(data => {
                connectLogsWebSocket(data.job_id);
            });
            statusElem.textContent = 'Update started!';

        } else {
            await response.json().then(data => {
                if (data.detail) {
                    statusElem.textContent = 'Error: ' + data.detail;
                }
            });
        }
    } catch (e) {
        statusElem.textContent = 'Error triggering update.';
    }
}


function connectLogsWebSocket(jobId) {
    const logsElem = document.getElementById('logs');
    if (!logsElem) return;

    let wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    let wsUrl = wsProto + '://' + window.location.host + '/update/ws/logs?job_id=' + jobId;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = function (event) {
        // Append new log line
        logsElem.textContent += event.data + '\n';
        logsElem.scrollTop = logsElem.scrollHeight;
    };
    ws.onerror = function () {
        logsElem.textContent += '[WebSocket error connecting to logs]\n';
    };
    ws.onclose = function () {
        logsElem.textContent += '[WebSocket closed]\n';

        const statusElem = document.getElementById('status');
        statusElem.textContent = 'Update completed. Reloading in a 5 seconds...';

        for (let i = 5; i > 0; i--) {
            setTimeout(() => {
                statusElem.textContent = 'Update completed. Reloading in ' + i + ' seconds...';
            }, (5 - i) * 1000);
        }
        setTimeout(() => {
            window.location.reload();
        }, 5000);
    };
}

window.onload = function () {
    checkUpdates();
};
