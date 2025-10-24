const updateManager = {
    busy: false,
    preRelease: false,

    isUpdateAvailable: async () => {
        return Math.random() < 0.5;

        fetch('/update/available?pre_release=' + updateManager.preRelease)
            .then(response => response.json())
            .then(data => {
                return data.update.reachy_mini;
            }).catch(error => {
                console.error('Error checking for updates:', error);
            });
    },

    startUpdate: async () => {
        if (updateManager.busy) {
            console.warn('An update is already in progress.');
            return;
        }
        updateManager.busy = true;

        fetch('/update/start?pre_release=' + updateManager.preRelease, { method: 'POST' })
            .then(response => {
                if (response.ok) {
                    return response.json();
                } else {
                    return response.json().then(data => {
                        throw new Error(data.detail || 'Error starting update');
                    });
                }
            })
            .then(data => {
                const jobId = data.job_id;
                updateManager.connectLogsWebSocket(jobId);
            })
            .catch(error => {
                console.error('Error triggering update:', error);
                updateManager.busy = false;
            });
    },

    connectLogsWebSocket: (jobId) => {
        const updateModal = document.getElementById('update-modal');
        const updateModalTitle = updateModal.queryElementById('update-modal-title');
        const logsDiv = document.getElementById('update-logs');
        const closeButton = document.getElementById('update-modal-close-button');

        updateModalTitle.textContent = 'Updating...';

        closeButton.onclick = () => {
            installModal.classList.add('hidden');
        };
        closeButton.classList = "hidden";
        closeButton.textContent = '';

        updateModal.classList.removeAttribute('hidden');

        const ws = new WebSocket(`ws://${location.host}/api/update/ws/logs?job_id=${jobId}`);

        ws.onmessage = (event) => {
        };
        ws.onclose = async () => {
            updateManager.busy = false;
            updateManager.updateUI();
        };
    },

    updateUI: async () => {
        const isUpdateAvailable = await updateManager.isUpdateAvailable();

        updateManager.updateMainPage(isUpdateAvailable);
        updateManager.updateUpdatePage(isUpdateAvailable);
    },

    updateMainPage: async (isUpdateAvailable) => {
        const daemonUpdateBtn = document.getElementById('daemon-update-btn');
        if (!daemonUpdateBtn) return;

        if (isUpdateAvailable) {
            daemonUpdateBtn.innerHTML = 'Update <span class="rounded-full bg-blue-700 text-white text-xs font-semibold px-2 py-1 ml-2">1</span>';
        } else {
            daemonUpdateBtn.innerHTML = 'Update';
        }
    },
    updateUpdatePage: async (isUpdateAvailable) => {
        const statusElem = document.getElementById('update-status');
        if (!statusElem) return;

        const checkAgainBtn = document.getElementById('check-again-btn');
        const startUpdateBtn = document.getElementById('start-update-btn');

        if (isUpdateAvailable) {
            statusElem.innerHTML = 'An update is available!';
            checkAgainBtn.classList.add('hidden');
            startUpdateBtn.classList.remove('hidden');
        } else {
            statusElem.innerHTML = 'Your system is up to date.';
            checkAgainBtn.classList.remove('hidden');
            startUpdateBtn.classList.add('hidden');
        }
    }
};

window.addEventListener('load', async () => {
    updateManager.updateUI();
});

// async function checkUpdates() {
//     const statusElem = document.getElementById('status');
//     const updateBtn = document.getElementById('update-btn');
//     statusElem.textContent = 'Checking for updates...';
//     updateBtn.style.display = 'none';

//     try {
//         const response = await fetch('/update/available');
//         const data = await response.json();
//         if (data.update.reachy_mini) {
//             statusElem.textContent = 'An update is available!';
//             updateBtn.style.display = 'inline-block';
//         } else {
//             statusElem.textContent = 'Your system is up to date.';
//         }
//     } catch (e) {
//         statusElem.textContent = 'Error checking for updates.';
//     }
// }

// async function triggerUpdate() {
//     const statusElem = document.getElementById('status');
//     statusElem.textContent = 'Updating...';
//     try {
//         const response = await fetch('/update/start', { method: 'POST' });
//         if (response.ok) {
//             await response.json().then(data => {
//                 connectLogsWebSocket(data.job_id);
//             });
//             statusElem.textContent = 'Update started!';

//         } else {
//             await response.json().then(data => {
//                 if (data.detail) {
//                     statusElem.textContent = 'Error: ' + data.detail;
//                 }
//             });
//         }
//     } catch (e) {
//         statusElem.textContent = 'Error triggering update.';
//     }
// }


// function connectLogsWebSocket(jobId) {
//     const logsElem = document.getElementById('logs');
//     if (!logsElem) return;

//     let wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
//     let wsUrl = wsProto + '://' + window.location.host + '/update/ws/logs?job_id=' + jobId;
//     const ws = new WebSocket(wsUrl);

//     ws.onmessage = function (event) {
//         // Append new log line
//         logsElem.textContent += event.data + '\n';
//         logsElem.scrollTop = logsElem.scrollHeight;
//     };
//     ws.onerror = function () {
//         logsElem.textContent += '[WebSocket error connecting to logs]\n';
//     };
//     ws.onclose = function () {
//         logsElem.textContent += '[WebSocket closed]\n';

//         const statusElem = document.getElementById('status');
//         statusElem.textContent = 'Update completed. Reloading in a 5 seconds...';

//         for (let i = 5; i > 0; i--) {
//             setTimeout(() => {
//                 statusElem.textContent = 'Update completed. Reloading in ' + i + ' seconds...';
//             }, (5 - i) * 1000);
//         }
//         setTimeout(() => {
//             window.location.reload();
//         }, 5000);
//     };
// }

// window.onload = function () {
//     checkUpdates();
// };
