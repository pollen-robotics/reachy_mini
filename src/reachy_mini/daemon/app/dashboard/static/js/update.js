const updateManager = {
    busy: false,
    preRelease: false,

    isUpdateAvailable: async () => {
        await fetch('/update/available?pre_release=' + updateManager.preRelease)
            .then(async response => {
                if (!response.ok) {
                    return false;
                }
                const data = await response.json();
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