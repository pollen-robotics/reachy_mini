const updateManager = {
    busy: false,
    preRelease: false,

    checkForUpdate: async () => {
        await updateManager.updateUI();  // Clear previous state

        await fetch('/update/available?pre_release=' + updateManager.preRelease)
            .then(async response => {
                if (!response.ok) {
                    return false;
                }
                const data = await response.json();
                await updateManager.updateUI(data);
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
        const updateModalTitle = document.getElementById('update-modal-title');
        const logsDiv = document.getElementById('update-logs');
        const closeButton = document.getElementById('update-modal-close-button');

        updateModalTitle.textContent = 'Updating...';

        closeButton.onclick = () => {
            updateModal.classList.add('hidden');
        };

        updateModal.classList.remove('hidden');

        const ws = new WebSocket(`ws://${location.host}/update/ws/logs?job_id=${jobId}`);

        ws.onmessage = (event) => {
            // console.log('Update log:', event);
            logsDiv.innerHTML += event.data + '<br>';
            logsDiv.scrollTop = logsDiv.scrollHeight;
        };
        ws.onclose = async () => {
            console.log('Update logs WebSocket closed');
            closeButton.classList.remove('hidden');
            closeButton.textContent = 'Close';
            updateModalTitle.textContent = 'Update Completed ✅';

            updateManager.busy = false;
            await updateManager.checkForUpdate();
        };
    },

    updateUI: async (update) => {
        // updateManager.updateMainPage(isUpdateAvailable);
        updateManager.updateUpdatePage(update);
    },

    // updateMainPage: async (update) => {
    //     const daemonUpdateBtn = document.getElementById('daemon-update-btn');
    //     if (!daemonUpdateBtn) return;

    //     if (isUpdateAvailable) {
    //         daemonUpdateBtn.innerHTML = 'Update <span class="rounded-full bg-blue-700 text-white text-xs font-semibold px-2 py-1 ml-2">1</span>';
    //     } else {
    //         daemonUpdateBtn.innerHTML = 'Update';
    //     }
    // },
    updateUpdatePage: async (data) => {
        const statusElem = document.getElementById('update-status');
        if (!statusElem) return;

        const currentVersionElem = document.getElementById('current-version');
        const availableVersionElem = document.getElementById('available-version');
        const availableVersionContainer = document.getElementById('available-version-container');
        const startUpdateBtn = document.getElementById('start-update-btn');

        if (!data || !data.update || !data.update.reachy_mini) {
            statusElem.innerHTML = 'Checking for updates...';
            if (currentVersionElem) currentVersionElem.textContent = '';
            if (availableVersionElem) availableVersionElem.textContent = '';
            return;
        }

        const updateInfo = data.update.reachy_mini;
        const isUpdateAvailable = updateInfo.is_available;
        const currentVersion = updateInfo.current_version || '-';
        const availableVersion = updateInfo.available_version || '-';

        if (currentVersionElem) currentVersionElem.textContent = `Current version: ${currentVersion}`;
        if (availableVersionElem) availableVersionElem.textContent = `Available version: ${availableVersion}`;

        if (isUpdateAvailable) {
            statusElem.innerHTML = 'An update is available!';
            if (availableVersionContainer) availableVersionContainer.classList.remove('hidden');
            startUpdateBtn.classList.remove('hidden');
        } else {
            statusElem.innerHTML = 'Your system is up to date.';
            if (availableVersionContainer) availableVersionContainer.classList.add('hidden');
            startUpdateBtn.classList.add('hidden');
        }
    }
};

window.addEventListener('load', async () => {
    await updateManager.checkForUpdate();
});