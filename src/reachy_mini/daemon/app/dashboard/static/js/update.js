const updateManager = {
    busy: false,
    preRelease: false,

    // Check for PyPI updates
    checkForUpdate: async () => {
        await updateManager.updateUI();
        try {
            const response = await fetch('/update/available?pre_release=' + updateManager.preRelease);
            if (response.ok) {
                const data = await response.json();
                await updateManager.updateUI(data);
            }
        } catch (error) {
            console.error('Error checking for updates:', error);
        }
    },

    // Start PyPI update
    startUpdate: async () => {
        if (updateManager.busy) return;
        updateManager.busy = true;

        try {
            const response = await fetch('/update/start?pre_release=' + updateManager.preRelease, { method: 'POST' });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Error starting update');
            updateManager.connectLogsWebSocket(data.job_id);
        } catch (error) {
            console.error('Error triggering update:', error);
            updateManager.busy = false;
        }
    },

    // Install from a specific git ref (tag/branch)
    installFromRef: async () => {
        if (updateManager.busy) return;

        const gitRef = document.getElementById('git-ref-input').value.trim();
        const errorDiv = document.getElementById('git-ref-error');
        errorDiv.classList.add('hidden');

        if (!gitRef) {
            errorDiv.textContent = 'Please enter a tag or branch name';
            errorDiv.classList.remove('hidden');
            return;
        }

        // Validate ref exists on GitHub
        try {
            const validateResp = await fetch('/update/validate-ref?git_ref=' + encodeURIComponent(gitRef));
            const validateData = await validateResp.json();
            if (!validateData.valid) {
                errorDiv.textContent = validateData.error || `Ref '${gitRef}' not found`;
                errorDiv.classList.remove('hidden');
                return;
            }
        } catch (error) {
            errorDiv.textContent = 'Failed to validate ref: ' + error.message;
            errorDiv.classList.remove('hidden');
            return;
        }

        // Start installation
        updateManager.busy = true;
        try {
            const response = await fetch('/update/start-from-ref?git_ref=' + encodeURIComponent(gitRef), { method: 'POST' });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Error starting install');
            updateManager.connectLogsWebSocket(data.job_id);
        } catch (error) {
            console.error('Error installing from ref:', error);
            updateManager.busy = false;
        }
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