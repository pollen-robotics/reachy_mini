
window.onload = () => {
    // Collapse/expand logic
    const collapseToggle = document.getElementById('collapse-toggle');
    const collapseContent = document.getElementById('collapse-content');
    const plugRow = collapseToggle && collapseToggle.parentElement;

    let expanded = false;
    if (collapseToggle && plugRow) {
        collapseToggle.onclick = () => {
            expanded = !expanded;
            collapseToggle.textContent = expanded ? '>' : '<';
            if (collapseContent) {
                collapseContent.classList.toggle('hidden', !expanded);
            }
        };
    }
    document.getElementById('daemon-toggle').onchange = onDaemonToggleSwitch;

    // checkDaemonStatus();
    setInterval(checkDaemonStatus, 2000);
}

const getDaemonStatus = async () => {
    return await fetch('/api/daemon/status')
        .then(response => response.json())
        .then(data => {
            return data;
        })
        .catch(error => {
            console.error('Error fetching daemon status:', error);
        });
};

const checkDaemonStatus = async () => {
    const status = await getDaemonStatus();
    applyDaemonStatus(status);
};

const onDaemonToggleSwitch = () => {
    const toggle = document.getElementById('daemon-toggle');
    const dashboard = document.getElementById('daemon-dashboard');
    const overlay = document.getElementById('daemon-loading-overlay');

    const toggleOnLabel = document.getElementById('daemon-toggle-on');
    const toggleOffLabel = document.getElementById('daemon-toggle-off');

    if (toggle.checked) {
        toggleOnLabel.classList.remove('hidden');
        toggleOffLabel.classList.add('hidden');
    } else {
        toggleOnLabel.classList.add('hidden');
        toggleOffLabel.classList.remove('hidden');
    }

    const isOn = toggle.checked;
    const wakeUpOnStart = true;
    const gotoSleepOnStop = true;

    toggle.disabled = true;
    dashboard.classList.add('daemon-dashboard-loading');
    overlay.classList.add('active');

    if (isOn) {
        applyDaemonStatus({ state: 'starting' });

        fetch(`/api/daemon/start?wake_up=${wakeUpOnStart}`, { method: 'POST' })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to start daemon');
                }
                return response.json();
            })
            .then(data => {
                checkDaemonStatus();
            })
            .catch(error => {
                toggle.checked = false; // Revert toggle on error
                checkDaemonStatus();
            })
            .finally(() => {
                toggle.disabled = false;
                dashboard.classList.remove('daemon-dashboard-loading');
                overlay.classList.remove('active');
            });
    } else {
        applyDaemonStatus({ state: 'stopping' });

        fetch(`/api/daemon/stop?goto_sleep=${gotoSleepOnStop}`, { method: 'POST' })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to stop daemon');
                }
                return response.json();
            })
            .then(data => {
                checkDaemonStatus();
            })
            .catch(error => {
                toggle.checked = true; // Revert toggle on error
                checkDaemonStatus();
            })
            .finally(() => {
                toggle.disabled = false;
                dashboard.classList.remove('daemon-dashboard-loading');
                overlay.classList.remove('active');
            });
    }
};



let currentDaemonState;
const applyDaemonStatus = (status) => {
    // State update
    if (status.state === currentDaemonState) return;
    currentDaemonState = status.state;

    const overlay = document.getElementById('daemon-loading-overlay');
    const toggle = document.getElementById('daemon-toggle');
    const toggleOnLabel = document.getElementById('daemon-toggle-on');
    const toggleOffLabel = document.getElementById('daemon-toggle-off');
    const thumb = document.getElementById('daemon-mini-thumb')
    const thumbImgObj = document.createElement('object');
    const backendStatus = document.getElementById('backend-status');
    const backendStatusIcon = document.getElementById('backend-status-icon');
    const backendStatusText = document.getElementById('backend-status-text');
    const collapseToggle = document.getElementById('collapse-toggle');
    const collapseContent = document.getElementById('collapse-content');

    overlay.classList.remove('active');
    thumb.classList = 'rounded-sm'; // Reset classes
    thumb.innerHTML = ''; // Clear previous content
    thumbImgObj.type = 'image/svg+xml';
    thumbImgObj.classList = 'w-full';
    thumb.appendChild(thumbImgObj);

    backendStatus.classList = ''; // Reset classes
    backendStatusIcon.classList = 'inline-block  w-4 h-4 rounded-full align-middle mr-2';
    collapseToggle.classList.remove('hidden');
    collapseContent.innerHTML = ''; // Clear previous content

    if (currentDaemonState === 'starting') {
        toggle.checked = true;
        thumb.classList.add('starting');
        overlay.classList.add('active');
        thumbImgObj.data = '/dashboard/assets/reachy-mini-wake-up-animation.svg';
        backendStatusIcon.classList.add('bg-yellow-500');
        backendStatusText.textContent = 'Waking up...';
        collapseToggle.classList.add('hidden');
    }
    else if (currentDaemonState === 'running') {
        toggle.checked = true;
        thumb.classList.add('running');
        thumbImgObj.data = '/dashboard/assets/reachy-mini-awake.svg';
        backendStatusIcon.classList.add('bg-green-500');
        backendStatusText.textContent = 'Up and ready';
        collapseToggle.classList.add('hidden');
    }
    else if (currentDaemonState === 'stopping') {
        toggle.checked = false;
        thumb.classList.add('stopping');
        overlay.classList.add('active');
        thumbImgObj.data = '/dashboard/assets/reachy-mini-go-to-sleep-animation.svg';
        backendStatusIcon.classList.add('bg-yellow-500');
        backendStatusText.textContent = 'Going to sleep...';
        collapseToggle.classList.add('hidden');
    }
    else if (currentDaemonState === 'stopped' || currentDaemonState === 'not_initialized') {
        toggle.checked = false;
        thumb.classList.add('stopped');
        thumbImgObj.data = '/dashboard/assets/reachy-mini-sleeping.svg';
        backendStatus.classList.add('hidden');
    } else if (currentDaemonState === 'error') {
        toggle.checked = false;
        thumb.classList.add('error');
        thumbImgObj.data = '/dashboard/assets/reachy-mini-ko-animation.svg';
        backendStatusIcon.classList.add('bg-red-500');
        backendStatusText.textContent = 'ERROR';
        collapseContent.innerHTML = status.error;


    } else {
        console.error('Unknown daemon state:', currentDaemonState);
        return;
    }

    if (toggle.checked) {
        toggleOnLabel.classList.remove('hidden');
        toggleOffLabel.classList.add('hidden');
    } else {
        toggleOnLabel.classList.add('hidden');
        toggleOffLabel.classList.remove('hidden');
    }
};