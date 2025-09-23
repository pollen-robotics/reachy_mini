
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

    checkDaemonStatus();
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
    applyDaemonStatus(status.state);
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
        applyDaemonStatus('starting');

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
            })
            .finally(() => {
                toggle.disabled = false;
                dashboard.classList.remove('daemon-dashboard-loading');
                overlay.classList.remove('active');
            });
    } else {
        applyDaemonStatus('stopping');

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
            })
            .finally(() => {
                toggle.disabled = false;
                dashboard.classList.remove('daemon-dashboard-loading');
                overlay.classList.remove('active');
            });
    }
};

let currentDaemonState;
const applyDaemonStatus = (state) => {
    if (state === currentDaemonState) return;
    currentDaemonState = state;

    console.log('Applying daemon state:', state);

    const overlay = document.getElementById('daemon-loading-overlay');
    overlay.classList.remove('active');

    const toggle = document.getElementById('daemon-toggle');

    const thumb = document.getElementById('daemon-mini-thumb')
    thumb.classList = 'rounded-sm'; // Reset classes

    thumb.innerHTML = ''; // Clear previous content
    const thumbImgObj = document.createElement('object');
    thumbImgObj.type = 'image/svg+xml';
    thumbImgObj.classList = 'w-full';
    thumb.appendChild(thumbImgObj);

    if (currentDaemonState === 'starting') {
        toggle.checked = true;
        thumb.classList.add('starting');
        overlay.classList.add('active');
        thumbImgObj.data = '/dashboard/assets/reachy-mini-wake-up-animation.svg';
    }
    else if (currentDaemonState === 'running') {
        toggle.checked = true;
        thumb.classList.add('running');
        thumbImgObj.data = '/dashboard/assets/reachy-mini-awake.svg';
    }
    else if (currentDaemonState === 'stopping') {
        toggle.checked = false;
        thumb.classList.add('stopping');
        overlay.classList.add('active');
        thumbImgObj.data = '/dashboard/assets/reachy-mini-go-to-sleep-animation.svg';
    }
    else if (currentDaemonState === 'stopped') {
        toggle.checked = false;
        thumb.classList.add('stopped');
        thumbImgObj.data = '/dashboard/assets/reachy-mini-sleeping.svg';
    } else if (currentDaemonState === 'error') {
        toggle.checked = false;
        thumb.classList.add('error');
        thumbImgObj.data = '/dashboard/assets/reachy-mini-ko-animation.svg';
    } else {
        console.error('Unknown daemon state:', currentDaemonState);
        return;
    }
};