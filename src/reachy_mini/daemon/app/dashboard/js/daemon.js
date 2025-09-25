
let lost_svg = null;

window.onload = async () => {
    const collapseToggle = document.getElementById('collapse-toggle');
    const collapseContent = document.getElementById('collapse-content');
    const plugRow = collapseToggle && collapseToggle.parentElement;

    if (collapseToggle && plugRow) {
        collapseToggle.onclick = () => {
            let expanded = !collapseContent.classList.contains('hidden');
            expanded = !expanded;
            collapseToggle.textContent = expanded ? '>' : '<';
            if (collapseContent) {
                collapseContent.classList.toggle('hidden', !expanded);
            }
        };
    }
    document.getElementById('daemon-toggle').onchange = onDaemonToggleSwitch;

    await fetch('/dashboard/assets/reachy-mini-connection-lost-animation.svg')
        .then(response => response.text())
        .then(svg => {
            lost_svg = svg;
        });

    checkDaemonStatus();
}

const getDaemonStatus = async () => {
    return await fetch('/api/daemon/status')
        .then(response => response.json())
        .then(status => {
            applyDaemonStatus(status);
            return status;
        })
        .catch(error => {
            console.error('Error fetching daemon status:', error);
        });
};

const checkDaemonStatus = async () => {
    const status = await getDaemonStatus();
    if (!status) {
        setTimeout(checkDaemonStatus, 1000); // Retry after 1 second if failed
        return;
    }

    let ws = new WebSocket(`ws://${window.location.host}/api/daemon/status/ws/notifications`);

    ws.onmessage = (event) => {
        const status = JSON.parse(event.data);
        applyDaemonStatus(status);
    };

    ws.onclose = (event) => {
        applyDaemonStatus({ state: 'lost-connection', error: 'Connection to daemon lost' });
        setTimeout(checkDaemonStatus, 1000); // Retry connection after 1 second
    }
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
        fetch(`/api/daemon/start?wake_up=${wakeUpOnStart}`, { method: 'POST' })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to start daemon');
                }
                return response.json();
            })
            .catch(error => {
                toggle.checked = false;
            })
            .finally(() => {
                toggle.disabled = false;
                dashboard.classList.remove('daemon-dashboard-loading');
            });
    } else {
        fetch(`/api/daemon/stop?goto_sleep=${gotoSleepOnStop}`, { method: 'POST' })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to stop daemon');
                }
                return response.json();
            })
            .catch(error => {
                toggle.checked = true; // Revert toggle on error
            })
            .finally(() => {
                toggle.disabled = false;
                dashboard.classList.remove('daemon-dashboard-loading');
            });
    }
};

let currentDaemonState;
const applyDaemonStatus = (status) => {
    console.log('Applying daemon status:', status);

    // State update
    if (status.state === currentDaemonState) return;
    currentDaemonState = status.state;

    const overlay = document.getElementById('daemon-loading-overlay');
    const toggle = document.getElementById('daemon-toggle');
    const toggleSlider = document.getElementById('daemon-toggle-slider');
    const toggleOnLabel = document.getElementById('daemon-toggle-on');
    const toggleOffLabel = document.getElementById('daemon-toggle-off');
    const thumb = document.getElementById('daemon-mini-thumb');
    const backendStatus = document.getElementById('backend-status');
    const backendStatusIcon = document.getElementById('backend-status-icon');
    const backendStatusText = document.getElementById('backend-status-text');
    const collapseToggle = document.getElementById('collapse-toggle');
    const collapseContent = document.getElementById('collapse-content');

    toggleSlider.classList.remove('hidden');
    overlay.classList.remove('active');
    thumb.innerHTML = ''; // Clear previous content
    const thumbImgObj = document.createElement('object');
    thumbImgObj.type = 'image/svg+xml';
    thumbImgObj.classList = 'w-full';
    thumb.appendChild(thumbImgObj);

    backendStatus.classList = ''; // Reset classes
    backendStatusIcon.classList = 'inline-block  w-4 h-4 rounded-full align-middle mr-2';
    collapseToggle.classList.remove('hidden');
    collapseContent.innerHTML = ''; // Clear previous content

    if (currentDaemonState === 'starting') {
        toggle.checked = true;
        overlay.classList.add('active');
        thumbImgObj.data = '/dashboard/assets/reachy-mini-wake-up-animation.svg';
        backendStatusIcon.classList.add('bg-yellow-500');
        backendStatusText.textContent = 'Waking up...';
        collapseToggle.classList.add('hidden');
    }
    else if (currentDaemonState === 'running') {
        toggle.checked = true;
        thumbImgObj.data = '/dashboard/assets/reachy-mini-awake.svg';
        backendStatusIcon.classList.add('bg-green-500');
        backendStatusText.textContent = 'Up and ready';
        collapseToggle.classList.add('hidden');
    }
    else if (currentDaemonState === 'stopping') {
        toggle.checked = false;
        overlay.classList.add('active');
        thumbImgObj.data = '/dashboard/assets/reachy-mini-go-to-sleep-animation.svg';
        backendStatusIcon.classList.add('bg-yellow-500');
        backendStatusText.textContent = 'Going to sleep...';
        collapseToggle.classList.add('hidden');
    }
    else if (currentDaemonState === 'stopped' || currentDaemonState === 'not_initialized') {
        toggle.checked = false;
        thumbImgObj.data = '/dashboard/assets/reachy-mini-sleeping.svg';
        backendStatus.classList.add('hidden');
    } else if (currentDaemonState === 'error') {
        toggle.checked = false;
        thumbImgObj.data = '/dashboard/assets/reachy-mini-ko-animation.svg';
        backendStatusIcon.classList.add('bg-red-500');
        backendStatusText.textContent = 'Something bad happened';
        collapseToggle.textContent = '>'; // Reset to default
        collapseContent.innerHTML = status.error;
        collapseContent.classList.remove('hidden');
    }
    else if (currentDaemonState === 'lost-connection') {
        toggleSlider.classList.add('hidden');
        overlay.classList.add('active');
        backendStatusIcon.classList.add('bg-red-500');
        backendStatusText.textContent = 'Connection lost';
        collapseToggle.textContent = '>'; // Reset to default
        collapseContent.innerHTML = 'Will try to reconnect automatically...';
        collapseContent.classList.remove('hidden');
        thumb.innerHTML = lost_svg;
    }
    else {
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


const preload_svg_lost_connection = () => {
    fetch('/dashboard/assets/reachy-mini-connection-lost-animation.svg')
        .then(response => response.text())
        .then(svg => {
            lost_svg = svg;
        });
}
