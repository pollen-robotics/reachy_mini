


const daemon = {
    currentStatus: {},

    start: async (wakeUp) => {
        fetch(`/api/daemon/start?wake_up=${wakeUp}`, {
            method: 'POST',
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(async (data) => {
                await daemon.checkStatusUpdate();
            })
            .catch((error) => {
                console.error('Error starting daemon:', error);
            });
    },

    stop: async (gotoSleep) => {
        fetch(`/api/daemon/stop?goto_sleep=${gotoSleep}`, {
            method: 'POST',
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(async (data) => {
                await daemon.checkStatusUpdate();
            })
            .catch((error) => {
                console.error('Error stopping daemon:', error);
            });
    },

    getStatus: async () => {
        fetch('/api/daemon/status')
            .then((response) => response.json())
            .then(async (data) => {
                let previousState = daemon.currentStatus.state;
                daemon.currentStatus = data;

                if (previousState !== daemon.currentStatus.state) {
                    await daemon.updateUI();
                }
            })
            .catch((error) => {
                console.error('Error fetching daemon status:', error);
            });
    },

    checkStatusUpdate: async (initialState) => {
        await daemon.getStatus();

        if (!initialState) {
            initialState = daemon.currentStatus.state;
        }

        let currentState = daemon.currentStatus.state;

        if (currentState === initialState || currentState === "starting" || currentState === "stopping") {
            setTimeout(() => {
                daemon.checkStatusUpdate(initialState);
            }, 500);
        }
    },

    toggleSwitch: async () => {
        const toggleDaemonSwitch = document.getElementById('daemon-toggle');

        if (toggleDaemonSwitch.checked) {
            console.log('Toggle switched ON. Starting daemon...');
            await daemon.start(true);
        } else {
            console.log('Toggle switched OFF. Stopping daemon...');
            await daemon.stop(true);
        }

        await daemon.updateToggle();
    },

    updateUI: async () => {
        const daemonStatusAnim = document.getElementById('daemon-status-anim');
        const toggleDaemonSwitch = document.getElementById('daemon-toggle');
        const backendStatusIcon = document.getElementById('backend-status-icon');
        const backendStatusText = document.getElementById('backend-status-text');

        let daemonState = daemon.currentStatus.state;

        toggleDaemonSwitch.disabled = false;
        backendStatusIcon.classList.remove('bg-green-500', 'bg-yellow-500', 'bg-red-500');

        if (daemonState === 'starting') {
            daemonStatusAnim.setAttribute('data', '/static/assets/reachy-mini-wake-up-animation.svg');
            toggleDaemonSwitch.disabled = true;
            toggleDaemonSwitch.checked = true;
            backendStatusIcon.classList.add('bg-yellow-500');
            backendStatusText.textContent = 'Waking up...';
        }
        else if (daemonState === 'running') {
            daemonStatusAnim.setAttribute('data', '/static/assets/reachy-mini-awake.svg');
            toggleDaemonSwitch.checked = true;
            backendStatusIcon.classList.add('bg-green-500');
            backendStatusText.textContent = 'Up and ready';
        }
        else if (daemonState === 'stopping') {
            daemonStatusAnim.setAttribute('data', '/static/assets/reachy-mini-go-to-sleep-animation.svg');
            toggleDaemonSwitch.disabled = true;
            toggleDaemonSwitch.checked = false;
            backendStatusIcon.classList.add('bg-yellow-500');
            backendStatusText.textContent = 'Going to sleep...';
        }
        else if (daemonState === 'stopped' || daemonState === 'not_initialized') {
            daemonStatusAnim.setAttribute('data', '/static/assets/reachy-mini-sleeping.svg');
            toggleDaemonSwitch.checked = false;
            backendStatusIcon.classList.add('bg-yellow-500');
            backendStatusText.textContent = 'Stopped';
        }
        else if (daemonState === 'error') {
            daemonStatusAnim.setAttribute('data', '/static/assets/reachy-mini-ko-animation.svg');
            toggleDaemonSwitch.checked = false;
            backendStatusIcon.classList.add('bg-red-500');
            backendStatusText.textContent = 'Error occurred';
        }

        await daemon.updateToggle();
    },

    updateToggle: async () => {
        const toggle = document.getElementById('daemon-toggle');
        const toggleSlider = document.getElementById('daemon-toggle-slider');
        const toggleOnLabel = document.getElementById('daemon-toggle-on');
        const toggleOffLabel = document.getElementById('daemon-toggle-off');

        toggleSlider.classList.remove('hidden');

        if (toggle.checked) {
            toggleOnLabel.classList.remove('hidden');
            toggleOffLabel.classList.add('hidden');
        } else {
            toggleOnLabel.classList.add('hidden');
            toggleOffLabel.classList.remove('hidden');
        }
    },
};


window.addEventListener('load', async () => {
    document.getElementById('daemon-toggle').onchange = daemon.toggleSwitch;
    await daemon.getStatus();
});