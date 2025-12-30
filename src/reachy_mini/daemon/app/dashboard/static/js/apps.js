const installedApps = {
    refreshAppList: async () => {
        const appsData = await installedApps.fetchInstalledApps();
        await installedApps.displayInstalledApps(appsData);
    },

    currentlyRunningApp: null,
    busy: false,
    toggles: {},

    startApp: async (appName) => {
        if (installedApps.busy) {
            console.log(`Another app is currently being started or stopped.`);
            return;
        }
        installedApps.setBusy(true);

        console.log(`Current running app: ${installedApps.currentlyRunningApp}`);

        if (installedApps.currentlyRunningApp) {
            console.log(`Stopping currently running app: ${installedApps.currentlyRunningApp}...`);
            await installedApps.stopApp(installedApps.currentlyRunningApp, true);
        }

        console.log(`Starting app: ${appName}...`);
        const endpoint = `/api/apps/start-app/${appName}`;
        const resp = await fetch(endpoint, { method: 'POST' });
        if (!resp.ok) {
            console.error(`Failed to staret app ${appName}: ${resp.statusText}`);
            installedApps.toggles[appName].setChecked(false);
            installedApps.setBusy(false);
            return;
        } else {
            console.log(`App ${appName} started successfully.`);
        }

        installedApps.currentlyRunningApp = appName;
        installedApps.setBusy(false);
    },

    stopApp: async (appName, force = false) => {
        if (installedApps.busy && !force) {
            console.log(`Another app is currently being started or stopped.`);
            return;
        }
        installedApps.setBusy(true);

        console.log(`Stopping app: ${appName}...`);

        if (force) {
            console.log(`Force stopping app: ${appName}...`);
            installedApps.toggles[appName].setChecked(false);
        }

        const endpoint = `/api/apps/stop-current-app`;
        const resp = await fetch(endpoint, { method: 'POST' });
        if (!resp.ok) {
            console.error(`Failed to stop app ${appName}: ${resp.statusText}`);
            installedApps.setBusy(false);
            return;
        } else {
            console.log(`App ${appName} stopped successfully.`);
            installedApps.toggles[appName].setChecked(false);
        }

        if (installedApps.currentlyRunningApp === appName) {
            installedApps.currentlyRunningApp = null;
        }
        installedApps.setBusy(false);
    },

    setBusy: (isBusy) => {
        installedApps.busy = isBusy;
        for (const toggle of Object.values(installedApps.toggles)) {
            if (isBusy) {
                toggle.disable();
            } else {
                toggle.enable();
            }
        }
    },

    fetchInstalledApps: async () => {
        const resp = await fetch('/api/apps/list-available/installed');
        const appsData = await resp.json();
        return appsData;
    },

    displayInstalledApps: async (appsData) => {
        const appsListElement = document.getElementById('installed-apps');
        appsListElement.innerHTML = '';

        if (!appsData || appsData.length === 0) {
            appsListElement.innerHTML = '<li>No installed apps found.</li>';
            return;
        }

        const runningApp = await installedApps.getRunningApp();

        installedApps.toggles = {};
        appsData.forEach(app => {
            const li = document.createElement('li');
            li.className = 'app-list-item';
            const isRunning = (app.name === runningApp);
            li.appendChild(installedApps.createAppElement(app, isRunning));
            appsListElement.appendChild(li);
        });
    },

    createAppElement: (app, isRunning) => {
        const container = document.createElement('div');
        container.className = 'grid grid-cols-[auto_6rem_2rem] justify-stretch gap-x-2';

        const title = document.createElement('div');
        const titleSpan = document.createElement('span');
        titleSpan.className = 'installed-app-title top-1/2 ';

        // Add [private] tag if this is a private space
        const isPrivate = app.extra && app.extra.private === true;
        if (isPrivate) {
            titleSpan.innerHTML = app.name + ' <span style="color: #dc2626; font-size: 0.75rem; font-weight: 600; margin-left: 0.25rem;">[private]</span>';
        } else {
            titleSpan.innerHTML = app.name;
        }

        title.appendChild(titleSpan);
        if (app.extra && app.extra.custom_app_url) {
            const settingsLink = document.createElement('a');
            settingsLink.className = 'installed-app-settings ml-2 text-gray-500 cursor-pointer';
            settingsLink.innerHTML = '⚙️';

            const url = new URL(app.extra.custom_app_url);
            url.hostname = window.location.host.split(':')[0];

            settingsLink.href = url.toString();
            settingsLink.target = '_blank';
            settingsLink.rel = 'noopener noreferrer';
            title.appendChild(settingsLink);
        }
        container.appendChild(title);
        const slider = document.createElement('div');
        const toggle = new ToggleSlider({
            checked: isRunning,
            onChange: (checked) => {
                if (installedApps.busy) {
                    toggle.setChecked(!checked);
                    return;
                }
                if (checked) {
                    installedApps.startApp(app.name);
                } else {
                    installedApps.stopApp(app.name);
                }
            }
        });
        installedApps.toggles[app.name] = toggle;
        slider.appendChild(toggle.element);
        container.appendChild(slider);

        const remove = document.createElement('button');
        remove.innerHTML = '🗑️';
        remove.className = '-translate-y-1 text-xl';
        container.appendChild(remove);
        remove.onclick = async () => {
            console.log(`Removing ${app.name}...`);
            const resp = await fetch(`/api/apps/remove/${app.name}`, { method: 'POST' });
            const data = await resp.json();
            const jobId = data.job_id;

            installedApps.appUninstallLogHandler(app.name, jobId);
        };

        return container;
    },

    getRunningApp: async () => {
        const resp = await fetch('/api/apps/current-app-status');
        const data = await resp.json();
        if (!data) {
            return null;
        }
        installedApps.currentlyRunningApp = data.info.name;
        return data.info.name;
    },

    appUninstallLogHandler: async (appName, jobId) => {
        const installModal = document.getElementById('install-modal');
        const modalTitle = installModal.querySelector('#modal-title');
        modalTitle.textContent = `Uninstalling ${appName}...`;
        installModal.classList.remove('hidden');

        const logsDiv = document.getElementById('install-logs');
        logsDiv.textContent = '';

        const closeButton = document.getElementById('modal-close-button');
        closeButton.onclick = () => {
            installModal.classList.add('hidden');
        };
        closeButton.classList = "hidden";
        closeButton.textContent = '';

        const ws = new WebSocket(`ws://${location.host}/api/apps/ws/apps-manager/${jobId}`);
        ws.onmessage = (event) => {
            try {
                if (event.data.startsWith('{') && event.data.endsWith('}')) {

                    const data = JSON.parse(event.data);

                    if (data.status === "failed") {
                        closeButton.classList = "text-white bg-red-700 hover:bg-red-800 focus:ring-4 focus:outline-none focus:ring-red-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center dark:bg-red-600 dark:hover:bg-red-700 dark:focus:ring-red-800";
                        closeButton.textContent = 'Close';
                        console.error(`Uninstallation of ${appName} failed.`);
                    } else if (data.status === "done") {
                        closeButton.classList = "text-white bg-green-700 hover:bg-green-800 focus:ring-4 focus:outline-none focus:ring-green-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center dark:bg-green-600 dark:hover:bg-green-700 dark:focus:ring-green-800";
                        closeButton.textContent = 'Uninstall done';
                        console.log(`Uninstallation of ${appName} completed.`);

                    }
                } else {
                    logsDiv.innerHTML += event.data + '\n';
                    logsDiv.scrollTop = logsDiv.scrollHeight;
                }
            } catch {
                logsDiv.innerHTML += event.data + '\n';
                logsDiv.scrollTop = logsDiv.scrollHeight;
            }
        };
        ws.onclose = async () => {
            hfAppsStore.refreshAppList();
            installedApps.refreshAppList();
        };
    },
};

class ToggleSlider {
    constructor({ checked = false, onChange = null } = {}) {
        this.label = document.createElement('label');
        this.label.className = 'relative inline-block w-28 h-8 cursor-pointer';

        this.input = document.createElement('input');
        this.input.type = 'checkbox';
        this.input.className = 'sr-only peer';
        this.input.checked = checked;
        this.label.appendChild(this.input);

        // Off label
        this.offLabel = document.createElement('span');
        this.offLabel.textContent = 'Off';
        this.offLabel.className = 'absolute left-0 top-1/2 -translate-x-8 -translate-y-1/2 text-base select-none transition-colors duration-200 text-gray-900 peer-checked:text-gray-400';
        this.label.appendChild(this.offLabel);

        this.track = document.createElement('div');
        this.track.className = 'absolute top-0 left-0 w-16 h-8 bg-gray-200 rounded-full transition-colors duration-200 peer-checked:bg-blue-800 dark:bg-gray-400 dark:peer-checked:bg-blue-800';
        this.label.appendChild(this.track);

        this.thumb = document.createElement('div');
        this.thumb.className = 'absolute top-0.5 left-0.5 w-7 h-7 bg-white border border-gray-300 rounded-full transition-all duration-200';
        this.track.appendChild(this.thumb);

        // On label
        this.onLabel = document.createElement('span');
        this.onLabel.textContent = 'On';
        this.onLabel.className = 'absolute right-0 top-1/2 -translate-y-1/2 -translate-x-4 text-base select-none transition-colors duration-200 text-gray-400 peer-checked:text-gray-900';
        this.label.appendChild(this.onLabel);


        this.input.addEventListener('change', () => {
            if (this.input.checked) {
                this.thumb.style.transform = 'translateX(31px)';
                this.onLabel.classList.remove('text-gray-400');
                this.onLabel.classList.add('text-gray-900');
                this.offLabel.classList.remove('text-gray-900');
                this.offLabel.classList.add('text-gray-400');
            } else {
                this.thumb.style.transform = 'translateX(0)';
                this.onLabel.classList.remove('text-gray-900');
                this.onLabel.classList.add('text-gray-400');
                this.offLabel.classList.remove('text-gray-400');
                this.offLabel.classList.add('text-gray-900');
            }
            if (onChange) onChange(this.input.checked);
        });

        // Set initial thumb and label color
        if (checked) {
            this.thumb.style.transform = 'translateX(31px)';
            this.onLabel.classList.remove('text-gray-400');
            this.onLabel.classList.add('text-gray-900');
        } else {
            this.onLabel.classList.remove('text-gray-900');
            this.onLabel.classList.add('text-gray-400');
        }

        this.element = this.label;
    }

    setChecked(val) {
        this.input.checked = val;
        if (this.input.checked) {
            this.thumb.style.transform = 'translateX(48px)';
            this.onLabel.classList.remove('text-gray-400');
            this.onLabel.classList.add('text-gray-900');
            this.offLabel.classList.remove('text-gray-900');
            this.offLabel.classList.add('text-gray-400');
        } else {
            this.thumb.style.transform = 'translateX(0)';
            this.onLabel.classList.remove('text-gray-900');
            this.onLabel.classList.add('text-gray-400');
            this.offLabel.classList.remove('text-gray-400');
            this.offLabel.classList.add('text-gray-900');
        }
    }

    getChecked() {
        return this.input.checked;
    }

    disable() {
        this.input.disabled = true;
        this.label.classList.add('opacity-50', 'pointer-events-none');
    }

    enable() {
        this.input.disabled = false;
        this.label.classList.remove('opacity-50', 'pointer-events-none');
    }
};

window.addEventListener('load', async () => {
    await installedApps.refreshAppList();
});