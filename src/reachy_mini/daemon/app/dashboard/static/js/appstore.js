
const hfAppsStore = {
    refreshAppList: async () => {
        const appsData = await hfAppsStore.fetchAvailableApps();
        await hfAppsStore.displayAvailableApps(appsData);
    },
    fetchAvailableApps: async () => {
        const resAvailable = await fetch('/api/apps/list-available');
        const appsData = await resAvailable.json();
        return appsData;
    },

    isInstalling: false,

    installApp: async (app) => {
        if (hfAppsStore.isInstalling) {
            console.warn('An installation is already in progress.');
            return;
        }
        hfAppsStore.isInstalling = true;

        const appName = app.extra.cardData.title || app.name;
        console.log(`Installing ${app.name}...`);

        const resp = await fetch('/api/apps/install', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(app)
        });
        const data = await resp.json();
        const jobId = data.job_id;

        hfAppsStore.appInstallLogHandler(appName, jobId);
    },

    displayAvailableApps: async (appsData) => {
        const appsListElement = document.getElementById('hf-available-apps');
        appsListElement.innerHTML = '';

        if (!appsData || appsData.length === 0) {
            appsListElement.innerHTML = '<li>No available apps found.</li>';
            return;
        }

        const hfApps = appsData.filter(app => app.source_kind === 'hf_space');
        const installedApps = appsData.filter(app => app.source_kind === 'installed');

        hfApps.forEach(app => {
            const li = document.createElement('li');
            li.className = 'app-list-item';
            const isInstalled = installedApps.some(installedApp => installedApp.name === app.name);
            li.appendChild(hfAppsStore.createAppElement(app, isInstalled));
            appsListElement.appendChild(li);
        });
    },

    createAppElement: (app, isInstalled) => {
        const container = document.createElement('div');
        container.className = 'grid grid-cols-[2rem_auto_8rem] justify-stretch gap-x-6';

        const iconDiv = document.createElement('div');
        iconDiv.className = 'hf-app-icon row-span-2 my-1';
        iconDiv.textContent = app.extra.cardData.emoji || '📦';
        container.appendChild(iconDiv);

        const nameDiv = document.createElement('div');
        nameDiv.className = 'flex flex-col';

        const titleDiv = document.createElement('a');
        titleDiv.href = app.url;
        titleDiv.target = '_blank';
        titleDiv.rel = 'noopener noreferrer';
        titleDiv.className = 'hf-app-title';
        titleDiv.textContent = app.extra.cardData.title || app.name;
        nameDiv.appendChild(titleDiv);
        const descriptionDiv = document.createElement('span');
        descriptionDiv.className = 'hf-app-description';
        descriptionDiv.textContent = app.description || 'No description available.';
        nameDiv.appendChild(descriptionDiv);
        container.appendChild(nameDiv);

        const installButtonDiv = document.createElement('button');
        installButtonDiv.className = 'row-span-2 my-2 hf-app-install-button';

        if (isInstalled) {
            installButtonDiv.classList.add('bg-gray-400', 'cursor-not-allowed');
            installButtonDiv.textContent = 'Installed';
            installButtonDiv.disabled = true;
        } else {
            installButtonDiv.classList.add('border', 'border-red-600');
            installButtonDiv.textContent = 'Install';
            installButtonDiv.onclick = async () => {
                hfAppsStore.installApp(app);
            };
        }

        container.appendChild(installButtonDiv);

        return container;
    },

    appInstallLogHandler: async (appName, jobId) => {
        const installModal = document.getElementById('install-modal');
        const modalTitle = installModal.querySelector('#modal-title');
        modalTitle.textContent = `Installing ${appName}...`;
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
                const data = JSON.parse(event.data);
                console.log(data.status);

                if (data.status === "failed") {
                    closeButton.classList = "text-white bg-red-700 hover:bg-red-800 focus:ring-4 focus:outline-none focus:ring-red-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center dark:bg-red-600 dark:hover:bg-red-700 dark:focus:ring-red-800";
                    closeButton.textContent = 'Close';
                    console.error(`Installation of ${appName} failed.`);
                } else if (data.status === "done") {
                    closeButton.classList = "text-white bg-green-700 hover:bg-green-800 focus:ring-4 focus:outline-none focus:ring-green-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center dark:bg-green-600 dark:hover:bg-green-700 dark:focus:ring-green-800";
                    closeButton.textContent = 'Install done';
                    console.log(`Installation of ${appName} completed.`);

                }
            } catch {
                logsDiv.innerHTML += event.data + '\n';
                logsDiv.scrollTop = logsDiv.scrollHeight;
            }
        };
        ws.onclose = async () => {
            hfAppsStore.isInstalling = false;
            hfAppsStore.refreshAppList();
            installedApps.refreshAppList();
        };
    },
};


window.addEventListener('load', async () => {
    await hfAppsStore.refreshAppList();
});