
const hfAppsStore = {
    refreshAppList: async () => {
        const appsData = await hfAppsStore.fetchAvailableApps();
        await hfAppsStore.displayAvailableApps(appsData);
    },
    fetchAvailableApps: async () => {
        // Decide which source to query based on the toggle state.
        const includeCommunity = document.getElementById('hf-show-community')?.checked === true;
        const source = includeCommunity ? 'hf_space' : 'dashboard_selection';
        const resAvailable = await fetch(`/api/apps/list-available/${source}`);
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
        const installedApps = await fetch('/api/apps/list-available/installed').then(res => res.json());

        hfApps.forEach(app => {
            const li = document.createElement('li');
            li.className = 'app-list-item';
            const isInstalled = installedApps.some(installedApp => {
                // Match by HuggingFace space ID (extra.id) - most reliable
                if (installedApp.extra?.id && app.extra?.id) {
                    if (installedApp.extra.id === app.extra.id) {
                        return true;
                    }
                }
                // Fallback: direct name match
                return installedApp.name === app.name;
            });
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
                if (event.data.startsWith('{') && event.data.endsWith('}')) {
                    const data = JSON.parse(event.data);

                    if (data.status === "failed") {
                        closeButton.classList = "text-white bg-red-700 hover:bg-red-800 focus:ring-4 focus:outline-none focus:ring-red-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center dark:bg-red-600 dark:hover:bg-red-700 dark:focus:ring-red-800";
                        closeButton.textContent = 'Close';
                        console.error(`Installation of ${appName} failed.`);
                    } else if (data.status === "done") {
                        closeButton.classList = "text-white bg-green-700 hover:bg-green-800 focus:ring-4 focus:outline-none focus:ring-green-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center dark:bg-green-600 dark:hover:bg-green-700 dark:focus:ring-green-800";
                        closeButton.textContent = 'Install done';
                        console.log(`Installation of ${appName} completed.`);

                    }
                }
                else {
                    logsDiv.innerHTML += event.data + '\n';
                    logsDiv.scrollTop = logsDiv.scrollHeight;
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

    // Advanced functionality for private spaces
    advanced: {
        isAuthenticated: false,
        username: null,

        init: async () => {
            // Initialize advanced section for all versions (wireless and Lite)
            try {
                // Show the advanced section
                document.getElementById('hf-advanced-section').classList.remove('hidden');

                // Set up event listeners
                document.getElementById('hf-advanced-toggle').onclick = hfAppsStore.advanced.toggleSection;

                const installBtn = document.getElementById('hf-install-private-button');
                if (installBtn) {
                    installBtn.onclick = hfAppsStore.advanced.installPrivateSpace;
                }

                // Add Enter key support for space ID input
                const spaceIdInput = document.getElementById('hf-space-id-input');
                if (spaceIdInput) {
                    spaceIdInput.addEventListener('keypress', (e) => {
                        if (e.key === 'Enter') {
                            hfAppsStore.advanced.installPrivateSpace();
                        }
                    });
                }

                // Sync auth status with global hfAuth (after a small delay to let hfAuth init first)
                setTimeout(() => {
                    if (typeof hfAuth !== 'undefined') {
                        hfAppsStore.advanced.isAuthenticated = hfAuth.isAuthenticated;
                        hfAppsStore.advanced.username = hfAuth.username;
                    }
                    hfAppsStore.advanced.updateAuthUI();
                }, 100);

            } catch (error) {
                console.error('Error initializing advanced section:', error);
            }
        },

        toggleSection: () => {
            const content = document.getElementById('hf-advanced-content');
            const chevron = document.getElementById('hf-advanced-chevron');

            if (content.classList.contains('hidden')) {
                content.classList.remove('hidden');
                chevron.style.transform = 'rotate(90deg)';
            } else {
                content.classList.add('hidden');
                chevron.style.transform = 'rotate(0deg)';
            }
        },

        updateAuthUI: () => {
            const indicator = document.getElementById('hf-auth-indicator');
            const authText = document.getElementById('hf-auth-text');
            const loginForm = document.getElementById('hf-login-form');
            const loggedInView = document.getElementById('hf-logged-in-view');

            if (!indicator || !authText) return;

            if (hfAppsStore.advanced.isAuthenticated) {
                // Connected state
                indicator.classList.remove('bg-gray-400');
                indicator.classList.add('bg-green-500');
                authText.textContent = `🤗 Connected as ${hfAppsStore.advanced.username || 'user'}`;
                authText.style.color = '#065f46';
                if (loginForm) loginForm.classList.add('hidden');
                if (loggedInView) loggedInView.classList.remove('hidden');
            } else {
                // Not connected state
                indicator.classList.remove('bg-green-500');
                indicator.classList.add('bg-gray-400');
                authText.textContent = 'Not connected';
                authText.style.color = '#374151';
                if (loginForm) loginForm.classList.remove('hidden');
                if (loggedInView) loggedInView.classList.add('hidden');
            }
        },

        installPrivateSpace: async () => {
            const spaceIdInput = document.getElementById('hf-space-id-input');
            const errorDiv = document.getElementById('hf-private-install-error');
            const installButton = document.getElementById('hf-install-private-button');

            const spaceId = spaceIdInput?.value.trim();

            if (!spaceId) {
                if (errorDiv) {
                    errorDiv.textContent = 'Please enter a space ID';
                    errorDiv.classList.remove('hidden');
                }
                return;
            }

            // Validate format (should be "username/space-name")
            if (!spaceId.includes('/')) {
                if (errorDiv) {
                    errorDiv.textContent = 'Space ID should be in format: username/space-name';
                    errorDiv.classList.remove('hidden');
                }
                return;
            }

            // Disable button during request
            if (installButton) {
                installButton.disabled = true;
                installButton.textContent = 'Installing...';
            }
            if (errorDiv) {
                errorDiv.classList.add('hidden');
            }

            try {
                const response = await fetch('/api/apps/install-private-space', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ space_id: spaceId })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Installation failed');
                }

                const data = await response.json();
                const jobId = data.job_id;

                // Clear input
                if (spaceIdInput) spaceIdInput.value = '';

                // Show installation modal (reuse existing modal)
                const spaceName = spaceId.split('/')[1];
                hfAppsStore.appInstallLogHandler(spaceName, jobId);

            } catch (error) {
                if (error.message.includes('authenticate') || error.message.includes('401')) {
                    // Token expired or invalid - update global auth state
                    if (typeof hfAuth !== 'undefined') {
                        hfAuth.isAuthenticated = false;
                        hfAuth.username = null;
                        hfAuth.updateHeaderUI();
                    }
                    hfAppsStore.advanced.isAuthenticated = false;
                    hfAppsStore.advanced.username = null;
                    hfAppsStore.advanced.updateAuthUI();
                    if (errorDiv) {
                        errorDiv.textContent = 'Authentication expired. Please login again from the header.';
                    }
                } else if (errorDiv) {
                    errorDiv.textContent = error.message;
                }
                if (errorDiv) errorDiv.classList.remove('hidden');
            } finally {
                if (installButton) {
                    installButton.disabled = false;
                    installButton.textContent = 'Install Private Space';
                }
            }
        },
    },
};


window.addEventListener('load', async () => {
    // Attach change listener to community toggle if present
    const communityToggle = document.getElementById('hf-show-community');
    if (communityToggle) {
        communityToggle.addEventListener('change', async () => {
            await hfAppsStore.refreshAppList();
        });
    }
    await hfAppsStore.refreshAppList();

    // Initialize advanced section for private spaces
    await hfAppsStore.advanced.init();
});
