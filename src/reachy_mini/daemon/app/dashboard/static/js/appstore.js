
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
        isOAuthConfigured: false,
        oauthSessionId: null,

        init: async () => {
            // Initialize advanced section for all versions (wireless and Lite)
            try {
                // Show the advanced section
                document.getElementById('hf-advanced-section').classList.remove('hidden');

                // Check if OAuth is configured
                try {
                    const oauthConfig = await fetch('/api/hf-auth/oauth/configured').then(r => r.json());
                    hfAppsStore.advanced.isOAuthConfigured = oauthConfig.configured;

                    // Show/hide OAuth button based on configuration
                    const oauthSection = document.getElementById('hf-oauth-login');
                    const manualSection = document.getElementById('hf-manual-token-section');

                    if (hfAppsStore.advanced.isOAuthConfigured) {
                        // OAuth configured - show OAuth button, collapse manual
                        oauthSection.classList.remove('hidden');
                        manualSection.open = false;
                    } else {
                        // No OAuth - hide OAuth button, expand manual
                        oauthSection.classList.add('hidden');
                        manualSection.open = true;
                    }
                } catch {
                    // OAuth check failed, just show manual
                    document.getElementById('hf-manual-token-section').open = true;
                }

                // Set up event listeners
                document.getElementById('hf-advanced-toggle').onclick = hfAppsStore.advanced.toggleSection;
                document.getElementById('hf-login-button').onclick = hfAppsStore.advanced.login;
                document.getElementById('hf-logout-button').onclick = hfAppsStore.advanced.logout;
                document.getElementById('hf-install-private-button').onclick = hfAppsStore.advanced.installPrivateSpace;

                // Chevron rotation for manual section
                const manualSection = document.getElementById('hf-manual-token-section');
                if (manualSection) {
                    manualSection.addEventListener('toggle', () => {
                        const chevron = document.getElementById('hf-manual-chevron');
                        if (chevron) {
                            chevron.style.transform = manualSection.open ? 'rotate(90deg)' : 'rotate(0deg)';
                        }
                    });
                }

                // Token input listeners for auto-enable button
                const tokenInput = document.getElementById('hf-token-input');
                if (tokenInput) {
                    // Listen for input changes (typing, pasting)
                    tokenInput.addEventListener('input', hfAppsStore.advanced.onTokenInput);

                    // Listen for paste event
                    tokenInput.addEventListener('paste', (e) => {
                        // Small delay to let paste complete
                        setTimeout(hfAppsStore.advanced.onTokenInput, 50);
                    });

                    // Enter key to submit
                    tokenInput.addEventListener('keypress', (e) => {
                        if (e.key === 'Enter') {
                            const btn = document.getElementById('hf-login-button');
                            if (!btn.disabled) {
                                hfAppsStore.advanced.login();
                            }
                        }
                    });
                }

                // Add Enter key support for space ID input
                document.getElementById('hf-space-id-input').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        hfAppsStore.advanced.installPrivateSpace();
                    }
                });

                // Check authentication status
                await hfAppsStore.advanced.checkAuthStatus();
            } catch (error) {
                console.error('Error initializing advanced section:', error);
            }
        },

        startOAuthLogin: async () => {
            const button = document.getElementById('hf-oauth-button');
            const statusEl = document.getElementById('hf-oauth-status');

            // Show loading state
            button.disabled = true;
            button.innerHTML = '⏳ Redirecting...';
            statusEl.textContent = 'Opening Hugging Face login...';
            statusEl.classList.remove('hidden');

            try {
                // Get the OAuth URL from the backend
                const response = await fetch('/api/hf-auth/oauth/start');
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to start OAuth');
                }

                const data = await response.json();

                // Store session ID for polling
                hfAppsStore.advanced.oauthSessionId = data.session_id;

                // Open OAuth in new window/tab
                window.open(data.auth_url, '_blank');

                // Update status
                statusEl.innerHTML = 'Complete login in the new window. This page will update automatically.';
                button.innerHTML = '🤗 Login with Hugging Face';
                button.disabled = false;

                // Poll for completion
                hfAppsStore.advanced.pollForAuth();

            } catch (error) {
                statusEl.textContent = 'Error: ' + error.message;
                statusEl.style.color = '#dc2626';
                button.innerHTML = '🤗 Login with Hugging Face';
                button.disabled = false;
            }
        },

        pollForAuth: () => {
            // Poll every 2 seconds to check if user completed auth
            const pollInterval = setInterval(async () => {
                try {
                    const response = await fetch('/api/hf-auth/status');
                    const data = await response.json();

                    if (data.is_logged_in) {
                        // User logged in! Update UI
                        clearInterval(pollInterval);
                        hfAppsStore.advanced.isAuthenticated = true;
                        hfAppsStore.advanced.username = data.username;
                        hfAppsStore.advanced.updateAuthUI();
                    }
                } catch {
                    // Ignore polling errors
                }
            }, 2000);

            // Stop polling after 5 minutes
            setTimeout(() => clearInterval(pollInterval), 300000);
        },

        onTokenInput: () => {
            const tokenInput = document.getElementById('hf-token-input');
            const loginButton = document.getElementById('hf-login-button');
            const connectHint = document.getElementById('hf-connect-hint');
            const tokenHint = document.getElementById('hf-token-hint');

            const token = tokenInput.value.trim();
            const isValidFormat = token.startsWith('hf_') && token.length > 10;

            if (isValidFormat) {
                // Enable the button with nice styling
                loginButton.disabled = false;
                loginButton.style.backgroundColor = '#10b981';
                loginButton.style.color = 'white';
                loginButton.style.cursor = 'pointer';
                loginButton.classList.add('hover:bg-green-600');
                connectHint.textContent = 'Click to connect!';
                connectHint.style.color = '#10b981';
                tokenHint.innerHTML = '✓ Token looks good!';
                tokenHint.style.color = '#10b981';
                tokenInput.style.borderColor = '#10b981';
            } else if (token.length > 0) {
                // Something entered but not valid format
                loginButton.disabled = true;
                loginButton.style.backgroundColor = '#d1d5db';
                loginButton.style.color = '#6b7280';
                loginButton.style.cursor = 'not-allowed';
                loginButton.classList.remove('hover:bg-green-600');
                tokenHint.innerHTML = 'Token should start with <code style="background: #e5e7eb; padding: 2px 6px; border-radius: 4px;">hf_</code>';
                tokenHint.style.color = '#f59e0b';
                tokenInput.style.borderColor = '#f59e0b';
                connectHint.textContent = 'Check your token format';
                connectHint.style.color = '#f59e0b';
            } else {
                // Empty
                loginButton.disabled = true;
                loginButton.style.backgroundColor = '#d1d5db';
                loginButton.style.color = '#6b7280';
                loginButton.style.cursor = 'not-allowed';
                loginButton.classList.remove('hover:bg-green-600');
                tokenHint.innerHTML = 'The token starts with <code style="background: #e5e7eb; padding: 2px 6px; border-radius: 4px;">hf_</code>';
                tokenHint.style.color = '#6b7280';
                tokenInput.style.borderColor = '#3b82f6';
                connectHint.textContent = 'Paste your token above to enable this button';
                connectHint.style.color = '#9ca3af';
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

        checkAuthStatus: async () => {
            try {
                const response = await fetch('/api/hf-auth/status');
                const data = await response.json();

                hfAppsStore.advanced.isAuthenticated = data.is_logged_in;
                hfAppsStore.advanced.username = data.username;

                hfAppsStore.advanced.updateAuthUI();
            } catch (error) {
                console.error('Error checking auth status:', error);
            }
        },

        updateAuthUI: () => {
            const indicator = document.getElementById('hf-auth-indicator');
            const authText = document.getElementById('hf-auth-text');
            const loginForm = document.getElementById('hf-login-form');
            const loggedInView = document.getElementById('hf-logged-in-view');
            const usernameSpan = document.getElementById('hf-username');

            if (hfAppsStore.advanced.isAuthenticated) {
                // Connected state
                indicator.classList.remove('bg-gray-400');
                indicator.classList.add('bg-green-500');
                authText.textContent = '🤗 Connected';
                authText.style.color = '#065f46';
                loginForm.classList.add('hidden');
                loggedInView.classList.remove('hidden');
                usernameSpan.textContent = hfAppsStore.advanced.username || 'Unknown';
            } else {
                // Not connected state
                indicator.classList.remove('bg-green-500');
                indicator.classList.add('bg-gray-400');
                authText.textContent = 'Not connected';
                authText.style.color = '#374151';
                loginForm.classList.remove('hidden');
                loggedInView.classList.add('hidden');

                // Reset token input state
                const tokenInput = document.getElementById('hf-token-input');
                if (tokenInput) {
                    tokenInput.value = '';
                    hfAppsStore.advanced.onTokenInput();
                }
            }
        },

        login: async () => {
            const tokenInput = document.getElementById('hf-token-input');
            const errorDiv = document.getElementById('hf-login-error');
            const loginButton = document.getElementById('hf-login-button');
            const step1 = document.getElementById('hf-step1');
            const step2 = document.getElementById('hf-step2');
            const connectHint = document.getElementById('hf-connect-hint');

            const token = tokenInput.value.trim();

            if (!token) {
                errorDiv.textContent = 'Please paste your token first';
                errorDiv.classList.remove('hidden');
                return;
            }

            // Disable button and show loading state
            loginButton.disabled = true;
            loginButton.innerHTML = '<span class="animate-pulse">⏳ Connecting...</span>';
            loginButton.style.backgroundColor = '#6b7280';
            errorDiv.classList.add('hidden');
            connectHint.textContent = 'Verifying your token...';

            try {
                const response = await fetch('/api/hf-auth/save-token', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ token })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Login failed');
                }

                const data = await response.json();

                // Success! Show celebration
                loginButton.innerHTML = '🎉 Connected!';
                loginButton.style.backgroundColor = '#10b981';
                connectHint.textContent = `Welcome, ${data.username || 'friend'}!`;
                connectHint.style.color = '#10b981';

                // Fade out steps
                step1.style.opacity = '0.5';
                step2.style.opacity = '0.5';

                // Clear token input
                tokenInput.value = '';

                // Update state after short delay for visual feedback
                setTimeout(() => {
                    hfAppsStore.advanced.isAuthenticated = true;
                    hfAppsStore.advanced.username = data.username;
                    hfAppsStore.advanced.updateAuthUI();
                }, 1000);

            } catch (error) {
                errorDiv.innerHTML = `<strong>Oops!</strong> ${error.message}<br><small>Make sure you copied the entire token.</small>`;
                errorDiv.classList.remove('hidden');

                // Reset button
                loginButton.disabled = false;
                loginButton.innerHTML = '✓ Connect to Hugging Face';
                hfAppsStore.advanced.onTokenInput(); // Reset button state based on input
            }
        },

        logout: async () => {
            const logoutBtn = document.getElementById('hf-logout-button');
            logoutBtn.textContent = 'Disconnecting...';
            logoutBtn.disabled = true;

            try {
                await fetch('/api/hf-auth/token', { method: 'DELETE' });

                hfAppsStore.advanced.isAuthenticated = false;
                hfAppsStore.advanced.username = null;
                hfAppsStore.advanced.updateAuthUI();

            } catch (error) {
                console.error('Error logging out:', error);
                logoutBtn.textContent = 'Disconnect';
                logoutBtn.disabled = false;
            }
        },

        installPrivateSpace: async () => {
            const spaceIdInput = document.getElementById('hf-space-id-input');
            const errorDiv = document.getElementById('hf-private-install-error');
            const installButton = document.getElementById('hf-install-private-button');

            const spaceId = spaceIdInput.value.trim();

            if (!spaceId) {
                errorDiv.textContent = 'Please enter a space ID';
                errorDiv.classList.remove('hidden');
                return;
            }

            // Validate format (should be "username/space-name")
            if (!spaceId.includes('/')) {
                errorDiv.textContent = 'Space ID should be in format: username/space-name';
                errorDiv.classList.remove('hidden');
                return;
            }

            // Disable button during request
            installButton.disabled = true;
            installButton.textContent = 'Installing...';
            errorDiv.classList.add('hidden');

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
                spaceIdInput.value = '';

                // Show installation modal (reuse existing modal)
                const spaceName = spaceId.split('/')[1];
                hfAppsStore.appInstallLogHandler(spaceName, jobId);

            } catch (error) {
                if (error.message.includes('authenticate') || error.message.includes('401')) {
                    // Token expired or invalid - show login form
                    errorDiv.textContent = 'Authentication expired. Please login again.';
                    hfAppsStore.advanced.isAuthenticated = false;
                    hfAppsStore.advanced.username = null;
                    hfAppsStore.advanced.updateAuthUI();
                } else {
                    errorDiv.textContent = error.message;
                }
                errorDiv.classList.remove('hidden');
            } finally {
                installButton.disabled = false;
                installButton.textContent = 'Install Private Space';
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
