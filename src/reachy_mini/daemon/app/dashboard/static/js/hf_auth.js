/**
 * Global HuggingFace authentication handler.
 * Provides login/logout from the dashboard header.
 */
const hfAuth = {
    isAuthenticated: false,
    username: null,
    isOAuthConfigured: false,
    oauthSessionId: null,
    relayState: null,
    relayMessage: null,
    relayPollInterval: null,

    init: async () => {
        // Check OAuth configuration
        try {
            const response = await fetch('/api/hf-auth/oauth/configured');
            const data = await response.json();
            hfAuth.isOAuthConfigured = data.configured;
        } catch {
            hfAuth.isOAuthConfigured = false;
        }

        // Check current auth status
        await hfAuth.checkAuthStatus();

        // Setup token input listener
        const tokenInput = document.getElementById('hf-modal-token-input');
        if (tokenInput) {
            tokenInput.addEventListener('input', hfAuth.onTokenInput);
            tokenInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    const btn = document.getElementById('hf-modal-token-btn');
                    if (!btn.disabled) {
                        hfAuth.loginWithToken();
                    }
                }
            });
        }

        // Start relay status polling
        hfAuth.startRelayStatusPolling();
    },

    checkAuthStatus: async () => {
        try {
            const response = await fetch('/api/hf-auth/status');
            const data = await response.json();

            hfAuth.isAuthenticated = data.is_logged_in;
            hfAuth.username = data.username;

            hfAuth.updateHeaderUI();
        } catch (error) {
            console.error('Error checking HF auth status:', error);
        }
    },

    updateHeaderUI: () => {
        const authBtn = document.getElementById('hf-header-auth-btn');
        const userBadge = document.getElementById('hf-header-user-badge');
        const usernameSpan = document.getElementById('hf-header-username');

        if (hfAuth.isAuthenticated) {
            // Show badge, change button to Logout
            if (userBadge) userBadge.classList.remove('hidden');
            if (usernameSpan) usernameSpan.textContent = hfAuth.username || 'Connected';
            if (authBtn) {
                authBtn.textContent = 'Logout';
                authBtn.style.backgroundColor = '#fee2e2';
                authBtn.style.color = '#dc2626';
                authBtn.style.borderColor = '#fecaca';
                authBtn.onmouseover = () => { authBtn.style.backgroundColor = '#fecaca'; };
                authBtn.onmouseout = () => { authBtn.style.backgroundColor = '#fee2e2'; };
            }
        } else {
            // Hide badge, change button to Login
            if (userBadge) userBadge.classList.add('hidden');
            if (authBtn) {
                authBtn.textContent = '🤗 Login';
                authBtn.style.backgroundColor = '#fef3c7';
                authBtn.style.color = '#92400e';
                authBtn.style.borderColor = '#fcd34d';
                authBtn.onmouseover = () => { authBtn.style.backgroundColor = '#fde68a'; };
                authBtn.onmouseout = () => { authBtn.style.backgroundColor = '#fef3c7'; };
            }
        }

        // Also update appstore section if it exists
        if (typeof hfAppsStore !== 'undefined' && hfAppsStore.advanced) {
            hfAppsStore.advanced.isAuthenticated = hfAuth.isAuthenticated;
            hfAppsStore.advanced.username = hfAuth.username;
            hfAppsStore.advanced.updateAuthUI();
        }

        // Update relay status visibility
        hfAuth.updateRelayUI();

        // Trigger immediate relay status check on auth change
        if (hfAuth.isAuthenticated) {
            hfAuth.checkRelayStatus();
        }
    },

    openLoginModal: () => {
        const modal = document.getElementById('hf-login-modal');
        const oauthSection = document.getElementById('hf-modal-oauth');

        if (modal) {
            modal.classList.remove('hidden');
            // Trigger reflow for transition
            modal.offsetHeight;
            modal.classList.add('visible');

            // Show/hide OAuth based on configuration
            if (oauthSection) {
                if (hfAuth.isOAuthConfigured) {
                    oauthSection.classList.remove('hidden');
                } else {
                    oauthSection.classList.add('hidden');
                }
            }
        }
    },

    closeLoginModal: () => {
        const modal = document.getElementById('hf-login-modal');
        if (modal) {
            modal.classList.remove('visible');
            // Wait for transition to complete before hiding
            setTimeout(() => {
                modal.classList.add('hidden');
            }, 250);
        }

        // Reset state
        const tokenInput = document.getElementById('hf-modal-token-input');
        const errorDiv = document.getElementById('hf-modal-error');
        const statusEl = document.getElementById('hf-modal-oauth-status');

        if (tokenInput) tokenInput.value = '';
        if (errorDiv) errorDiv.classList.add('hidden');
        if (statusEl) statusEl.classList.add('hidden');

        hfAuth.onTokenInput();
    },

    onTokenInput: () => {
        const tokenInput = document.getElementById('hf-modal-token-input');
        const tokenBtn = document.getElementById('hf-modal-token-btn');
        const tokenHint = document.getElementById('hf-modal-token-hint');

        if (!tokenInput || !tokenBtn) return;

        const token = tokenInput.value.trim();
        const isValidFormat = token.startsWith('hf_') && token.length > 10;

        if (isValidFormat) {
            tokenBtn.disabled = false;
            tokenBtn.classList.remove('bg-gray-200', 'text-gray-400', 'cursor-not-allowed');
            tokenBtn.classList.add('bg-green-500', 'hover:bg-green-600', 'text-white', 'cursor-pointer');
            if (tokenHint) {
                tokenHint.textContent = 'Token looks good!';
                tokenHint.classList.remove('text-gray-400');
                tokenHint.classList.add('text-green-600');
            }
        } else {
            tokenBtn.disabled = true;
            tokenBtn.classList.add('bg-gray-200', 'text-gray-400', 'cursor-not-allowed');
            tokenBtn.classList.remove('bg-green-500', 'hover:bg-green-600', 'text-white', 'cursor-pointer');
            if (tokenHint) {
                tokenHint.textContent = 'Token starts with hf_';
                tokenHint.classList.add('text-gray-400');
                tokenHint.classList.remove('text-green-600');
            }
        }
    },

    startOAuthLogin: async () => {
        const button = document.getElementById('hf-modal-oauth-btn');
        const statusEl = document.getElementById('hf-modal-oauth-status');
        const errorDiv = document.getElementById('hf-modal-error');

        if (button) {
            button.disabled = true;
            button.innerHTML = 'Redirecting...';
        }
        if (statusEl) {
            statusEl.textContent = 'Opening Hugging Face login...';
            statusEl.classList.remove('hidden');
        }
        if (errorDiv) {
            errorDiv.classList.add('hidden');
        }

        try {
            const response = await fetch('/api/hf-auth/oauth/start');
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to start OAuth');
            }

            const data = await response.json();
            hfAuth.oauthSessionId = data.session_id;

            // Open OAuth in new window
            window.open(data.auth_url, '_blank');

            if (statusEl) {
                statusEl.innerHTML = 'Complete login in the new window...';
            }
            if (button) {
                button.innerHTML = '🤗 Login with Hugging Face';
                button.disabled = false;
            }

            // Poll for completion
            hfAuth.pollForAuth();

        } catch (error) {
            if (errorDiv) {
                errorDiv.textContent = error.message;
                errorDiv.classList.remove('hidden');
            }
            if (button) {
                button.innerHTML = '🤗 Login with Hugging Face';
                button.disabled = false;
            }
            if (statusEl) {
                statusEl.classList.add('hidden');
            }
        }
    },

    pollForAuth: () => {
        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/hf-auth/status');
                const data = await response.json();

                if (data.is_logged_in) {
                    clearInterval(pollInterval);
                    hfAuth.isAuthenticated = true;
                    hfAuth.username = data.username;
                    hfAuth.updateHeaderUI();
                    hfAuth.closeLoginModal();
                }
            } catch {
                // Ignore polling errors
            }
        }, 2000);

        // Stop polling after 5 minutes
        setTimeout(() => clearInterval(pollInterval), 300000);
    },

    loginWithToken: async () => {
        const tokenInput = document.getElementById('hf-modal-token-input');
        const tokenBtn = document.getElementById('hf-modal-token-btn');
        const errorDiv = document.getElementById('hf-modal-error');

        const token = tokenInput?.value.trim();
        if (!token) return;

        if (tokenBtn) {
            tokenBtn.disabled = true;
            tokenBtn.innerHTML = 'Connecting...';
        }
        if (errorDiv) {
            errorDiv.classList.add('hidden');
        }

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

            hfAuth.isAuthenticated = true;
            hfAuth.username = data.username;
            hfAuth.updateHeaderUI();
            hfAuth.closeLoginModal();

        } catch (error) {
            if (errorDiv) {
                errorDiv.textContent = error.message;
                errorDiv.classList.remove('hidden');
            }
            if (tokenBtn) {
                tokenBtn.disabled = false;
                tokenBtn.innerHTML = 'Connect';
                hfAuth.onTokenInput();
            }
        }
    },

    logout: async () => {
        const authBtn = document.getElementById('hf-header-auth-btn');
        if (authBtn) {
            authBtn.textContent = '...';
            authBtn.disabled = true;
        }

        try {
            await fetch('/api/hf-auth/token', { method: 'DELETE' });

            hfAuth.isAuthenticated = false;
            hfAuth.username = null;
            hfAuth.updateHeaderUI();

        } catch (error) {
            console.error('Error logging out:', error);
        } finally {
            if (authBtn) {
                authBtn.disabled = false;
            }
        }
    },

    // Central relay status methods
    startRelayStatusPolling: () => {
        // Initial check
        hfAuth.checkRelayStatus();
        // Poll every 5 seconds
        hfAuth.relayPollInterval = setInterval(hfAuth.checkRelayStatus, 5000);
    },

    checkRelayStatus: async () => {
        try {
            const response = await fetch('/api/hf-auth/relay-status');
            const data = await response.json();
            hfAuth.relayState = data.state;
            hfAuth.relayMessage = data.message;
            hfAuth.updateRelayUI();
        } catch {
            hfAuth.relayState = 'error';
            hfAuth.relayMessage = 'Cannot fetch status';
            hfAuth.updateRelayUI();
        }
    },

    updateRelayUI: () => {
        const statusDiv = document.getElementById('hf-relay-status');
        const indicator = document.getElementById('hf-relay-indicator');
        const text = document.getElementById('hf-relay-text');

        if (!statusDiv || !indicator || !text) return;

        // Show for Lite (unavailable) always, otherwise only when authenticated
        const isLite = hfAuth.relayState === 'unavailable';
        if (!isLite && !hfAuth.isAuthenticated) {
            statusDiv.classList.add('hidden');
            return;
        }

        statusDiv.classList.remove('hidden');

        // State-based styling - user-friendly labels
        const states = {
            'connected': {
                color: '#10b981', bg: '#d1fae5', border: '#86efac',
                textColor: '#065f46', label: 'Ready',
                tooltip: 'HF Space Apps: Ready'
            },
            'connecting': {
                color: '#f59e0b', bg: '#fef3c7', border: '#fcd34d',
                textColor: '#92400e', label: 'Connecting...',
                tooltip: 'HF Space Apps: Connecting...'
            },
            'reconnecting': {
                color: '#f59e0b', bg: '#fef3c7', border: '#fcd34d',
                textColor: '#92400e', label: 'Connecting...',
                tooltip: 'HF Space Apps: Connecting...'
            },
            'waiting_for_token': {
                color: '#6b7280', bg: '#f3f4f6', border: '#d1d5db',
                textColor: '#6b7280', label: 'Offline',
                tooltip: 'HF Space Apps: Login required'
            },
            'error': {
                color: '#ef4444', bg: '#fee2e2', border: '#fecaca',
                textColor: '#dc2626', label: 'Offline',
                tooltip: 'HF Space Apps: Connection error'
            },
            'stopped': {
                color: '#6b7280', bg: '#f3f4f6', border: '#d1d5db',
                textColor: '#6b7280', label: 'Offline',
                tooltip: 'HF Space Apps: Offline'
            },
            'unavailable': {
                color: '#9ca3af', bg: '#f3f4f6', border: '#e5e7eb',
                textColor: '#9ca3af', label: 'Lite',
                tooltip: 'HF Space Apps: Coming soon to Lite version'
            }
        };

        const style = states[hfAuth.relayState] || states['stopped'];

        indicator.style.backgroundColor = style.color;
        statusDiv.style.backgroundColor = style.bg;
        statusDiv.style.borderColor = style.border;
        text.style.color = style.textColor;
        text.textContent = style.label;
        statusDiv.title = style.tooltip;
    }
};

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    hfAuth.init();
});
