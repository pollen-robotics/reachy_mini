/**
 * Global HuggingFace authentication handler.
 * Provides login/logout from the dashboard header.
 */
const hfAuth = {
    isAuthenticated: false,
    username: null,
    isOAuthConfigured: false,
    oauthSessionId: null,

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
        const loginBtn = document.getElementById('hf-header-login-btn');
        const loggedIn = document.getElementById('hf-header-logged-in');
        const usernameSpan = document.getElementById('hf-header-username');

        if (hfAuth.isAuthenticated) {
            if (loginBtn) loginBtn.classList.add('hidden');
            if (loggedIn) loggedIn.classList.remove('hidden');
            if (usernameSpan) {
                usernameSpan.textContent = hfAuth.username || 'Connected';
            }
        } else {
            if (loginBtn) loginBtn.classList.remove('hidden');
            if (loggedIn) loggedIn.classList.add('hidden');
        }

        // Also update appstore section if it exists
        if (typeof hfAppsStore !== 'undefined' && hfAppsStore.advanced) {
            hfAppsStore.advanced.isAuthenticated = hfAuth.isAuthenticated;
            hfAppsStore.advanced.username = hfAuth.username;
            hfAppsStore.advanced.updateAuthUI();
        }
    },

    openLoginModal: () => {
        const modal = document.getElementById('hf-login-modal');
        const oauthSection = document.getElementById('hf-modal-oauth');

        if (modal) {
            modal.classList.remove('hidden');

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
            modal.classList.add('hidden');
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
        const logoutBtn = document.getElementById('hf-header-logout-btn');
        if (logoutBtn) {
            logoutBtn.textContent = '...';
            logoutBtn.disabled = true;
        }

        try {
            await fetch('/api/hf-auth/token', { method: 'DELETE' });

            hfAuth.isAuthenticated = false;
            hfAuth.username = null;
            hfAuth.updateHeaderUI();

        } catch (error) {
            console.error('Error logging out:', error);
        } finally {
            if (logoutBtn) {
                logoutBtn.textContent = 'Logout';
                logoutBtn.disabled = false;
            }
        }
    }
};

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    hfAuth.init();
});
