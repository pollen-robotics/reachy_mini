/**
 * WebRTC Manual Pairing for Daemon Dashboard
 *
 * Allows connecting web apps via copy-paste pairing (no HTTP signaling needed)
 */

let pairingPeerId = null;

// Generate offer on page load
document.addEventListener('DOMContentLoaded', async () => {
    await generatePairingOffer();
});

async function generatePairingOffer() {
    try {
        pairingPeerId = 'manual-' + Math.random().toString(36).substring(7);

        // Create offer via backend API (this registers the peer with message handler)
        const response = await fetch('/api/webrtc/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ peer_id: pairingPeerId })
        });

        if (!response.ok) {
            throw new Error('Failed to create offer: ' + response.statusText);
        }

        const data = await response.json();

        // Encode offer as base64 for easy copy-paste
        const offerBase64 = btoa(JSON.stringify({
            sdp: data.sdp,
            type: data.type
        }));

        // Display in textarea
        document.getElementById('daemon-offer').value = offerBase64;
        updateStatus('Offer generated. Copy to your app.');

    } catch (error) {
        console.error('Error generating offer:', error);
        updateStatus('Error generating offer: ' + error.message, 'error');
    }
}

// Monitor connection status via polling
let statusCheckInterval = null;

function startStatusMonitoring() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }

    statusCheckInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/webrtc/status');
            const data = await response.json();

            if (data.peer_count > 0) {
                updateStatus('Connected! Data channel ready.', 'success');
                document.getElementById('webrtc-peers').classList.remove('hidden');
                document.getElementById('peer-count').textContent = data.peer_count;
            } else {
                // No peers connected yet
                if (pairingPeerId) {
                    updateStatus('Waiting for app to connect...');
                }
            }
        } catch (error) {
            console.error('Status check failed:', error);
        }
    }, 1000);
}

function copyDaemonOffer() {
    const offerText = document.getElementById('daemon-offer').value;
    if (!offerText || offerText === 'Generating offer...') {
        alert('Offer not ready yet. Please wait...');
        return;
    }

    navigator.clipboard.writeText(offerText).then(() => {
        alert('Offer copied! Paste it in your app.');
    }).catch((err) => {
        console.error('Failed to copy:', err);
        // Fallback: select text
        document.getElementById('daemon-offer').select();
    });
}

async function connectWithAnswer() {
    const answerBase64 = document.getElementById('app-answer').value.trim();

    if (!answerBase64) {
        alert('Please paste the answer from your app first.');
        return;
    }

    if (!pairingPeerId) {
        alert('No offer was generated. Please refresh the page.');
        return;
    }

    try {
        // Decode answer
        const answerStr = atob(answerBase64);
        const answer = JSON.parse(answerStr);

        // Send answer to backend
        const response = await fetch('/api/webrtc/answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                peer_id: pairingPeerId,
                sdp: answer.sdp,
                type: answer.type
            })
        });

        if (!response.ok) {
            throw new Error('Failed to apply answer: ' + response.statusText);
        }

        updateStatus('Answer applied. Waiting for connection...');

        // Start monitoring for connection
        startStatusMonitoring();

    } catch (error) {
        console.error('Error applying answer:', error);
        updateStatus('Error: ' + error.message, 'error');
        alert('Invalid answer format. Please check and try again.');
    }
}

function updateStatus(message, type = 'info') {
    const statusEl = document.getElementById('webrtc-status');
    let color = 'text-gray-500';

    if (type === 'success') {
        color = 'text-green-600';
    } else if (type === 'error') {
        color = 'text-red-600';
    }

    statusEl.innerHTML = `<span class="${color}">Status: ${message}</span>`;
}

