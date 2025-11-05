
const refreshStatus = () => {
    fetch('/wifi/status')
        .then(response => response.json())
        .then(data => {
            handleStatus(data);
        })
        .catch(error => {
            console.error('Error fetching WiFi status:', error);
            handleStatus('error');
        });

    fetch('/wifi/error')
        .then(response => response.json())
        .then(data => {
            if (data.error !== null) {
                console.log('Error data:', data);
                alert(`Error while trying to connect: ${data.error}.\n Switching back to hotspot mode.`);
                fetch('/wifi/reset_error', { method: 'POST' });
            }
        })
        .catch(error => {
            console.error('Error fetching WiFi error:', error);
        });
};

const connectToWifi = (_) => {
    const ssid = document.getElementById('ssid').value;
    const password = document.getElementById('password').value;

    if (!ssid) {
        alert('Please enter an SSID.');
        return;
    }

    fetch(`/wifi/connect?ssid=${encodeURIComponent(ssid)}&password=${encodeURIComponent(password)}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw new Error(errData.detail || 'Failed to connect to WiFi');
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Connection response:', data);
            handleStatus('busy');
        })
        .catch(error => {
            console.error('Error connecting to WiFi:', error);
            alert(`Error connecting to WiFi: ${error.message}`);
        });
    return false; // Prevent form submission
};

const handleStatus = (status) => {
    const statusDiv = document.getElementById('wifi-status');
    const addWifiDiv = document.getElementById('add-wifi');
    const busyDiv = document.getElementById('busy');

    addWifiDiv.classList = 'hidden';

    if (status == 'hotspot') {
        statusDiv.innerText = 'Hotspot mode active. 🔌';
        addWifiDiv.classList.remove('hidden');
    } else if (status == 'wlan') {
        statusDiv.innerText = 'Connected to WiFi. 📶';
    } else if (status == 'disconnected') {
        statusDiv.innerText = 'WiFi disconnected. ❌';
    } else if (status == 'busy') {
        statusDiv.innerText = 'Changing your WiFi configuration... Please wait ⏳';
        busyDiv.hidden = false;
    } else if (status == 'error') {
        statusDiv.innerText = 'Error connecting to WiFi. ⚠️';
    } else {
        console.warn(`Unknown status: ${status}`);
    }

    currentStatus = status;
};

window.addEventListener('load', () => {
    refreshStatus();
    setInterval(refreshStatus, 1000);
});