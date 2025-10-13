
window.onload = () => {
    refreshStatus();
    setInterval(refreshStatus, 1000);
};


const refreshStatus = () => {
    const statusDiv = document.getElementById('status');

    fetch('/wifi/status')
        .then(response => response.json())
        .then(data => {
            statusDiv.innerText = `Status: ${data}`;

            handleStatus(data);
        })
        .catch(error => {
            console.error('Error fetching WiFi status:', error);
            statusDiv.innerText = 'Make sure you are connected on the right WiFi.\n Attempt to reconnect...';
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
}

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
}

const handleStatus = (status) => {
    const addWifiDiv = document.getElementById('add-wifi');
    addWifiDiv.hidden = true;

    const busyDiv = document.getElementById('busy');
    busyDiv.hidden = true;

    const connectedDiv = document.getElementById('connected');
    connectedDiv.hidden = true;

    if (status === 'hotspot') {
        addWifiDiv.hidden = false;
    } else if (status === 'wlan') {
        connectedDiv.hidden = false;
    }
    else if (status === 'busy') {
        busyDiv.hidden = false;
    }
    else {
        console.warn(`Unknown status: ${status}`);
    }

    currentStatus = status;
};