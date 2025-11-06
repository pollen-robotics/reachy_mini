
const getStatus = async () => {
    return await fetch('/wifi/status')
        .then(response => response.json())
        .catch(error => {
            console.error('Error fetching WiFi status:', error);
            return { mode: 'error' };
        });
};

const refreshStatus = async () => {
    const status = await getStatus();
    handleStatus(status);

    await fetch('/wifi/error')
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

const scanAndListWifiNetworks = async () => {
    await fetch('/wifi/scan_and_list', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            const ssidSelect = document.getElementById('ssid');
            data.forEach(ssid => {
                const option = document.createElement('option');
                option.value = ssid;
                option.textContent = ssid;
                ssidSelect.appendChild(option);
            });
        })
        .catch(() => {
            const ssidSelect = document.getElementById('ssid');
            const option = document.createElement('option');
            option.value = "";
            option.textContent = "Unable to load networks";
            ssidSelect.appendChild(option);
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

            // Clear the form fields
            document.getElementById('ssid').value = '';
            document.getElementById('password').value = '';

            return response.json();
        })
        .then(data => {
            handleStatus({ mode: 'busy' });
        })
        .catch(error => {
            console.error('Error connecting to WiFi:', error);
            alert(`Error connecting to WiFi: ${error.message}`);
        });
    return false; // Prevent form submission
};

let currentMode = null;

const handleStatus = (status) => {
    const statusDiv = document.getElementById('wifi-status');

    const knownNetworksDiv = document.getElementById('known-networks');
    const knownNetworksList = document.getElementById('known-networks-list');
    knownNetworksDiv.classList.remove('hidden');

    const mode = status.mode;

    knownNetworksList.innerHTML = '';
    if (status.known_networks !== undefined && Array.isArray(status.known_networks)) {
        status.known_networks.forEach((network) => {
            const li = document.createElement('li');
            li.classList = 'flex flex-row items-center mb-1 gap-4 justify-left';

            const nameSpan = document.createElement('span');
            nameSpan.innerText = network;
            li.appendChild(nameSpan);

            // const removeBtn = document.createElement('span');
            // removeBtn.innerText = ' (remove ❌)';
            // removeBtn.style.cursor = 'pointer';
            // removeBtn.title = 'Remove network';
            // removeBtn.onclick = async () => {
            //     if (confirm(`Remove network '${network}'?`)) {
            //         removeNetwork(network);
            //     }
            // };
            // li.appendChild(removeBtn);

            knownNetworksList.appendChild(li);
        });
    }

    if (mode == 'hotspot') {
        statusDiv.innerText = 'Hotspot mode active. 🔌';

    } else if (mode == 'wlan') {
        if (currentMode !== null && currentMode !== 'wlan') {
            alert(`Successfully connected to WiFi network: ${status.connected_network} ✅`);
        }

        statusDiv.innerText = `Connected to WiFi (SSID: ${status.connected_network}). 📶`;

    } else if (mode == 'disconnected') {
        statusDiv.innerText = 'WiFi disconnected. ❌';
    } else if (mode == 'busy') {
        statusDiv.innerText = 'Changing your WiFi configuration... Please wait ⏳';
    } else if (mode == 'error') {
        statusDiv.innerText = 'Error connecting to WiFi. ⚠️';
    } else {
        console.warn(`Unknown status: ${status}`);
    }

    currentMode = mode;
};

const removeNetwork = async (ssid) => {
    const status = await getStatus();

    // TODO:
    // if ssid !== status.connected_network:
    //    remove connection
    // else:
    //    refresh nmcli? go back to hotspot if needed?
};

const cleanAndRefresh = async () => {
    const statusDiv = document.getElementById('wifi-status');
    statusDiv.innerText = 'Checking WiFi configuration...';

    const knownNetworksDiv = document.getElementById('known-networks');
    knownNetworksDiv.classList.add('hidden');

    const addWifi = document.getElementById('add-wifi');
    addWifi.classList.add('hidden');

    await scanAndListWifiNetworks();
    await refreshStatus();

    addWifi.classList.remove('hidden');
};

window.addEventListener('load', async () => {
    await cleanAndRefresh();
    setInterval(refreshStatus, 1000);
});