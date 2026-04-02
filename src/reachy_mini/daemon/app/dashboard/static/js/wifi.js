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

    const connectedDiv = document.getElementById('wifi-connected');
    connectedDiv.classList.add('hidden');

    if (mode == 'hotspot') {
        statusDiv.innerText = 'Hotspot mode active. 🔌';

    } else if (mode == 'wlan') {
        if (currentMode !== null && currentMode !== 'wlan') {
            alert(`Successfully connected to WiFi network: ${status.connected_network} ✅`);
        }

        statusDiv.innerText = `Connected to WiFi (SSID: ${status.connected_network}). 📶`;
        document.getElementById('wifi-network').innerText = status.connected_network;
        document.getElementById('wifi-ip').innerText = status.ip_address || '';
        connectedDiv.classList.remove('hidden');

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

    const connectedDiv = document.getElementById('wifi-connected');
    connectedDiv.classList.add('hidden');

    const knownNetworksDiv = document.getElementById('known-networks');
    knownNetworksDiv.classList.add('hidden');

    const addWifi = document.getElementById('add-wifi');
    addWifi.classList.add('hidden');

    await scanAndListWifiNetworks();
    await refreshStatus();

    addWifi.classList.remove('hidden');
};

// --- Secondary WiFi (wlan1) — mirrors primary pattern ---

const getStatus2 = async () => {
    return await fetch('/wifi/status2')
        .then(response => response.json())
        .catch(error => {
            console.error('Error fetching secondary WiFi status:', error);
            return { exists: false, connected: false, ssid: null, ip_address: null, known_networks: [], busy: false };
        });
};

const refreshStatus2 = async () => {
    const status = await getStatus2();
    handleStatus2(status);
};

const scanAndListWifiNetworks2 = async () => {
    await fetch('/wifi/scan2', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            const ssidSelect = document.getElementById('ssid2');
            ssidSelect.innerHTML = '<option value="" disabled selected>Select your WiFi network</option>';
            data.forEach(ssid => {
                const option = document.createElement('option');
                option.value = ssid;
                option.textContent = ssid;
                ssidSelect.appendChild(option);
            });
        })
        .catch(() => {
            const ssidSelect = document.getElementById('ssid2');
            const option = document.createElement('option');
            option.value = "";
            option.textContent = "Unable to load networks";
            ssidSelect.appendChild(option);
        });
};

const connectToWifi2 = (_) => {
    const ssid = document.getElementById('ssid2').value;
    const password = document.getElementById('password2').value;

    if (!ssid) {
        alert('Please enter an SSID.');
        return;
    }

    fetch(`/wifi/connect2?ssid=${encodeURIComponent(ssid)}&password=${encodeURIComponent(password)}`, {
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
            document.getElementById('ssid2').value = '';
            document.getElementById('password2').value = '';

            return response.json();
        })
        .then(data => {
            handleStatus2({ exists: true, connected: false, ssid: null, ip_address: null, known_networks: [], busy: true });
        })
        .catch(error => {
            console.error('Error connecting to secondary WiFi:', error);
            alert(`Error connecting to WiFi: ${error.message}`);
        });
    return false; // Prevent form submission
};

const handleStatus2 = (status) => {
    const statusDiv = document.getElementById('wifi2-status');
    const connectedDiv = document.getElementById('wifi2-connected');
    const noAdapterDiv = document.getElementById('wifi2-no-adapter');

    const knownNetworksDiv = document.getElementById('known-networks2');
    const knownNetworksList = document.getElementById('known-networks-list2');

    connectedDiv.classList.add('hidden');
    noAdapterDiv.classList.add('hidden');

    // Known networks — same rendering as primary
    knownNetworksList.innerHTML = '';
    if (status.known_networks !== undefined && Array.isArray(status.known_networks) && status.known_networks.length > 0) {
        knownNetworksDiv.classList.remove('hidden');
        status.known_networks.forEach((network) => {
            const li = document.createElement('li');
            li.classList = 'flex flex-row items-center mb-1 gap-4 justify-left';

            const nameSpan = document.createElement('span');
            nameSpan.innerText = network;
            li.appendChild(nameSpan);

            knownNetworksList.appendChild(li);
        });
    } else {
        knownNetworksDiv.classList.add('hidden');
    }

    if (status.busy) {
        statusDiv.innerText = 'Changing your WiFi configuration... Please wait ⏳';
        return;
    }

    if (!status.exists) {
        statusDiv.innerText = 'WiFi adapter not available. ❌';
        document.getElementById('add-wifi2').classList.add('hidden');
        noAdapterDiv.classList.remove('hidden');
        return;
    }

    if (status.connected) {
        statusDiv.innerText = `Connected to WiFi (SSID: ${status.ssid}). 📶`;
        document.getElementById('wifi2-network').innerText = status.ssid;
        document.getElementById('wifi2-ip').innerText = status.ip_address || '';
        connectedDiv.classList.remove('hidden');
    } else {
        statusDiv.innerText = 'WiFi disconnected. ❌';
    }

    document.getElementById('add-wifi2').classList.remove('hidden');
};

const disconnectWifi2 = () => {
    fetch('/wifi/disconnect2', { method: 'POST' })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw new Error(errData.detail || 'Failed to disconnect');
                });
            }
            handleStatus2({ exists: true, connected: false, ssid: null, ip_address: null, known_networks: [], busy: true });
        })
        .catch(error => {
            console.error('Error disconnecting secondary WiFi:', error);
            alert(`Error disconnecting: ${error.message}`);
        });
};

const createWlan1 = () => {
    fetch('/wifi/create_interface', { method: 'POST' })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw new Error(errData.detail || 'Failed to create interface');
                });
            }
            cleanAndRefresh2();
        })
        .catch(error => {
            console.error('Error creating wlan1:', error);
            alert(`Error creating interface: ${error.message}`);
        });
};

const cleanAndRefresh2 = async () => {
    const statusDiv = document.getElementById('wifi2-status');
    statusDiv.innerText = 'Checking WiFi configuration...';

    const knownNetworksDiv = document.getElementById('known-networks2');
    knownNetworksDiv.classList.add('hidden');

    const connectedDiv = document.getElementById('wifi2-connected');
    connectedDiv.classList.add('hidden');

    const addWifi = document.getElementById('add-wifi2');
    addWifi.classList.add('hidden');

    await scanAndListWifiNetworks2();
    await refreshStatus2();

    addWifi.classList.remove('hidden');
};

// --- Initialization ---

window.addEventListener('load', async () => {
    await cleanAndRefresh();
    await cleanAndRefresh2();
    setInterval(() => {
        refreshStatus();
        refreshStatus2();
    }, 1000);
});
