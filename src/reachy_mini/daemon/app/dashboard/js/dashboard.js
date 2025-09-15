async function fetchDaemonStatus() {
    try {
        const res = await fetch('/api/daemon/status');
        const status = await res.json();
        document.getElementById('daemon-status').textContent = `State: ${status.state}${status.error ? ' | Error: ' + status.error : ''}`;
        if (status.backend_status) {
            document.getElementById('daemon-status').innerHTML += `<br>Backend status: ${JSON.stringify(status.backend_status)}`;
        }
        // Enable/disable buttons based on daemon state
        const startBtn = document.getElementById('start-daemon');
        const stopBtn = document.getElementById('stop-daemon');
        const restartBtn = document.getElementById('restart-daemon');
        if (status.state === 'not_initialized' || status.state === 'stopped' || status.state === 'error') {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            restartBtn.disabled = true;
        } else if (status.state === 'running') {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            restartBtn.disabled = false;
        } else if (status.state === 'starting') {
            startBtn.disabled = true;
            stopBtn.disabled = true;
            restartBtn.disabled = true;
        } else if (status.state === 'stopping') {
            startBtn.disabled = true;
            stopBtn.disabled = true;
            restartBtn.disabled = true;
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            restartBtn.disabled = true;
        }
    } catch (e) {
        document.getElementById('daemon-status').textContent = 'Error fetching daemon status.';
    }
}
fetchDaemonStatus();
setInterval(fetchDaemonStatus, 2000);

document.getElementById('start-daemon').onclick = async function () {
    const wakeUp = document.getElementById('wake-up-on-start').checked;
    await fetch('/api/daemon/start?wake_up=' + wakeUp, {
        method: 'POST',
    });
    fetchDaemonStatus();
};
document.getElementById('stop-daemon').onclick = async function () {
    const gotoSleep = document.getElementById('goto-sleep-on-stop').checked;
    await fetch('/api/daemon/stop?goto_sleep=' + gotoSleep, {
        method: 'POST',
    });
    fetchDaemonStatus();
};
document.getElementById('restart-daemon').onclick = async function () {
    await fetch('/api/daemon/restart', { method: 'POST' });
    fetchDaemonStatus();
};



function onConnectVideo() {
    const ip = document.getElementById('video-ip').value;
    console.log('Connect to video stream at IP:', ip);

    const signalingProtocol = window.location.protocol.startsWith("https") ? "wss" : "ws";
    const gstWebRTCConfig = {
        meta: { name: `WebClient-${Date.now()}` },
        signalingServerUrl: `${signalingProtocol}://${ip}:8443`,
    };
    const api = new GstWebRTCAPI(gstWebRTCConfig);

    const listener = {
        producerAdded: (producer) => {
            console.log("Found producer: ", producer);
            if (producer.meta.name !== "reachymini") {
                console.log("Ignoring producer with name: ", producer.meta.name);
                return;
            }

            const session = api.createConsumerSession(producer.id);
            console.log("Created session: ", session);

            session.addEventListener("error", (event) => {
                if (entryElement._consumerSession === session) {
                    console.error(event.message, event.error);
                }
            });

            session.addEventListener("streamsChanged", () => {
                console.log("Streams changed: ", session);
                if (session.streams.length > 0) {
                    // Do something with the active streams
                    const videoElement = document.getElementById("video");
                    videoElement.srcObject = session.streams[0];
                    videoElement.play();
                }
            });
            session.addEventListener("remoteControllerChanged", () => {
                console.log("Remote controller changed: ", session);
            });

            session.addEventListener("closed", (event) => {
                console.log("Session closed: ", session);
            });

            session.connect();
        },
        producerRemoved: (producer) => {
            console.log("Producer removed: ", producer);
        },
    };

    api.registerProducersListener(listener);
    for (const producer of api.getAvailableProducers()) {
        console.log("Found producer: ", producer);
        listener.producerAdded(producer);
    }
}

document.getElementById('connect-video').onclick = onConnectVideo;
