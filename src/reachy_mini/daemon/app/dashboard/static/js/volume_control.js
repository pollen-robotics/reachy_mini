const volumeControl = {
    currentVolume: 50,
    device: 'unknown',
    platform: 'unknown',
    isUpdating: false,

    init: async () => {
        const volumeSlider = document.getElementById('volume-slider');
        const volumeValue = document.getElementById('volume-value');
        const volumeDeviceInfo = document.getElementById('volume-device-info');

        if (!volumeSlider || !volumeValue || !volumeDeviceInfo) {
            console.warn('Volume control elements not found in DOM');
            return;
        }

        // Load current volume
        try {
            await volumeControl.loadCurrentVolume();
        } catch (error) {
            console.error('Error loading current volume:', error);
            volumeDeviceInfo.textContent = 'Error loading volume';
        }

        // Handle slider input (for live feedback)
        volumeSlider.addEventListener('input', (e) => {
            volumeValue.textContent = e.target.value;
        });

        // Handle slider change (when user releases)
        volumeSlider.addEventListener('change', async (e) => {
            const newVolume = parseInt(e.target.value);
            await volumeControl.setVolume(newVolume);
        });
    },

    loadCurrentVolume: async () => {
        try {
            const response = await fetch('/api/volume/current');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            
            volumeControl.currentVolume = data.volume;
            volumeControl.device = data.device;
            volumeControl.platform = data.platform;

            // Update UI
            const volumeSlider = document.getElementById('volume-slider');
            const volumeValue = document.getElementById('volume-value');
            const volumeDeviceInfo = document.getElementById('volume-device-info');

            if (volumeSlider) volumeSlider.value = data.volume;
            if (volumeValue) volumeValue.textContent = data.volume;
            if (volumeDeviceInfo) {
                volumeDeviceInfo.textContent = `${data.platform} - ${data.device}`;
            }

            console.log('Loaded volume:', data);
        } catch (error) {
            console.error('Error loading current volume:', error);
            throw error;
        }
    },

    setVolume: async (volume) => {
        if (volumeControl.isUpdating) {
            console.log('Volume update already in progress, skipping...');
            return;
        }

        volumeControl.isUpdating = true;

        try {
            const response = await fetch('/api/volume/set', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ volume: volume }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            volumeControl.currentVolume = data.volume;

            console.log('Volume set to:', data.volume);
        } catch (error) {
            console.error('Error setting volume:', error);
            // Reload current volume in case of error
            await volumeControl.loadCurrentVolume();
        } finally {
            volumeControl.isUpdating = false;
        }
    },
};

window.addEventListener('DOMContentLoaded', (event) => {
    volumeControl.init();
});
