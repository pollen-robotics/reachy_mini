const volumeControl = {
  currentVolume: 50,
  device: 'unknown',
  platform: 'unknown',
  isUpdating: false,

  init: async () => {
    const slider = document.getElementById('volume-slider');
    const valueLabel = document.getElementById('volume-value');
    const deviceInfo = document.getElementById('volume-device-info');

    if (!slider || !valueLabel || !deviceInfo) {
      console.warn('Volume control elements not found in DOM');
      return;
    }

    try {
      await volumeControl.loadCurrentVolume();
    } catch (error) {
      console.error('Error loading current volume:', error);
      deviceInfo.textContent = 'Error loading volume';
    }

    slider.addEventListener('input', (e) => {
      const v = Number(e.target.value);
      valueLabel.textContent = String(v);
    });

    slider.addEventListener('change', async (e) => {
      const newVolume = Number(e.target.value);
      if (!Number.isFinite(newVolume)) return;
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

      const volume = Number(data.volume);
      if (!Number.isFinite(volume)) {
        throw new Error('Invalid volume in response');
      }

      volumeControl.currentVolume = volume;
      volumeControl.device = data.device ?? 'unknown';
      volumeControl.platform = data.platform ?? 'unknown';

      const slider = document.getElementById('volume-slider');
      const valueLabel = document.getElementById('volume-value');
      const deviceInfo = document.getElementById('volume-device-info');

      if (slider) slider.value = String(volume);
      if (valueLabel) valueLabel.textContent = String(volume);
      if (deviceInfo) {
        deviceInfo.textContent = `${volumeControl.platform} - ${volumeControl.device}`;
      }

      console.log('Loaded volume:', data);
    } catch (error) {
      console.error('Error loading current volume:', error);
      throw error;
    }
  },

  setVolume: async (volume) => {
    if (!Number.isFinite(volume)) {
      console.warn('Ignoring invalid volume:', volume);
      return;
    }

    const safeVolume = Math.max(0, Math.min(100, volume));
    if (volumeControl.isUpdating) {
      console.log('Volume update already in progress, skipping...');
      return;
    }

    volumeControl.isUpdating = true;
    const slider = document.getElementById('volume-slider');

    if (slider) slider.disabled = true;

    try {
      const response = await fetch('/api/volume/set', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ volume: safeVolume }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const serverVolume = Number(data.volume);

      if (Number.isFinite(serverVolume)) {
        volumeControl.currentVolume = serverVolume;
        const s = document.getElementById('volume-slider');
        const valueLabel = document.getElementById('volume-value');
        if (s) s.value = String(serverVolume);
        if (valueLabel) valueLabel.textContent = String(serverVolume);
      }

      console.log('Volume set to:', serverVolume);
    } catch (error) {
      console.error('Error setting volume:', error);
      try {
        await volumeControl.loadCurrentVolume();
      } catch (loadError) {
        console.error('Also failed to reload volume:', loadError);
      }
    } finally {
      volumeControl.isUpdating = false;
      const s = document.getElementById('volume-slider');
      if (s) s.disabled = false;
    }
  },
};

const microphoneControl = {
  currentVolume: 50,
  device: 'unknown',
  platform: 'unknown',
  isUpdating: false,

  init: async () => {
    const slider = document.getElementById('microphone-slider');
    const valueLabel = document.getElementById('microphone-value');
    const deviceInfo = document.getElementById('microphone-device-info');

    if (!slider || !valueLabel || !deviceInfo) {
      console.warn('Microphone control elements not found in DOM');
      return;
    }

    try {
      await microphoneControl.loadCurrentVolume();
    } catch (error) {
      console.error('Error loading current microphone volume:', error);
      deviceInfo.textContent = 'Error loading microphone';
    }

    slider.addEventListener('input', (e) => {
      const v = Number(e.target.value);
      valueLabel.textContent = String(v);
    });

    slider.addEventListener('change', async (e) => {
      const newVolume = Number(e.target.value);
      if (!Number.isFinite(newVolume)) return;
      await microphoneControl.setVolume(newVolume);
    });
  },

  loadCurrentVolume: async () => {
    try {
      const response = await fetch('/api/volume/microphone/current');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();

      const volume = Number(data.volume);
      if (!Number.isFinite(volume)) {
        throw new Error('Invalid microphone volume in response');
      }

      microphoneControl.currentVolume = volume;
      microphoneControl.device = data.device ?? 'unknown';
      microphoneControl.platform = data.platform ?? 'unknown';

      const slider = document.getElementById('microphone-slider');
      const valueLabel = document.getElementById('microphone-value');
      const deviceInfo = document.getElementById('microphone-device-info');

      if (slider) slider.value = String(volume);
      if (valueLabel) valueLabel.textContent = String(volume);
      if (deviceInfo) {
        deviceInfo.textContent = `${microphoneControl.platform} - ${microphoneControl.device}`;
      }

      console.log('Loaded microphone volume:', data);
    } catch (error) {
      console.error('Error loading current microphone volume:', error);
      throw error;
    }
  },

  setVolume: async (volume) => {
    if (!Number.isFinite(volume)) {
      console.warn('Ignoring invalid microphone volume:', volume);
      return;
    }

    const safeVolume = Math.max(0, Math.min(100, volume));
    if (microphoneControl.isUpdating) {
      console.log('Microphone volume update already in progress, skipping...');
      return;
    }

    microphoneControl.isUpdating = true;
    const slider = document.getElementById('microphone-slider');

    if (slider) slider.disabled = true;

    try {
      const response = await fetch('/api/volume/microphone/set', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ volume: safeVolume }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const serverVolume = Number(data.volume);

      if (Number.isFinite(serverVolume)) {
        microphoneControl.currentVolume = serverVolume;
        const s = document.getElementById('microphone-slider');
        const valueLabel = document.getElementById('microphone-value');
        if (s) s.value = String(serverVolume);
        if (valueLabel) valueLabel.textContent = String(serverVolume);
      }

      console.log('Microphone volume set to:', serverVolume);
    } catch (error) {
      console.error('Error setting microphone volume:', error);
      try {
        await microphoneControl.loadCurrentVolume();
      } catch (loadError) {
        console.error('Also failed to reload microphone volume:', loadError);
      }
    } finally {
      microphoneControl.isUpdating = false;
      const s = document.getElementById('microphone-slider');
      if (s) s.disabled = false;
    }
  },
};

window.addEventListener('DOMContentLoaded', () => {
  volumeControl.init();
  microphoneControl.init();
});
