const notificationCenter = {

    showInfo: (message, duration = 5000, autoclose = true) => {
        notificationCenter.showNotification(message, false, duration, autoclose);
    },
    showError: (message, duration = 5000, autoclose = false) => {
        message = notificationCenter.errorTranslation(message);
        notificationCenter.showNotification(message, true, duration, autoclose);
    },

    showNotification: (message, error, duration = 5000, autoclose = true) => {
        document.getElementById('notification-message').textContent = message;
        document.getElementById('notification-modal').style.display = 'block';

        if (error) {
            document.getElementById('notification-content').classList.add('error');
            document.getElementById('notification-content').classList.remove('info');
        } else {
            document.getElementById('notification-content').classList.remove('error');
            document.getElementById('notification-content').classList.add('info');
        }

        if (autoclose) {
            setTimeout(notificationCenter.closeNotification, duration);
        }
    },

    closeNotification: () => {
        document.getElementById('notification-modal').style.display = 'none';
    },

    errorTranslation: (errorMessage) => {
        if (errorMessage.includes('No Reachy Mini serial port found.')) {
            return 'Reachy Mini not detected on USB. Please check that the USB cable is properly connected.';
        }

        console.log('No translation found for error message:', errorMessage);
        return errorMessage;
    },

};