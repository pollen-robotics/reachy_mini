window.addEventListener("load", () => {
    const healthCheckInterval = setInterval(async () => {
        const response = await fetch("/health-check", { method: "POST" });
        if (!response.ok) {
            console.error("Health check failed");
            clearInterval(healthCheckInterval);
        }
    }, 2500);
});
