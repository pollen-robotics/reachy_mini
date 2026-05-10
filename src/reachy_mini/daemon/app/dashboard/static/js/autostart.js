// Reachy Mini dashboard — Autostart section
// Talks to /api/autostart/* endpoints. Vanilla JS, no framework.

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const fb = (msg, kind) => {
    const el = $("autostartFeedback");
    if (!el) return;
    el.className = "small mb-3 " + (
      kind === "ok"   ? "text-success" :
      kind === "err"  ? "text-danger"  :
      kind === "warn" ? "text-warning" : "text-muted"
    );
    el.textContent = msg || "";
  };

  async function loadConfig() {
    const r = await fetch("/api/autostart/config");
    if (!r.ok) {
      fb(`Failed to load config (${r.status})`, "err");
      return;
    }
    const cfg = await r.json();
    $("autostartEnabled").checked = !!cfg.app_autostart_enabled;
    $("appModule").value = cfg.app_module || "";
    $("appArgs").value = (cfg.app_args || []).join(" ");
  }

  async function loadInstalledApps() {
    const r = await fetch("/api/autostart/installed_apps");
    const dl = $("installedApps");
    dl.innerHTML = "";
    if (!r.ok) return;
    const data = await r.json();
    for (const app of data.apps || []) {
      const opt = document.createElement("option");
      opt.value = app.suggested_module;
      opt.label = `${app.name} ${app.version}`;
      dl.appendChild(opt);
    }
  }

  async function loadStatus() {
    try {
      const [s, d] = await Promise.all([
        fetch("/api/autostart/service_status").then(r => r.json()),
        fetch("/api/autostart/daemon_autostart_status").then(r => r.json()),
      ]);
      $("serviceActive").textContent  = s.active  || "unknown";
      $("serviceEnabled").textContent = s.enabled || "unknown";
      $("daemonEnabled").textContent  = d.enabled || "unknown";
    } catch (e) {
      $("serviceActive").textContent  = "error";
      $("serviceEnabled").textContent = "error";
      $("daemonEnabled").textContent  = "error";
    }
  }

  async function save() {
    const moduleVal = $("appModule").value.trim();
    const argsVal   = $("appArgs").value.trim();
    const enabled   = $("autostartEnabled").checked;

    // Friendly client-side guard: enabling autostart with no module is
    // almost certainly a mistake.
    if (enabled && !moduleVal) {
      fb("Enable autostart but no app module specified — pick or type one.", "warn");
      return;
    }

    const cfg = {
      app_autostart_enabled: enabled,
      app_module: moduleVal || null,
      app_args:   argsVal ? argsVal.split(/\s+/) : [],
    };

    fb("Saving…");
    const r = await fetch("/api/autostart/config", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(cfg),
    });

    if (!r.ok) {
      const text = await r.text();
      fb(`Save failed: ${text}`, "err");
      return;
    }
    fb("Saved. Will take effect on the next boot.", "ok");
    await loadStatus();
  }

  async function init() {
    // Only run if the autostart section is present
    if (!$("autostartEnabled")) return;
    await Promise.all([loadInstalledApps(), loadConfig(), loadStatus()]);

    $("autostartSave").addEventListener("click", save);
    $("autostartReload").addEventListener("click", async () => {
      fb("Reloading…");
      await Promise.all([loadConfig(), loadStatus()]);
      fb("Reloaded.", "ok");
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
