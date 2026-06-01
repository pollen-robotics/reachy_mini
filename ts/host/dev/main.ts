/**
 * Standalone dev harness entry for the host shell.
 *
 * The host package is self-contained - it bundles its own React +
 * MUI - so the shell can be exercised in complete isolation, with
 * no consuming app. This file is the dev-only counterpart of an
 * app's `dispatch.ts`: it routes the same origin between
 *
 *   - the **host shell** (a normal visit), via `mountHost()`, and
 *   - a deliberately minimal **vanilla embed** (the host's iframe,
 *     `/?embedded=1`), via `connectToHost()` - which is all the
 *     host needs on the other side of the postMessage bridge.
 *
 * Run it from `ts/`:
 *
 *   npm install        # once - pulls React/MUI/vite into ts/node_modules
 *   npm run dev        # script: `cd host && vite`  →  http://localhost:5173
 *
 * Auth in dev (optional, via `host/.env.local`, see `.env.example`):
 *   - VITE_HF_TOKEN + VITE_HF_USERNAME  seed a token and skip OAuth
 *     entirely (fastest; lands straight on the picker).
 *   - VITE_HF_OAUTH_CLIENT_ID           exercise the REAL OAuth
 *     redirect - required to see the post-OAuth splash, which only
 *     arms on a genuine redirect return.
 *
 * A live robot reachable through central is only needed once you
 * pick one (to exercise connect → wake → End-session → sleep). The
 * sign-in, picker, welcome and post-OAuth splash beats all work
 * with no robot at all.
 */
import { ReachyMini } from '@pollen-robotics/reachy-mini-sdk';

// Same bootstrap an app's dispatcher does: expose the SDK
// constructor on the global both the host shell and the embed
// client wait for.
(window as unknown as { ReachyMini: typeof ReachyMini }).ReachyMini = ReachyMini;
window.dispatchEvent(new Event('reachymini:ready'));

// Vite injects `import.meta.env`; read it defensively so this file
// stays valid TS even without `vite/client` types in scope.
const env = (import.meta as unknown as { env?: Record<string, string | undefined> })
  .env ?? {};

const isEmbed = new URLSearchParams(window.location.search).get('embedded') === '1';

if (isEmbed) {
  void bootEmbed();
} else {
  void bootHost();
}

async function bootHost(): Promise<void> {
  const { mountHost } = await import('@/mountHost');
  const token = env.VITE_HF_TOKEN;
  // The username is only needed for the dev-token shortcut. If it
  // wasn't supplied, resolve it from the token via HF `whoami` so a
  // bare `VITE_HF_TOKEN=...` is enough to skip OAuth.
  const userName = token
    ? env.VITE_HF_USERNAME ?? (await resolveUserName(token))
    : undefined;

  mountHost({
    appName: 'Host Dev Harness',
    appEmoji: '🧪',
    enableMicrophone: false,
    clientId: env.VITE_HF_OAUTH_CLIENT_ID,
    devToken: token && userName ? { token, userName } : undefined,
  });
}

/** Best-effort `whoami` so the harness only needs `VITE_HF_TOKEN`. */
async function resolveUserName(token: string): Promise<string | undefined> {
  try {
    const res = await fetch('https://huggingface.co/api/whoami-v2', {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) return undefined;
    const who = (await res.json()) as { name?: string };
    return who.name;
  } catch {
    return undefined;
  }
}

async function bootEmbed(): Promise<void> {
  const { connectToHost } = await import('@/embed');

  const root = document.getElementById('root') ?? document.body;
  root.innerHTML = `
    <main style="font-family:system-ui,sans-serif;display:flex;flex-direction:column;
                 align-items:center;justify-content:center;height:100%;gap:16px;padding:24px;
                 text-align:center;color:#111">
      <h1 style="margin:0;font-size:1.25rem">Minimal embed</h1>
      <p id="status" style="margin:0;opacity:.7">Connecting…</p>
      <button id="leave" type="button" hidden
              style="padding:8px 16px;border-radius:8px;border:1px solid #999;cursor:pointer">
        Request leave (in-app)
      </button>
    </main>`;

  const setStatus = (msg: string): void => {
    const el = document.getElementById('status');
    if (el) el.textContent = msg;
  };

  try {
    const handle = await connectToHost();
    setStatus('Live — robot awake. End the session (top bar) to test the leave contract.');

    const leaveBtn = document.getElementById('leave') as HTMLButtonElement | null;
    if (leaveBtn) {
      leaveBtn.hidden = false;
      leaveBtn.addEventListener('click', () => handle.requestLeave());
    }

    handle.onLeave(() => {
      // App's OWN cleanup only - per the leave contract the runtime
      // sleeps the robot and releases the torque for us.
      setStatus('onLeave fired — app cleaning up; runtime is sleeping the robot.');
    });
  } catch (err) {
    setStatus(`connectToHost failed: ${(err as Error)?.message ?? String(err)}`);
  }
}
