/**
 * Imperative mount helper for app authors.
 *
 *   import { mountHost } from '@pollen-robotics/reachy-mini-sdk/host/auto';
 *
 *   mountHost({
 *     appName: 'Emotions',
 *     appIconUrl: '/icon.svg',
 *     enableMicrophone: false,
 *   });
 *
 * This is the recommended entry for app authors. It handles:
 *  - Finding (or auto-creating) the mount node.
 *  - Wrapping in `React.StrictMode` so dev-mode double-mount is
 *    a working configuration (SPEC §8.4).
 *  - Seeding a dev token (`devToken`) for local development
 *    without OAuth.
 *  - Stamping the document title with `appName` for browser
 *    tab clarity.
 *
 * It returns an `unmount()` function for cleanup in tests /
 * Hot-Module-Reload teardown.
 */
import { StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';

import { ReachyHost } from './ReachyHost';
import { clearSignedOutFlag, seedDevToken } from './lib/settings';
import type { ConfigPayload } from './lib/protocol';

export interface MountHostOptions {
  /** REQUIRED. Passed to the SDK + shown in the top bar. */
  appName: string;
  /** Optional top-bar logo. Recommended PNG/SVG, 32×32. */
  appIconUrl?: string;
  /** Optional emoji fallback (used when no icon URL). */
  appEmoji?: string;
  /** Allow microphone capture inside the iframe. */
  enableMicrophone?: boolean;
  /** HF OAuth client ID override; falls back to HF Spaces
   *  injection then localStorage. */
  clientId?: string;
  /** Dev shortcut: seed a personal access token so the host
   *  skips the OAuth dance. Strip this before deploying. */
  devToken?: { token: string; userName: string };
  /** Mount target: an HTMLElement or a CSS selector. Defaults
   *  to `#root` (auto-created if missing). */
  target?: HTMLElement | string;
  /** Embed path within the same origin. Defaults to
   *  `/?embedded=1`. */
  embedPath?: string;
  /** Override the host display name in the top bar. */
  hostName?: string;
  /** Pre-set the initial config (mostly for tests; production
   *  reads `?config=` automatically). */
  initialConfig?: ConfigPayload;
}

export interface MountedHost {
  unmount(): void;
}

export function mountHost(options: MountHostOptions): MountedHost {
  if (!options.appName) {
    throw new Error('[reachy-mini-sdk/host] mountHost: `appName` is required.');
  }

  if (options.devToken) {
    // Local dev shortcut: also clear the signed-out flag from a
    // previous session. Otherwise the user-initiated logout flag
    // sticks across reloads and `useOAuth` skips the auto-auth
    // step, leaving the host stuck on SignInView even though we
    // just seeded a fresh token.
    clearSignedOutFlag();
    // NOTE: we deliberately do NOT call `markOAuthPending()` here.
    // The welcome-back animation is reserved for genuine OAuth
    // round-trips (`sdk.login()` -> HF redirect -> return). A
    // dev-token seed is not an OAuth flow, it's a paste-in token,
    // so playing the animation on every fresh page load would lie
    // to the user about what just happened.
    seedDevToken(options.devToken);
  }

  const target = resolveTarget(options.target);
  applyAppMeta(options);

  const root: Root = createRoot(target);
  root.render(
    <StrictMode>
      <ReachyHost
        appName={options.appName}
        appIconUrl={options.appIconUrl}
        appEmoji={options.appEmoji}
        enableMicrophone={options.enableMicrophone ?? false}
        clientId={options.clientId}
        embedPath={options.embedPath ?? '/?embedded=1'}
        hostName={options.hostName ?? 'Reachy Mini'}
        initialConfig={options.initialConfig}
      />
    </StrictMode>,
  );

  return {
    unmount() {
      root.unmount();
    },
  };
}

function resolveTarget(target: MountHostOptions['target']): HTMLElement {
  if (target instanceof HTMLElement) return target;
  const selector = typeof target === 'string' ? target : '#root';
  const found = document.querySelector<HTMLElement>(selector);
  if (found) return found;
  if (selector === '#root') {
    // Standalone-mode contract: the page's `index.html` typically
    // ships markup for the embedded app (revealed by the embed
    // entry once mounted). In standalone mode that markup is not
    // relevant, so we wipe the body and inject a fresh `#root`.
    // App authors who need to keep specific nodes can pass an
    // explicit `target` (HTMLElement or selector that exists in
    // the DOM).
    document.body.replaceChildren();
    document.body.classList.remove('booting');
    const created = document.createElement('div');
    created.id = 'root';
    document.body.appendChild(created);
    return created;
  }
  throw new Error(
    `[reachy-mini-sdk/host] mountHost: target '${selector}' not found in DOM.`,
  );
}

function applyAppMeta(options: MountHostOptions): void {
  if (typeof document === 'undefined') return;
  // Best-effort title; apps that ship their own title can leave
  // it alone by setting it in their HTML.
  if (!document.title || document.title === '') {
    document.title = options.appName;
  }
  // If the app supplied an icon URL and the page has no
  // <link rel="icon">, drop one in.
  if (options.appIconUrl) {
    const existing = document.querySelector('link[rel="icon"]');
    if (!existing) {
      const link = document.createElement('link');
      link.rel = 'icon';
      link.href = options.appIconUrl;
      document.head.appendChild(link);
    }
  }
}
