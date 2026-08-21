/**
 * Resolve the signaling URL for the ReachyMini SDK.
 *
 * Priority:
 *   1. `?signaling_url=<url>` URL parameter, honoured only on a loopback page
 *      (local development against staging / self-hosted central).
 *   2. `window.huggingface.variables.SIGNALING_URL` if injected
 *      by the HF Spaces frontmatter (rare).
 *   3. The canonical Pollen-Robotics central
 *      (`https://pollen-robotics-reachy-mini-central.hf.space`).
 *
 * The host attaches the user's Hugging Face bearer to whatever this returns, so
 * a hosted page must not be able to choose that origin: validating the query
 * parameter is not enough, since an attacker-controlled URL can be perfectly
 * well-formed HTTPS. Overrides must also be HTTPS unless both the target and
 * the page are loopback, and must carry no credentials, query, or hash.
 *
 * Why we don't fall back to the SDK's bundled default
 * ───────────────────────────────────────────────────
 * The SDK shipped with `tfrere-reachy-mini-central.hf.space`
 * baked in for legacy reasons. Reachies registered through the
 * mobile app (the canonical onboarding path) sit on the
 * `pollen-robotics-...` Space, so a host that lets the SDK fall
 * back to the legacy URL ends up showing an empty list even when
 * the user owns one of those robots. We override the default
 * here so the host and the mobile app talk to the same central
 * out of the box.
 *
 * Never throws. A malformed or untrusted override just falls back to the
 * canonical default.
 */

export const DEFAULT_CENTRAL_SIGNALING_URL =
  'https://pollen-robotics-reachy-mini-central.hf.space';

function isLoopbackHost(hostname: string): boolean {
  const host = hostname.replace(/^\[|\]$/g, '').toLowerCase();
  return host === 'localhost' || host === '127.0.0.1' || host === '::1';
}

/** Normalise an override, or return null when it must not receive the bearer. */
function trustedOverride(
  candidate: string | undefined | null,
  pageIsLoopback: boolean,
): string | null {
  if (!candidate || candidate !== candidate.trim() || candidate.includes('\\')) {
    return null;
  }

  let parsed: URL;
  try {
    parsed = new URL(candidate);
  } catch {
    return null;
  }

  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') return null;
  if (parsed.username || parsed.password || parsed.search || parsed.hash) return null;

  if (isLoopbackHost(parsed.hostname)) {
    if (!pageIsLoopback) return null;
  } else if (parsed.protocol !== 'https:') {
    return null;
  }

  return `${parsed.origin}${parsed.pathname.replace(/\/+$/, '')}`;
}

export function resolveSignalingUrl(): string {
  if (typeof window === 'undefined') return DEFAULT_CENTRAL_SIGNALING_URL;

  let pageIsLoopback = false;
  try {
    pageIsLoopback = isLoopbackHost(new URL(window.location.origin).hostname);
  } catch {
    pageIsLoopback = false;
  }

  if (pageIsLoopback) {
    try {
      const fromQuery = new URLSearchParams(window.location.search).get(
        'signaling_url',
      );
      const target = trustedOverride(fromQuery, pageIsLoopback);
      if (target) return target;
    } catch {
      /* ignore */
    }
  }

  const configured = trustedOverride(
    window.huggingface?.variables?.SIGNALING_URL,
    pageIsLoopback,
  );
  return configured ?? DEFAULT_CENTRAL_SIGNALING_URL;
}
