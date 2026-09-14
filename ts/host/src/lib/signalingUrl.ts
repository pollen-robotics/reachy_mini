// Hosted query parameters cannot choose where the user's bearer token is sent.

export const DEFAULT_CENTRAL_SIGNALING_URL =
  'https://pollen-robotics-reachy-mini-central.hf.space';

let warnedIgnoredOverride = false;

function isLoopbackHost(hostname: string): boolean {
  return ['localhost', '127.0.0.1', '[::1]'].includes(hostname);
}

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

  const pageIsLoopback = isLoopbackHost(window.location.hostname);
  const fromQuery = new URLSearchParams(window.location.search).get('signaling_url');

  if (pageIsLoopback) {
    const target = trustedOverride(fromQuery, pageIsLoopback);
    if (target) return target;
  } else if (fromQuery !== null && !warnedIgnoredOverride) {
    warnedIgnoredOverride = true;
    console.warn('Ignoring ?signaling_url= outside local development.');
  }

  const configured = trustedOverride(
    window.huggingface?.variables?.SIGNALING_URL,
    pageIsLoopback,
  );
  return configured ?? DEFAULT_CENTRAL_SIGNALING_URL;
}
