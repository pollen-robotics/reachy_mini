/**
 * Resolve the signaling URL for the ReachyMini SDK.
 *
 * Priority:
 *   1. `?signaling_url=<url>` URL parameter (override for
 *      staging / self-hosted central).
 *   2. `window.huggingface.variables.SIGNALING_URL` if injected
 *      by the HF Spaces frontmatter (rare).
 *   3. The canonical Pollen-Robotics central
 *      (`https://pollen-robotics-reachy-mini-central.hf.space`).
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
 * Never throws. A malformed override just falls back to the
 * canonical default.
 */

export const DEFAULT_CENTRAL_SIGNALING_URL =
  'https://pollen-robotics-reachy-mini-central.hf.space';

export function resolveSignalingUrl(): string {
  if (typeof window === 'undefined') return DEFAULT_CENTRAL_SIGNALING_URL;

  try {
    const fromQuery = new URLSearchParams(window.location.search).get(
      'signaling_url',
    );
    if (fromQuery && /^https?:\/\//.test(fromQuery)) {
      return fromQuery;
    }
  } catch {
    /* ignore */
  }

  const fromEnv = (window.huggingface?.variables as
    | (Record<string, string | undefined> | undefined)) ?? undefined;
  const envSignaling = fromEnv?.['SIGNALING_URL'];
  if (envSignaling && /^https?:\/\//.test(envSignaling)) {
    return envSignaling;
  }

  return DEFAULT_CENTRAL_SIGNALING_URL;
}
