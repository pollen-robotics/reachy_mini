/**
 * URL / SDP helpers used by the ReachyMini SDK at construction
 * time and during signaling.
 *
 * Each function is a no-op outside the browser (SSR / Worker
 * contexts) so the module is import-safe everywhere.
 */

/**
 * Pick up HuggingFace credentials passed via the URL fragment and move them
 * into `sessionStorage`, where `authenticate()` looks them up.
 *
 * This is the bridge that lets a host page (e.g. the Reachy Mini mobile
 * app, or the vibe-coder preview iframe) embed a Space hosting a SDK
 * consumer despite `X-Frame-Options: SAMEORIGIN` on `huggingface.co/login`:
 * the host already holds a valid token (through its own OAuth flow) and
 * appends it to the iframe URL as
 *
 *     #hf_token=<jwt>&hf_username=<handle>&hf_token_expires=<iso>
 *
 * Fragments are NOT sent over HTTP, so the credentials never leak to
 * the HF Space backend or to intermediate proxies.
 *
 * Why all three keys: `authenticate()`'s cache check requires the token,
 * the username AND a future expiry to ALL be present in `sessionStorage`,
 * otherwise it returns `false` and the app falls through to a full OAuth
 * round-trip — which can't complete inside an iframe.
 *
 * Called once from the top of `authenticate()` so SDK consumers don't
 * need any boilerplate of their own. We clear the fragment right after
 * reading it so a page reload does not keep the credentials visible in
 * the address bar.
 *
 * No-op when:
 *   - there is no `window` (SSR / Worker contexts),
 *   - the URL has no fragment,
 *   - the fragment carries no `hf_token` (other apps may use the
 *     fragment for theme / route / etc.; we leave those alone).
 */
export function consumeFragmentCredentials(): void {
    if (typeof window === 'undefined' || !window.location.hash) return;
    const raw = window.location.hash.startsWith('#')
        ? window.location.hash.slice(1)
        : window.location.hash;
    let params: URLSearchParams;
    try { params = new URLSearchParams(raw); } catch { return; }
    const token = params.get('hf_token');
    if (!token) return;
    // `hf_username` is required by the cache check. Hosts that haven't
    // resolved the user's HF handle yet may pass a literal "user"
    // placeholder; the SDK only uses the value for display and never
    // round-trips it server-side, so the placeholder is harmless.
    const username = params.get('hf_username') || 'user';
    // `hf_token_expires` is a far-future ISO date for personal access
    // tokens (no real expiry). Hosts typically synthesise ~1 year out;
    // we accept whatever was sent and fall back to "1 year from now"
    // if the parameter is missing or unparseable, so a partial fragment
    // still gets the user logged in.
    const expiresParam = params.get('hf_token_expires');
    const expires =
        expiresParam && !Number.isNaN(new Date(expiresParam).getTime())
            ? expiresParam
            : new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString();
    try {
        sessionStorage.setItem('hf_token', token);
        sessionStorage.setItem('hf_username', username);
        sessionStorage.setItem('hf_token_expires', expires);
    } catch (err) {
        console.warn('[reachy-mini] could not persist pre-seeded HF credentials:', err);
    }
    // Strip the auth keys from the address bar but keep any other hash
    // params the app or SDK might care about (theme, embedded, …).
    params.delete('hf_token');
    params.delete('hf_username');
    params.delete('hf_token_expires');
    const remaining = params.toString();
    const cleanUrl =
        window.location.pathname +
        window.location.search +
        (remaining ? '#' + remaining : '');
    try { window.history.replaceState(null, '', cleanUrl); } catch { /* ignore */ }
}

/**
 * Pick up a preselected robot peer id from the URL.
 *
 * Looked up in this order:
 *   1. URL fragment   (`#robot_peer_id=<peerId>`)
 *   2. URL query      (`?robot_peer_id=<peerId>`)
 *
 * Both spellings are accepted because:
 *   - the Reachy Mini mobile shell sends it in the query today,
 *   - the vibe-coder preview / future hosts may prefer the fragment for
 *     symmetry with `consumeFragmentCredentials`,
 *   - the value is NOT a secret (peer ids are public on the central
 *     signaling server's robot listing) so query is fine.
 *
 * Returns `null` when no peer id is found in either location, when there
 * is no `window` (SSR / Worker context), or on parse error. Unlike
 * credentials, we do NOT strip the param from the URL: the value is
 * harmless to keep visible and removing it would break tools that read
 * the URL for context.
 */
export function readPreselectedRobotIdFromUrl(): string | null {
    if (typeof window === 'undefined') return null;
    // 1. Fragment (`#robot_peer_id=…`).
    if (window.location.hash) {
        const raw = window.location.hash.startsWith('#')
            ? window.location.hash.slice(1)
            : window.location.hash;
        try {
            const params = new URLSearchParams(raw);
            const fromHash = params.get('robot_peer_id');
            if (fromHash) return fromHash;
        } catch { /* malformed fragment — fall through */ }
    }
    // 2. Query (`?robot_peer_id=…`).
    if (window.location.search) {
        try {
            const params = new URLSearchParams(window.location.search);
            const fromQuery = params.get('robot_peer_id');
            if (fromQuery) return fromQuery;
        } catch { /* malformed query — fall through */ }
    }
    return null;
}

/** Check if the audio m= section of an SDP has a=sendrecv (bidirectional audio). */
export function sdpHasAudioSendRecv(sdp: string): boolean {
    const lines = sdp.split('\r\n');
    let inAudio = false;
    for (const line of lines) {
        if (line.startsWith('m=audio')) inAudio = true;
        else if (line.startsWith('m=')) inAudio = false;
        if (inAudio && line === 'a=sendrecv') return true;
    }
    return false;
}
