/**
 * Which JS app this page is, for the robot's anonymous telemetry.
 *
 * Hugging Face injects `window.huggingface.variables.SPACE_ID` into every
 * Space page. The id is only sent when the Hub confirms the Space is public;
 * anything else (private, missing, offline) is sent as `private`, so a
 * private Space's name never leaves the page.
 */

const PUBLIC_CHECK_TIMEOUT_MS = 5000;

let cached: Promise<string | undefined> | null = null;

function spaceIdFromPage(): string | undefined {
    const id = (globalThis as { huggingface?: { variables?: { SPACE_ID?: unknown } } })
        .huggingface?.variables?.SPACE_ID;
    return typeof id === 'string' && id.includes('/') ? id : undefined;
}

async function check(id: string): Promise<string> {
    try {
        const path = id.split('/').map(encodeURIComponent).join('/');
        const res = await fetch(`https://huggingface.co/api/spaces/${path}`, {
            credentials: 'omit',
            signal: AbortSignal.timeout(PUBLIC_CHECK_TIMEOUT_MS),
        });
        if (!res.ok) return 'private';
        const info = (await res.json()) as { private?: unknown };
        return info.private === false ? id : 'private';
    } catch {
        return 'private';
    }
}

/**
 * The page's public Space id, `private`, or `undefined` when the page isn't
 * a Space (local dev, self-hosted). Checked once per page.
 */
export function resolvePublicSpaceId(): Promise<string | undefined> {
    if (!cached) {
        const id = spaceIdFromPage();
        cached = id ? check(id) : Promise.resolve(undefined);
    }
    return cached;
}

/** Test hook: forget the cached answer. */
export function resetPublicSpaceIdCache(): void {
    cached = null;
}
