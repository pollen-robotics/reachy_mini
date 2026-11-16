import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { resetPublicSpaceIdCache, resolvePublicSpaceId } from './client-hello.js';

const setSpaceId = (id: unknown): void => {
    (globalThis as { huggingface?: unknown }).huggingface = { variables: { SPACE_ID: id } };
};

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
    resetPublicSpaceIdCache();
    fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
    delete (globalThis as { huggingface?: unknown }).huggingface;
    vi.unstubAllGlobals();
});

describe('resolvePublicSpaceId', () => {
    it('sends nothing when the page is not a Space', async () => {
        expect(await resolvePublicSpaceId()).toBeUndefined();
        expect(fetchMock).not.toHaveBeenCalled();
    });

    it('names a Space the Hub confirms is public, checking once', async () => {
        setSpaceId('pollen-robotics/emotions');
        fetchMock.mockResolvedValue(Response.json({ id: 'pollen-robotics/emotions', private: false }));
        expect(await resolvePublicSpaceId()).toBe('pollen-robotics/emotions');
        expect(await resolvePublicSpaceId()).toBe('pollen-robotics/emotions');
        expect(fetchMock).toHaveBeenCalledOnce();
        expect(fetchMock).toHaveBeenCalledWith(
            'https://huggingface.co/api/spaces/pollen-robotics/emotions',
            expect.objectContaining({ credentials: 'omit' }),
        );
    });

    it.each([
        ['401 (private or missing)', () => Promise.resolve(new Response('{}', { status: 401 }))],
        ['private: true', () => Promise.resolve(Response.json({ private: true }))],
        ['no private field', () => Promise.resolve(Response.json({}))],
        ['network error', () => Promise.reject(new TypeError('offline'))],
    ])('reports private on %s', async (_label, reply) => {
        setSpaceId('john/secret-project');
        fetchMock.mockImplementation(reply);
        expect(await resolvePublicSpaceId()).toBe('private');
    });
});
