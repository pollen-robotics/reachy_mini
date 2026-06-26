/**
 * SPA build of the host **dev harness**, for deployment as a static
 * Hugging Face Space.
 *
 * The default `vite.config.ts` is a *library* build (emits the
 * `dist/*.js` CDN bundles and ignores `index.html`). This config does
 * the opposite: a plain single-page-app build with `index.html` as the
 * entry, so the harness can be served as static files on a Space and
 * the host's OAuth flow exercised in real conditions (real redirect,
 * real `window.huggingface.variables.OAUTH_CLIENT_ID` injected by the
 * Space).
 *
 * It deliberately reuses the SAME source the local `npm run dev`
 * harness uses (`index.html` + `dev/main.ts`), so what we test on the
 * Space is byte-for-byte the host shell we ship - no CDN, no published
 * npm version in the loop.
 *
 * Build:   vite build --config vite.harness.config.ts   (from `host/`)
 * Output:  host/dist-harness/   →  push as the Space root.
 */
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  // Relative asset URLs so the bundle works regardless of the path the
  // Space serves it from (root in practice, but this keeps us safe).
  base: './',
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      // Same in-tree SDK source the library build and dev server use,
      // so the harness Space runs our working-copy host, not a
      // published package.
      '@pollen-robotics/reachy-mini-sdk': resolve(__dirname, '../reachy-mini-sdk.ts'),
    },
  },
  build: {
    target: 'es2022',
    outDir: 'dist-harness',
    emptyOutDir: true,
    sourcemap: false,
  },
});
