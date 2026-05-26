/**
 * Vite library build.
 *
 * Three ESM bundles, each loadable independently from a CDN:
 *
 *   - `dist/index.js`       : "npm" entry. Re-exports the public API
 *                             (mountHost, types). Useful for IDE
 *                             autocomplete + types in app projects.
 *                             React + MUI bundled.
 *
 *   - `dist/entry/auto.js`  : "CDN host" entry. Single tag loaded
 *                             from `index.html` in standalone mode.
 *                             Imperative `mountHost(opts)` does
 *                             everything. React + MUI bundled.
 *
 *   - `dist/entry/embed.js` : "CDN embed" entry. Loaded by an
 *                             embedded app to talk to the host
 *                             over postMessage. Vanilla
 *                             (no React, no MUI) - ~5 KB gz.
 *
 * Rationale: each app Space loads `auto.js` exactly once at boot,
 * and the iframe inside the host loads `embed.js`. Both come from
 * `https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/host/dist/...`,
 * so a single npm publish pushes updates to every Space at once.
 *
 * React bundled (not external) because:
 *  1. The CDN bundle MUST be self-contained - the app's index.html
 *     can't be expected to ship React from another CDN too.
 *  2. The app's iframe has its own `window`, so a second React
 *     instance there will not collide with this one.
 *  3. Browser cache + jsdelivr edges mean the bundle is shared
 *     across all Spaces - one network fetch, used by N apps.
 */
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      // Host imports the SDK by its published package name to keep the
      // import surface symmetric with external consumers. The package
      // isn't installed in `node_modules` (it IS the workspace), so we
      // map it back to the in-tree TypeScript source. Vite compiles
      // `.ts` natively, no separate build step needed for dev.
      '@pollen-robotics/reachy-mini-sdk': resolve(__dirname, '../reachy-mini-sdk.ts'),
    },
  },
  build: {
    target: 'es2022',
    minify: 'esbuild',
    cssCodeSplit: false,
    // Host runtime sourcemaps disabled: the mountHost chunk's map is
    // ~4.6 MB (React + MUI + shell), 80% of the published tarball and
    // of the jsdelivr CDN bundle every Space loads. App authors never
    // debug into the shell itself, only their own code; if Pollen
    // Robotics need a host-side stack trace they can rebuild locally
    // with `npm run dev`. Declaration maps (tsc) stay on — they're
    // tiny and power "go to definition" jumps in consumer IDEs.
    sourcemap: false,
    emptyOutDir: true,
    // Inline ALL asset imports (SVGs, fonts) as base64 data URIs
    // into the JS bundle. Without this Vite would emit them as
    // separate files in `dist/assets/`, which would force every
    // consuming Space to either fetch them from the CDN (extra
    // round-trips) or copy them into its `public/` (couples apps
    // to host versions). Inlining gives us a single-file CDN
    // bundle: load `auto.js`, get everything.
    assetsInlineLimit: Number.MAX_SAFE_INTEGER,
    lib: {
      entry: {
        index: resolve(__dirname, 'src/index.ts'),
        'entry/auto': resolve(__dirname, 'src/entry/auto.ts'),
        'entry/embed': resolve(__dirname, 'src/entry/embed.ts'),
        // Exposed separately so app authors can `import { PROTOCOL_VERSION,
        // isProtocolMessage, ... } from '@pollen-robotics/reachy-mini-sdk/host/protocol'`
        // without paying the cost of the full host shell bundle. Stays
        // vanilla (no React, no MUI) - tree-shaken to ~1 KB.
        'lib/protocol': resolve(__dirname, 'src/lib/protocol.ts'),
      },
      formats: ['es'],
    },
    rollupOptions: {
      // No externals: bundle everything for self-contained CDN
      // delivery. The embed entry doesn't import React anyway, so
      // its tree-shaken bundle stays tiny.
      output: {
        preserveModules: false,
        entryFileNames: '[name].js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
        // Let Rollup auto-split; manual chunking was causing
        // circular dependencies between the react / mui chunks
        // because MUI's runtime imports React internally and our
        // host components import MUI. The auto-split produces one
        // larger `chunks/index-*.js` shared between auto + index,
        // which jsdelivr caches once and serves to every Space.
      },
    },
  },
});
