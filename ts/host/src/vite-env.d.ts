/**
 * Minimal ambient typing for the Vite `import.meta.env` flags the host
 * shell relies on (currently just `DEV`, used to gate the preview
 * harness out of production builds).
 *
 * We deliberately do NOT `/// <reference types="vite/client" />` here:
 * vite/client also declares `*.svg` (and friends), which would clash
 * with the package's own `src/assets/svg.d.ts`. Declaring only the
 * `ImportMeta.env` shape keeps the literal `import.meta.env.DEV` form
 * (so esbuild's define still statically replaces it and tree-shakes
 * the dev-only branch) without pulling in the asset module ambients.
 */
interface ImportMetaEnv {
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly MODE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
