import { resolve } from 'node:path';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    alias: {
      // Same alias as host/vite.config.ts: host sources import the SDK by
      // its package name, which node would otherwise resolve to ./dist -
      // the output of the LAST build, not the sources under test.
      '@pollen-robotics/reachy-mini-sdk': resolve(__dirname, 'reachy-mini-sdk.ts'),
    },
  },
  test: {
    environment: 'jsdom',
    include: ['host/src/**/*.test.{ts,tsx}', 'lib/**/*.test.ts'],
  },
});
