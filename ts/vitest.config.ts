import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    include: ['host/src/**/*.test.{ts,tsx}', 'lib/**/*.test.ts'],
  },
});
