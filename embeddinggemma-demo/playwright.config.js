import { defineConfig, devices } from '@playwright/test';

const PORT = Number(process.env.PORT ?? 5173);

export default defineConfig({
  testDir: 'tests/e2e',
  outputDir: 'test-results',
  // One browser at a time: every test loads a 200 MB - 1.2 GB model and the timings
  // are only meaningful when nothing else competes for the CPU.
  workers: 1,
  fullyParallel: false,
  timeout: 15 * 60 * 1000,
  expect: { timeout: 10_000 },
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    viewport: { width: 1280, height: 900 },
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 900 },
        // DEVICE=webgpu runs need WebGPU, which headless Chromium only offers behind this flag (a software
        // adapter when there is no GPU, as in CI: fine for checking results, not for timing them).
        launchOptions: process.env.DEVICE === 'webgpu' ? { args: ['--enable-unsafe-webgpu'] } : {},
      },
    },
  ],
  webServer: {
    command: 'node server.mjs',
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    env: { PORT: String(PORT) },
  },
});
