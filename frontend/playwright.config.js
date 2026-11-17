// @ts-check
/** @type {import('@playwright/test').PlaywrightTestConfig} */
const { devices } = require('@playwright/test');

const config = {
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 3 : 1,
  reporter: process.env.CI ? "github" : "list",
  // Start both backend (FastAPI :8000) and frontend (Next.js :3000) via webServer.
  // Backend command from scripts/run_demo.ps1.
  webServer: [
    {
      command: ".venv\\Scripts\\python.exe -m uvicorn nexus.api.main:app --port 8000",
      cwd: "..",
      port: 8000,
      timeout: 120 * 1000,
      reuseExistingServer: !process.env.CI,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    },
    {
      command: "npm run dev",
      cwd: ".",
      port: 3000,
      timeout: 120 * 1000,
      reuseExistingServer: !process.env.CI,
      env: { ...process.env, NEXT_PUBLIC_API_URL: "http://localhost:8000" },
    },
  ],
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    actionTimeout: 0,
    navigationTimeout: 0,
    // Health gates: fail on console errors, page errors, failed requests.
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
    },
    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },
  ],
};

module.exports = config;
