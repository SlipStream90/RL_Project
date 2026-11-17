// ============================================================================
// NEXUS E2E Smoke Tests
//
// Health gates on every test: fail on console error, page error, failed
// network request, or React hydration warning.
// ============================================================================

import { test, expect } from "@playwright/test";
import { attachHealthGates } from "./helpers/health";
import {
  mockStartAndStep,
  mockStep,
  mockStepWithDeal,
  mockStepDone,
  mockStepWait,
  mockStepWithdraw,
  mockQuery,
  mockQueryThin,
  mockQueryUnknown,
  mockQueryNotFound,
  mockAgents,
  mockTrajectory,
  mockHealthFail,
  mockStepFail,
  mockStepTimeout,
  mockSummaryMissing,
} from "./fixtures/responses";

const API = "/api/v1";

test.describe("smoke", () => {
  test.beforeEach(async ({ page }) => {
    const { assertNoHealthIssues } = attachHealthGates(page, test.info());
    page._healthAssert = assertNoHealthIssues;
  });

  test.afterEach(async ({ page }, testInfo) => {
    if (page._healthAssert) {
      await page._healthAssert(testInfo.title);
    }
  });

  test("loads homepage and navigates to marketplace", async ({ page }) => {
    await page.goto("/");
    // NEXUS brand is in the topbar link
    await expect(page.locator('a.brand')).toHaveText("NEXUS");
    await page.click('nav a:has-text("Marketplace")');
    await expect(page).toHaveURL("/marketplace");
  });

  test("start episode shows KPI tiles and network graph", async ({ page }) => {
    await page.route("**/api/v1/simulation/start", mockStartAndStep);
    await page.route("**/api/v1/simulation/**/step", mockStep);
    await page.route("**/api/v1/agents", mockAgents);

    await page.goto("/marketplace");
    await page.click('button:has-text("Start episode")');

    // KPI tiles visible
    await expect(page.locator(".kpi-tile")).toHaveCount(5);
    // Network graph present
    await expect(page.locator(".net-svg")).toBeVisible();
    // Agent nodes visible (uses "SELLER A" with space, not "SELLER_A")
    await expect(page.locator('.net-svg').locator('text=BUYER A')).toBeVisible();
    await expect(page.locator('.net-svg').locator('text=SELLER A')).toBeVisible();
    await expect(page.locator('.net-svg').locator('text=MARKET MANAGER')).toBeVisible();
  });

  test("step once advances the network and live feed", async ({ page }) => {
    await page.route("**/api/v1/simulation/start", mockStartAndStep);
    await page.route("**/api/v1/simulation/**/step", mockStep);
    await page.route("**/api/v1/agents", mockAgents);

    await page.goto("/marketplace");
    await page.click('button:has-text("Start episode")');

    const feedBefore = await page.locator(".feed li").count();
    await page.click('button:has-text("Step once")');
    const feedAfter = await page.locator(".feed li").count();

    expect(feedAfter).toBeGreaterThan(feedBefore);
    // Step counter updates
    await expect(page.locator('.mono').filter({ hasText: /Step 1/ })).toBeVisible();
  });

  test("auto-play starts and pauses", async ({ page }) => {
    await page.route("**/api/v1/simulation/start", mockStartAndStep);
    await page.route("**/api/v1/simulation/**/step", mockStep);
    await page.route("**/api/v1/agents", mockAgents);

    await page.goto("/marketplace");
    await page.click('button:has-text("Start episode")');

    // Click Auto-play
    await page.click('button:has-text("Auto-play")');
    await expect(page.locator('button:has-text("Pause")')).toBeVisible();

    // Click Pause
    await page.click('button:has-text("Pause")');
    await expect(page.locator('button:has-text("Auto-play")')).toBeVisible();
  });

  test("reset clears episode state", async ({ page }) => {
    await page.route("**/api/v1/simulation/start", mockStartAndStep);
    await page.route("**/api/v1/simulation/**/step", mockStep);
    await page.route("**/api/v1/agents", mockAgents);

    await page.goto("/marketplace");
    await page.click('button:has-text("Start episode")');
    await page.click('button:has-text("Step once")');
    await page.click('button:has-text("Reset")');

    // The empty state panel should show
    await expect(page.locator('text=No episode running')).toBeVisible();
  });

  test("replay scrubber loads current episode and highlights turn", async ({ page }) => {
    await page.route(`${API}/simulation/start`, mockStartAndStep);
    await page.route(`${API}/simulation/**/step`, mockStep);
    await page.route(`${API}/episodes/**/trajectory`, mockTrajectory);
    await page.route("**/api/v1/agents", mockAgents);

    await page.goto("/marketplace");
    await page.click('button:has-text("Start episode")');
    await page.click('button:has-text("Step once")');

    await page.click('nav a:has-text("Negotiation Replay")');
    await expect(page).toHaveURL("/negotiation-replay");
    // Auto-loaded trajectory
    await expect(page.locator("table")).toBeVisible();
    // Scrubber present
    await expect(page.locator('input[type="range"]')).toBeVisible();
  });

  test("explainability structured picker and free text", async ({ page }) => {
    await page.route("**/api/v1/simulation/start", mockStartAndStep);
    await page.route("**/api/v1/simulation/**/step", mockStep);
    await page.route("**/query", mockQuery);
    await page.route("**/api/v1/agents", mockAgents);

    await page.goto("/marketplace");
    await page.click('button:has-text("Start episode")');
    await page.waitForTimeout(200);

    await page.click('nav a:has-text("Explainability")');
    await expect(page).toHaveURL("/explainability");

    // Free text ask
    await page.fill('input[aria-label="Question"]', "Why did Buyer A reject?");
    await page.click('button:has-text("Ask")');
    await expect(page.locator(".evidence-item").first()).toBeVisible();
    await expect(page.locator("text=backend-reported")).toBeVisible();
  });

  test("explainability thin evidence is stated plainly", async ({ page }) => {
    await page.route("**/api/v1/simulation/start", mockStartAndStep);
    await page.route("**/api/v1/simulation/**/step", mockStep);
    await page.route("**/query", mockQueryThin);
    await page.route("**/api/v1/agents", mockAgents);

    await page.goto("/marketplace");
    await page.click('button:has-text("Start episode")');
    await page.waitForTimeout(200);

    await page.click('nav a:has-text("Explainability")');
    await page.fill('input[aria-label="Question"]', "test");
    await page.click('button:has-text("Ask")');
    await expect(page.locator("text=Thin evidence")).toBeVisible();
  });

  test("experiment lab empty state when summary.json missing", async ({ page }) => {
    await page.route("**/api/v1/simulation/start", mockStartAndStep);
    await page.route("**/api/v1/simulation/**/step", mockStep);
    // Block summary.json fetch (server component) - mock at page level
    await page.route("**/summary.json", mockSummaryMissing);
    await page.route("**/api/v1/agents", mockAgents);

    await page.goto("/marketplace");
    await page.click('button:has-text("Start episode")');

    await page.click('nav a:has-text("Experiment Lab")');
    await expect(page).toHaveURL("/experiment-lab");
    await expect(page.locator("code").filter({ hasText: "results/summary.json" })).toBeVisible();
  });

  test("how it works page loads and demos respond to sliders", async ({ page }) => {
    await page.route("**/api/v1/agents", mockAgents);
    await page.goto("/how-it-works");
    await expect(page).toHaveURL("/how-it-works");
    await expect(page.locator("h1")).toContainText("How NEXUS works");

    // Slider demo: move buyer reservation slider (keyboard works in all engines)
    const buyerSlider = page.locator('input[type="range"]').first();
    await buyerSlider.focus();
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("ArrowRight");
    // Use getByText with exact match to avoid strict mode violation
    await expect(page.getByText("Deal zone exists.", { exact: true })).toBeVisible();
  });

  test("checklist ticks from real state", async ({ page }) => {
    await page.route("**/api/v1/simulation/start", mockStartAndStep);
    await page.route("**/api/v1/simulation/**/step", mockStep);
    await page.route("**/api/v1/episodes/**/trajectory", mockTrajectory);
    await page.route("**/query", mockQuery);
    await page.route("**/api/v1/agents", mockAgents);

    await page.goto("/marketplace");
    // "Start an episode" should be ticked after start
    await page.click('button:has-text("Start episode")');
    // Wait for localStorage to be updated by the start effect
    await page.waitForTimeout(200);
    // Use the strong element specifically to avoid strict mode violation
    await expect(page.locator("span:has-text('Start an episode')")).toBeVisible();
    const firstMark = page.locator(".guide-checklist li").first().locator("span").first();
    await expect(firstMark).toContainText("✓");
  });

  test("keyboard navigation and focus visible", async ({ page }) => {
    await page.route("**/api/v1/simulation/start", mockStartAndStep);
    await page.route("**/api/v1/simulation/**/step", mockStep);
    await page.route("**/api/v1/agents", mockAgents);

    await page.goto("/marketplace");
    await page.keyboard.press("Tab");
    const focused = page.locator(":focus");
    await expect(focused).toBeVisible();
  });

  test("layout no horizontal overflow at multiple viewports", async ({ page }) => {
    await page.route("**/api/v1/simulation/start", mockStartAndStep);
    await page.route("**/api/v1/simulation/**/step", mockStep);
    await page.route("**/api/v1/agents", mockAgents);

    for (const width of [375, 768, 1280, 1920]) {
      await page.setViewportSize({ width, height: 900 });
      await page.goto("/marketplace");
      await page.click('button:has-text("Start episode")');
      const overflow = await page.evaluate(() => {
        return document.documentElement.scrollWidth > window.innerWidth;
      });
      expect(overflow).toBe(false);
    }
  });
});
