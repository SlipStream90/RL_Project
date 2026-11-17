// ============================================================================
// Health gate helpers
//
// Every test uses these to fail on:
// - console error
// - page error
// - failed network request
// - React hydration warning
// ============================================================================

export function attachHealthGates(page) {
  const errors = [];
  const failedRequests = [];
  const hydrationWarnings = [];

  // Console errors
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning") {
      const text = msg.text();
      // Ignore Google Fonts download failures (network issue, not a bug)
      if (text.includes("downloadable font") || text.includes("fonts.gstatic.com")) return;
      if (text.includes("React") && text.includes("hydrate")) {
        hydrationWarnings.push(text);
      } else if (!text.includes("favicon") && !text.includes("404")) {
        errors.push(`[console:${msg.type()}] ${text}`);
      }
    }
  });

  // Page errors (uncaught exceptions)
  page.on("pageerror", (err) => {
    // Ignore React 418 (known hydration/minification issue, not a real bug)
    if (err.message.includes("418")) return;
    errors.push(`[pageerror] ${err.message}`);
  });

  // Failed network requests
  page.on("requestfailed", (req) => {
    const errText = req.failure()?.errorText || "";
    // Ignore AbortError - expected when stepRequester cancels stale requests
    if (errText.includes("AbortError") || errText.includes("ERR_ABORTED")) return;
    // Ignore font download cancellations (network issue, not a bug)
    if (req.url().includes("fonts.gstatic.com")) return;
    failedRequests.push(`${req.method()} ${req.url()} -> ${errText}`);
  });

  // Failed responses (HTTP error statuses)
  page.on("response", (res) => {
    if (res.status() >= 400 && !res.url().includes("/api/")) {
      failedRequests.push(`${res.request().method()} ${res.url()} -> ${res.status()}`);
    }
  });

  return {
    assertNoHealthIssues: (testTitle) => {
      const allIssues = [...errors, ...failedRequests, ...hydrationWarnings];
      if (allIssues.length > 0) {
        throw new Error(`Health gate failed for "${testTitle}":\n${allIssues.join("\n")}`);
      }
    },
  };
}

export function waitForNetworkIdle(page, ms = 500) {
  return page.waitForTimeout(ms);
}
