import test from "node:test";
import assert from "node:assert/strict";
import { engines, fixture, gate } from "./helpers.mjs";

for (const engine of engines)
  test(`${engine.name()}: moving while search is pending cancels it and allows the next search`, async (t) => {
    const { page, origin } = await fixture(t, engine);
    const stale = gate();
    t.after(() => stale.release());
    let searches = 0,
      cancels = 0;
    await page.route("**/api/v1/analysis", (r) =>
      r.fulfill({
        json: {
          available: true,
          checkpoint: "fixture/checkpoint",
          max_simulations: 20000,
        },
      }),
    );
    await page.route("**/api/v1/analysis/cancel", (r) => {
      cancels++;
      return r.fulfill({ json: { cancelled: true } });
    });
    await page.route("**/api/v1/analyze/stream", async (r) => {
      searches++;
      if (searches === 1) {
        await stale.promise;
        try {
          await r.abort();
        } catch {}
        return;
      }
      await r.fulfill({
        contentType: "application/x-ndjson",
        body:
          JSON.stringify({
            result: {
              complete: true,
              searched_simulations: 1000,
              target_simulations: 1000,
              carried_visits: 0,
              elapsed_ms: 10,
              simulations_per_second: 100000,
              network_value: 0.1,
              search_value: 0.2,
              total_visits: 1000,
              moves: [],
              activations: [],
              terminal: null,
              search: {
                expanded_by_depth: [1],
                allocated_by_depth: [1],
                simulation_leaf_depth: [1],
              },
            },
          }) + "\n",
      });
    });
    const firstRequest = page.waitForRequest("**/api/v1/analyze/stream");
    await page.goto(origin + "/analyze");
    await firstRequest;
    await page
      .getByRole("button", { name: "Searching…", exact: true })
      .waitFor();
    const nextResponse = page.waitForResponse("**/api/v1/analyze/stream");
    const cell = page.locator('.board-cell[data-row="0"][data-col="0"]');
    // A single click places the move immediately; no arm-and-commit state.
    await cell.click();
    await page.getByRole("heading", { name: /Move 2/ }).waitFor();
    await nextResponse;
    await page.waitForFunction(() =>
      [...document.querySelectorAll("button")].some(
        (b) => b.textContent === "Analyze position" && b.disabled,
      ),
    );
    await page
      .getByRole("button", { name: "Analyze position", exact: true })
      .waitFor();
    assert.equal(cancels, 1);
    assert.equal(searches, 2);
    assert(
      await page
        .getByRole("button", { name: "Undo move", exact: true })
        .isEnabled(),
    );
    assert(
      await page
        .getByRole("button", { name: "Analyze position", exact: true })
        .isDisabled(),
      "target reached, not stuck busy",
    );
    stale.release();
  });
