import test from "node:test";
import assert from "node:assert/strict";
import { engines, fixture, gate } from "./helpers.mjs";

for (const engine of engines)
  test(`${engine.name()}: layer/channel changes preserve the activation board and selection`, async (t) => {
    const { page, origin } = await fixture(t, engine);
    const delayed = gate();
    t.after(() => delayed.release());
    const names = ["trunk.block_0.conv1", "trunk.block_0.conv2"];
    const result = (requested) => ({
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
        terminal: null,
        search: {
          expanded_by_depth: [1],
          allocated_by_depth: [1],
          simulation_leaf_depth: [1],
        },
        activations: requested
          ? names.map((name) => ({
              name,
              shape: [1, 2, 19, 19],
              axes: ["batch", "channel", "row", "column"],
              values:
                name === requested
                  ? Array.from({ length: 722 }, (_, i) => (i % 17) / 100)
                  : [],
            }))
          : [],
      },
    });
    await page.route("**/api/v1/analysis", (r) =>
      r.fulfill({
        json: {
          available: true,
          checkpoint: "fixture",
          max_simulations: 20000,
        },
      }),
    );
    await page.route("**/api/v1/analyze/stream", (r) =>
      r.fulfill({
        contentType: "application/x-ndjson",
        body: JSON.stringify(result(null)) + "\n",
      }),
    );
    await page.route("**/api/v1/analyze", async (r) => {
      const layer = r.request().postDataJSON().layers[0];
      if (layer === names[1]) await delayed.promise;
      await r.fulfill({ json: result(layer) });
    });
    await page.goto(origin + "/analyze");
    await page
      .getByRole("button", { name: "Analyze position", exact: true })
      .waitFor();
    await page.getByLabel("Capture activations").check();
    await page.locator(".service-activation").waitFor();
    const channel = page.getByRole("slider", { name: "Channel", exact: true });
    await channel.focus();
    await page.keyboard.press("ArrowRight");
    assert.equal(await channel.inputValue(), "1");
    const bounds = () =>
      page.locator(".service-activation").evaluate((el) => {
        const r = el.getBoundingClientRect();
        return { y: r.y + scrollY, height: r.height, width: r.width };
      });
    const before = await bounds();
    const request = page.waitForRequest("**/api/v1/analyze");
    await page.getByLabel("Layer", { exact: true }).selectOption("1");
    await request;
    await page.locator('.service-activation[aria-busy="true"]').waitFor();
    assert.deepEqual(await bounds(), before);
    assert.equal(await channel.inputValue(), "1");
    delayed.release();
    await page.locator('.service-activation[aria-busy="false"]').waitFor();
    assert.deepEqual(await bounds(), before);
    assert.equal(await channel.inputValue(), "1");
    await page.getByLabel("Analyze after each move").uncheck();
    const cell = page.locator('.board-cell[data-row="0"][data-col="0"]');
    await cell.click();
    await cell.click();
    await page.getByRole("heading", { name: /Move 2/ }).waitFor();
    assert.equal(
      await page.getByLabel("Layer", { exact: true }).inputValue(),
      "1",
    );
    assert.equal(await channel.inputValue(), "1");
  });
