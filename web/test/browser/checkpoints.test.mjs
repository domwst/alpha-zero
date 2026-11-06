import test from "node:test";
import assert from "node:assert/strict";
import { engines, fixture } from "./helpers.mjs";

for (const engine of engines) {
  test(`${engine.name()}: saved checkpoints refresh without losing the selected input`, async (t) => {
    const { page, origin } = await fixture(t, engine);
    await page.clock.install();
    await page.route("**/api/v1/jobs?*", (r) =>
      r.fulfill({ json: { jobs: [], total: 0 } }),
    );
    await page.route("**/api/v1/schema", (r) =>
      r.fulfill({ json: { kinds: {
        self_play: { device: ["cpu", "cuda"] },
        comparison: { device: ["cpu", "cuda"] },
      } } }),
    );
    let requests = 0;
    const artifact = (epoch) => ({
      id: `checkpoint-${epoch}`, job_id: "sharp", kind: "checkpoint",
      job_title: "Sharp Temp", metadata: JSON.stringify({ epoch }),
    });
    await page.route("**/api/v1/artifacts", (r) =>
      r.fulfill({ json: { artifacts: ++requests === 1
        ? [artifact(0)] : [artifact(1), artifact(0)] } }),
    );
    await page.goto(origin + "/experiments");
    await page.getByRole("button", { name: "Schedule job", exact: true }).click();
    await page.locator('.service-job-form select').first().selectOption("comparison");
    const first = page.locator('.service-input-grid select').first();
    await first.selectOption("checkpoint-0");
    await page.getByText("Loading saved checkpoints…", { exact: true }).waitFor({ state: "hidden" });
    const refreshed = page.waitForResponse("**/api/v1/artifacts");
    await page.clock.fastForward(15000);
    await refreshed;
    await page.waitForFunction(() => document.querySelector('.service-input-grid option[value="checkpoint-1"]'));
    assert.equal(await first.inputValue(), "checkpoint-0");
    assert.equal(await first.locator('option[value="checkpoint-1"]').textContent(), "Sharp Temp · epoch 2");
    assert.equal(await page.getByText("Loading saved checkpoints…", { exact: true }).count(), 0);
    assert.equal(requests, 2);
  });
}
