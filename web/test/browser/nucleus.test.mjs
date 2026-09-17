import test from "node:test";
import assert from "node:assert/strict";
import { engines, fixture } from "./helpers.mjs";

for (const engine of engines) {
  test(`${engine.name()}: nucleus filtering is opt-in and historical cutoffs remain usable`, async (t) => {
    const { page, origin } = await fixture(t, engine);
    await page.route("**/api/v1/jobs?*", (r) => r.fulfill({ json: { jobs: [], total: 0 } }));
    await page.route("**/api/v1/artifacts", (r) => r.fulfill({ json: { artifacts: [] } }));
    await page.route("**/api/v1/schema", (r) => r.fulfill({ json: { kinds: {
      self_play: { device: ["cpu", "cuda"], "top-p": "number" },
    } } }));
    const submitted = [];
    await page.route("**/api/v1/commands", (r) => {
      submitted.push(r.request().postDataJSON().request.spec);
      return r.fulfill({ json: { job_id: "fixture" } });
    });
    await page.goto(origin + "/experiments");
    await page.getByRole("button", { name: "Schedule job", exact: true }).click();
    const cutoff = page.locator("#job-top-p");
    await cutoff.waitFor({ state: "attached" });
    assert.equal(await cutoff.isVisible(), false);
    const created = page.waitForResponse("**/api/v1/commands");
    await page.getByRole("button", { name: "Add to queue", exact: true }).click();
    await created;
    assert.equal(submitted[0].options["top-p"], undefined, "uses native default 1.0");
    await page.getByRole("button", { name: "Schedule job", exact: true }).click();
    await page.getByText("Advanced experimental settings", { exact: true }).click();
    assert.equal(await cutoff.getAttribute("placeholder"), "1.0 (disabled)");
    await cutoff.fill("0");
    assert.equal(await cutoff.evaluate((el) => el.checkValidity()), false);
    await cutoff.fill("0.95");
    assert.equal(await cutoff.evaluate((el) => el.checkValidity()), true);
    // Collapsing the section must not discard an explicit research setting.
    await page.getByText("Advanced experimental settings", { exact: true }).click();
    const overridden = page.waitForResponse("**/api/v1/commands");
    await page.getByRole("button", { name: "Add to queue", exact: true }).click();
    await overridden;
    assert.equal(Number(submitted[1].options["top-p"]), 0.95);
  });
}
