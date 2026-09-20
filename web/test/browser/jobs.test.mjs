import test from "node:test";
import assert from "node:assert/strict";
import { engines, fixture, job, gate } from "./helpers.mjs";

for (const engine of engines) {
  test(`${engine.name()}: job pagination is above a twelve-item page`, async (t) => {
    const { page, origin } = await fixture(t, engine);
    const jobs = Array.from({ length: 13 }, (_, i) => job(`job-${i + 1}`));
    const requests = [];
    await page.route("**/api/v1/jobs?*", (route) => {
      const url = new URL(route.request().url());
      const offset = Number(url.searchParams.get("offset"));
      const limit = Number(url.searchParams.get("limit"));
      requests.push({ offset, limit });
      return route.fulfill({
        json: { jobs: jobs.slice(offset, offset + limit), total: jobs.length },
      });
    });
    await page.goto(origin + "/experiments");
    const list = page.locator(".service-job-list");
    await list.waitFor();
    assert.deepEqual(requests[0], { offset: 0, limit: 12 });
    assert.equal(await list.locator("a").count(), 12);
    assert.equal(
      await page.locator(".service-panel").first().evaluate((panel) => {
        const pager = panel.querySelector(".service-job-pagination");
        const jobs = panel.querySelector(".service-job-list");
        return !!(
          pager &&
          jobs &&
          pager.compareDocumentPosition(jobs) & Node.DOCUMENT_POSITION_FOLLOWING
        );
      }),
      true,
    );
    await page.getByText("Page 1 of 2", { exact: true }).waitFor();

    const secondPage = page.waitForRequest(
      (request) => {
        const url = new URL(request.url());
        return (
          url.pathname === "/api/v1/jobs" &&
          url.searchParams.get("offset") === "12" &&
          url.searchParams.get("limit") === "12"
        );
      },
    );
    await page.getByRole("button", { name: "Next", exact: true }).click();
    await secondPage;
    await page.getByText("Page 2 of 2", { exact: true }).waitFor();
    assert.equal(await list.locator("a").count(), 1);
  });

  test(`${engine.name()}: summaries and epochs load independently of diagnostic history`, async (t) => {
    const { page, origin } = await fixture(t, engine);
    const current = job("fixture", "comparison");
    let historyCalls = 0;
    const slow = gate();
    t.after(() => slow.release());
    await page.route("**/api/v1/jobs?*", (r) =>
      r.fulfill({ json: { jobs: [current], total: 1 } }),
    );
    await page.route("**/api/v1/jobs/fixture", (r) =>
      r.fulfill({ json: current }),
    );
    await page.route("**/api/v1/epoch-summaries?*", (r) =>
      r.fulfill({ json: { events: [] } }),
    );
    const count = { games: 1, wins: 1, losses: 0, draws: 0 },
      zero = { games: 0, wins: 0, losses: 0, draws: 0 };
    await page.route("**/api/v1/job-summary?*", (r) =>
      r.fulfill({
        json: {
          participants: null,
          benchmark: null,
          comparison: {
            games: 1,
            seat_results: {
              first_checkpoint: { first: count, second: zero },
              second_checkpoint: {
                first: zero,
                second: { ...count, wins: 0, losses: 1 },
              },
            },
            outcomes: { first_player_wins: 1, second_player_wins: 0, draws: 0 },
            lengths: {
              all: null,
              first_checkpoint: null,
              second_checkpoint: null,
              draws: null,
            },
            completion: null,
          },
        },
      }),
    );
    await page.route("**/api/v1/event-history?*", async (r) => {
      historyCalls++;
      await slow.promise;
      await r.fulfill({ json: { events: [], has_more: false } });
    });
    await page.goto(origin + "/experiments?job=fixture");
    await page.getByRole("heading", { name: "Playing strength" }).waitFor();
    assert.equal(
      historyCalls,
      0,
      "opening job details must not fetch journal pages",
    );
    const historyRequest = page.waitForRequest("**/api/v1/event-history?*");
    await page.getByText("Event history", { exact: true }).click();
    await historyRequest;
    await page
      .getByText("Loading historical events…", { exact: true })
      .waitFor();
    assert.equal(historyCalls, 1);
    assert(
      await page.getByRole("heading", { name: "Playing strength" }).isVisible(),
    );
    slow.release();
    await page
      .getByText("Loading historical events…", { exact: true })
      .waitFor({ state: "hidden" });
  });
  test(`${engine.name()}: live refreshes stay quiet and stale job responses are discarded`, async (t) => {
    const { page, origin } = await fixture(t, engine);
    const first = job("first"),
      second = job("second");
    const slow = gate();
    t.after(() => slow.release());
    let queueCalls = 0,
      epochCalls = 0;
    await page.route("**/api/v1/jobs?*", async (r) => {
      if (++queueCalls === 2) await slow.promise;
      await r.fulfill({ json: { jobs: [first, second], total: 2 } });
    });
    await page.route("**/api/v1/jobs/first", (r) => r.fulfill({ json: first }));
    await page.route("**/api/v1/jobs/second", (r) =>
      r.fulfill({ json: second }),
    );
    await page.route("**/api/v1/job-summary?*", (r) =>
      r.fulfill({
        json: { comparison: null, benchmark: null, participants: null },
      }),
    );
    await page.route("**/api/v1/epoch-summaries?*", async (r) => {
      const u = new URL(r.request().url());
      const id = u.searchParams.get("job_id");
      if (id === "first" && ++epochCalls === 2) await slow.promise;
      await r.fulfill({
        json: {
          events:
            u.searchParams.get("after") === "0"
              ? [
                  {
                    id: 1,
                    kind: "epoch_completed",
                    payload: {
                      epoch: 0,
                      training: {
                        value_loss: id === "first" ? 0.2 : 0.8,
                        policy_loss: 1,
                      },
                    },
                  },
                ]
              : [],
        },
      });
    });
    await page.goto(origin + "/experiments?job=first");
    await page
      .getByRole("heading", { name: "Epoch statistics", exact: true })
      .waitFor();
    await page.waitForTimeout(3600);
    assert.equal(
      await page.locator(".service-loading-status:visible").count(),
      0,
    );
    await page
      .locator(".service-job-list a")
      .filter({ hasText: "second" })
      .click();
    await page.getByRole("heading", { name: "second", exact: true }).waitFor();
    await page
      .locator(".chart-data tbody tr")
      .first()
      .waitFor({ state: "attached" });
    slow.release();
    await page.waitForTimeout(300);
    assert.match(await page.locator(".chart-data").textContent(), /0\.8/);
    assert.equal(
      await page.getByRole("heading", { name: "first", exact: true }).count(),
      0,
    );
  });
}
