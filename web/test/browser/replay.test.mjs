import test from "node:test";
import assert from "node:assert/strict";
import { engines, fixture } from "./helpers.mjs";
const pack = (c) =>
  Array.from({ length: 91 }, (_, i) =>
    [0, 1, 2, 3].reduce((a, j) => a | ((c[4 * i + j] || 0) << (2 * j)), 0),
  );
const cells = Array(361).fill(0),
  moves = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 2],
    [1, 2],
    [0, 3],
    [1, 3],
    [0, 4],
  ];
const plies = moves.map(([x, y], i) => {
  const n = 361 - i,
    actor = i % 2 ? "Second" : "First";
  const state = {
    state: pack(cells.map((v) => (v ? (v === (i % 2) + 1 ? 1 : 2) : 0))),
  };
  cells[x * 19 + y] = (i % 2) + 1;
  return {
    state,
    actor,
    action: { x, y },
    decision: {
      training_policy: Array(n).fill(1 / n),
      diagnostics: {
        network_prior: Array(n).fill(1 / n),
        sampling_policy: Array(n).fill(1 / n),
        root_visits: Array(n).fill(1),
        value_estimate: 0.5,
        search_value: 0.6,
        search: {
          expanded_by_depth: [1, 10],
          allocated_by_depth: [1, 20],
          simulation_leaf_depth: [0, 10],
        },
      },
    },
  };
});
const game = {
  game_type: "gomoku19_five_v1",
  game_id: "0",
  provenance: "fixture",
  policy_semantics: "raw_visits",
  model_identity: null,
  record: { plies, terminal_actor: "Second", terminal_value: -1 },
};

for (const engine of engines)
  test(`${engine.name()}: terminal replay layout`, async (t) => {
    const { page: p, origin } = await fixture(t, engine);
    await p.route("**/api/v1/session", (r) =>
      r.fulfill({ json: { authenticated: false } }),
    );
    await p.route("**/api/v1/analysis", (r) =>
      r.fulfill({ json: { available: false } }),
    );
    await p.route("**/api/v1/game-jobs", (r) =>
      r.fulfill({
        json: { jobs: [{ id: "fixture", title: "Replay fixture" }] },
      }),
    );
    await p.route("**/api/v1/games?*", (r) =>
      r.fulfill({
        json: {
          games: [
            {
              id: "0",
              epoch: "00000000",
              job_id: "fixture",
              plies: 9,
              winner: "First",
            },
          ],
          epochs: ["00000000"],
          epoch_labels: {},
          total: 1,
        },
      }),
    );
    await p.route("**/api/v1/game?*", (r) => r.fulfill({ json: game }));
    for (const width of [1440, 650]) {
      await p.setViewportSize({ width, height: 1000 });
      await p.goto(origin + "/games?job=fixture");
      await p.locator(".service-game-list button").first().click();
      const panel = p.locator(".service-panel").filter({
        has: p.getByRole("button", {
          name: "Last recorded position",
          exact: true,
        }),
      });
      const board = panel.locator(".board-with-labels");
      await board.waitFor();
      await panel
        .getByRole("button", { name: "Last recorded position", exact: true })
        .click();
      await panel
        .getByRole("button", { name: "Previous", exact: true })
        .click();
      const bounds = () =>
        board.evaluate((el) => {
          const r = el.getBoundingClientRect();
          return {
            x: r.x + scrollX,
            y: r.y + scrollY,
            width: r.width,
            height: r.height,
          };
        });
      const before = await bounds();
      await panel
        .getByRole("button", { name: "Last recorded position", exact: true })
        .click();
      await panel.getByText("Black won", { exact: true }).waitFor();
      const after = await bounds();
      for (const k of Object.keys(before))
        assert(
          Math.abs(after[k] - before[k]) < 1,
          `${engine.name()} ${width}: board ${k} shifted ${before[k]} -> ${after[k]}`,
        );
      assert.equal(
        await panel
          .getByRole("group", { name: "Distribution" })
          .locator("button:disabled")
          .count(),
        3,
      );
      assert.equal(
        await panel
          .locator(".service-position-values strong")
          .allTextContents()
          .then((a) => a.every((v) => v === "—" || v === "Unavailable")),
        true,
      );
      assert(
        await panel
          .getByRole("link", { name: "Explore this position" })
          .isVisible(),
      );
      await panel
        .getByRole("button", { name: "Previous", exact: true })
        .click();
      assert.equal(
        await panel
          .getByRole("group", { name: "Distribution" })
          .locator("button:disabled")
          .count(),
        0,
      );
      assert.deepEqual(await bounds(), before);
    }
  });
