import test from "node:test";
import assert from "node:assert/strict";
import { engines, fixture } from "./helpers.mjs";

const HELLO = {
  type: "hello",
  protocol_version: 2,
  board_size: 19,
  compute_device: "cpu",
  checkpoint: { architecture: "fixture", epoch: 7, model_digest: "fixture" },
  max_search_simulations: 20000,
  default_search_simulations: 2000,
  c_puct: 1,
  snapshot_interval_ms: 100,
};

const STONES = [
  { row: 9, column: 9, color: "black" },
  { row: 9, column: 10, color: "white" },
  { row: 9, column: 11, color: "black" },
];

for (const engine of engines)
  test(`${engine.name()}: a dropped connection reconnects and restores the game`, async (t) => {
    const { page, origin } = await fixture(t, engine, { service: false });
    let connections = 0;
    let stones = STONES;
    let positionId = 5;
    await page.routeWebSocket("**/api/ws", (socket) => {
      connections++;
      const connection = connections;
      const send = (message) => socket.send(JSON.stringify(message));
      // Tag the handshake so the page can tell connections apart.
      send({ ...HELLO, compute_device: `cpu #${connection}` });
      const position = () =>
        send({
          type: "position",
          position_id: positionId,
          ply: stones.length,
          human_color: "black",
          to_move: stones.length % 2 ? "white" : "black",
          stones,
          last_move: stones.at(-1) ?? null,
          outcome: null,
          carried_visits: 0,
        });
      const snapshot = () =>
        send({
          type: "search_snapshot",
          position_id: positionId,
          analysis_id: 1,
          searched_simulations: 1000,
          carried_visits: 0,
          total_visits: 1000,
          target_simulations: 1000,
          elapsed_ms: 100,
          simulations_per_second: 10000,
          network_value: 0.1,
          search_value: 0.2,
          complete: true,
          moves: [
            { row: 9, column: 12, prior: 0.5, visits: 600, mean_value: 0.3 },
            { row: 3, column: 3, prior: 0.5, visits: 400, mean_value: 0.1 },
          ],
        });
      socket.onMessage((raw) => {
        const message = JSON.parse(raw);
        if (message.type === "restore_game") {
          position();
          snapshot();
        }
      });
      if (connection === 1) {
        // The server drops shortly after the session is live.
        setTimeout(() => socket.close({ code: 1006, wasClean: false }), 400);
      }
    });
    await page.goto(origin);
    const stoneCount = () =>
      page.evaluate(() => document.querySelectorAll(".board-cell .stone").length);
    await page.waitForFunction(
      () => document.querySelectorAll(".board-cell .stone").length === 3,
    );

    // The pill passes through a reconnecting state on the way back; a fast
    // recovery makes that window brief, so observe it rather than poll it.
    await page.evaluate(() => {
      window.__pillTexts = [];
      const pill = document.querySelector(".connection");
      const record = () => {
        const text = pill?.textContent ?? "";
        if (window.__pillTexts.at(-1) !== text) window.__pillTexts.push(text);
      };
      record();
      new MutationObserver(record).observe(pill, {
        childList: true,
        characterData: true,
        subtree: true,
      });
    });

    // The second connection replays the restore and the game resumes.
    await page.getByText(/cpu #2 ready/).waitFor();
    assert.equal(await stoneCount(), 3, "the game is restored without user action");
    const pillTexts = await page.evaluate(() => window.__pillTexts);
    assert.ok(
      pillTexts.some((text) => text.includes("Reconnecting")),
      `pill should show the reconnecting state, saw: ${JSON.stringify(pillTexts)}`,
    );
    assert.equal(
      await page.locator(".notice").count(),
      0,
      "no banner once the connection is back",
    );
  });

for (const engine of engines)
  test(`${engine.name()}: a server that never answers escalates to a calm banner`, async (t) => {
    const { page, origin } = await fixture(t, engine, { service: false });
    let connections = 0;
    await page.routeWebSocket("**/api/ws", (socket) => {
      connections++;
      // A dead server never completes the handshake.
      socket.onMessage(() => {});
      socket.close({ code: 1006, wasClean: false });
    });
    await page.goto(origin);

    // The first failure escalates immediately: nothing was ever connected.
    const notice = page.locator(".notice", { hasText: "Waiting for the analysis server" });
    await notice.waitFor();
    assert.ok(
      (await notice.getByRole("button", { name: "Try now" }).count()) === 1,
      "banner offers an immediate retry",
    );
    await page.getByText(/Reconnecting/).waitFor();
    const attempts = connections;
    assert.ok(attempts >= 1, "at least one retry happened");
  });
