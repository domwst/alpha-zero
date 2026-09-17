import test from "node:test";
import assert from "node:assert/strict";
import { engines, fixture } from "./helpers.mjs";

async function demo(t, engine, options = {}) {
  const { page, origin } = await fixture(t, engine, { service: false, ...options });
  const played = [];
  let stones = [], positionId = 1;
  await page.routeWebSocket("**/api/ws", (socket) => {
    const send = (message) => socket.send(JSON.stringify(message));
    const position = () => send({
      type: "position", position_id: positionId, ply: stones.length,
      human_color: "black", to_move: stones.length % 2 ? "white" : "black",
      stones, last_move: stones.at(-1) ?? null, outcome: null, carried_visits: 0,
    });
    socket.onMessage((raw) => {
      const message = JSON.parse(raw);
      if (message.type === "restore_game") position();
      if (message.type === "play") {
        played.push({ row: message.row, column: message.column });
        stones = [...stones, {
          row: message.row, column: message.column,
          color: stones.length % 2 ? "white" : "black",
        }];
        positionId++;
        position();
      }
      if (message.type === "start_search") {
        const legal = Array.from({ length: 361 }, (_, i) => ({
          row: Math.floor(i / 19), column: i % 19,
        })).filter((cell) => !stones.some((s) => s.row === cell.row && s.column === cell.column));
        const visits = message.simulations - 1;
        send({
          type: "search_snapshot", position_id: positionId, analysis_id: 1,
          searched_simulations: message.simulations, target_simulations: message.simulations,
          carried_visits: 0, total_visits: message.simulations,
          elapsed_ms: 100, simulations_per_second: 20000,
          network_value: 0.1, search_value: 0.2, complete: true,
          moves: legal.map((cell, i) => ({
            ...cell, prior: 1 / legal.length,
            visits: Math.floor(visits / legal.length) + (i < visits % legal.length ? 1 : 0),
            mean_value: 0.2,
          })),
        });
      }
    });
    send({
      type: "hello", protocol_version: 2, board_size: 19, compute_device: "cpu",
      checkpoint: { architecture: "fixture", epoch: 1, model_digest: "fixture" },
      max_search_simulations: 20000, default_search_simulations: 2000,
      c_puct: 1, snapshot_interval_ms: 100,
    });
  });
  await page.goto(origin);
  await page.getByText("Complete", { exact: true }).waitFor();
  const cell = (row, col) => page.locator(`.board-cell[data-row="${row}"][data-col="${col}"]`);
  const waitForMoves = async (count) => {
    await page.waitForFunction((n) => document.querySelectorAll(".board-cell .stone").length === n, count);
    assert.equal(played.length, count);
  };
  return { page, played, cell, waitForMoves };
}

for (const engine of engines) {
  test(`${engine.name()}: demo board uses primary and semantic activation`, async (t) => {
    const { page, played, cell, waitForMoves } = await demo(t, engine);
    await cell(0, 0).click({ button: "right" });
    await cell(0, 0).click({ button: "middle" });
    assert.equal(played.length, 0, "secondary buttons must not place stones");
    await cell(0, 0).click();
    await waitForMoves(1);
    await cell(0, 1).evaluate((el) => el.click());
    await waitForMoves(2);
    await cell(0, 2).focus();
    await page.keyboard.press("Enter");
    await waitForMoves(3);
    await cell(0, 3).focus();
    await page.keyboard.press("Space");
    await waitForMoves(4);
    assert.deepEqual(played, [0, 1, 2, 3].map((column) => ({ row: 0, column })));
  });

  test(`${engine.name()}: touch inspection and cancelled gestures do not play`, async (t) => {
    const { page, played, cell, waitForMoves } = await demo(t, engine, { hasTouch: true });
    await page.getByText("Always", { exact: true }).click();
    const target = cell(0, 0);
    await target.scrollIntoViewIfNeeded();
    const bounds = await target.boundingBox();
    const pointer = {
      pointerType: "touch", pointerId: 7, isPrimary: true, button: 0,
      clientX: bounds.x + bounds.width / 2, clientY: bounds.y + bounds.height / 2,
    };
    await target.dispatchEvent("pointerdown", pointer);
    await page.locator('.board-cell.is-focus[data-row="0"][data-col="0"]').waitFor();
    // Interacting outside the board dismisses the pinned inspection.
    await page
      .getByRole("button", { name: "New game" })
      .dispatchEvent("pointerdown");
    await page.waitForFunction(() => !document.querySelector(".board-cell.is-focus"));
    await target.dispatchEvent("pointerup", pointer);
    await target.dispatchEvent("click", { button: 0, detail: 1 });
    assert.equal(played.length, 0, "long press pins inspection without playing");

    await target.dispatchEvent("pointerdown", pointer);
    await target.dispatchEvent("pointermove", { ...pointer, clientX: pointer.clientX + bounds.width });
    await target.dispatchEvent("pointerup", pointer);
    await target.dispatchEvent("click", { button: 0, detail: 1 });
    assert.equal(played.length, 0, "a drag must suppress the compatibility click");

    await target.dispatchEvent("pointerdown", pointer);
    await target.dispatchEvent("pointercancel", pointer);
    await target.dispatchEvent("click", { button: 0, detail: 1 });
    assert.equal(played.length, 0, "a cancelled touch must not play");

    // Cancellation need not produce a click; it must not poison a later
    // assistive activation or the next real touch tap.
    await target.dispatchEvent("pointerdown", pointer);
    await target.dispatchEvent("pointercancel", pointer);
    await target.evaluate((el) => el.click());
    await waitForMoves(1);
    await cell(0, 1).tap();
    await waitForMoves(2);
  });

  test(`${engine.name()}: demo coordinates align with cells at desktop and narrow sizes`, async (t) => {
    const { page } = await demo(t, engine);
    for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
      await page.setViewportSize(viewport);
      const offsets = await page.evaluate(() => {
        const ranks = [...document.querySelectorAll(".board-labels-ranks span")];
        const files = [...document.querySelectorAll(".board-labels-files span")];
        const centre = (el, axis) => {
          const r = el.getBoundingClientRect();
          return axis === "x" ? r.x + r.width / 2 : r.y + r.height / 2;
        };
        return {
          ranks: ranks.map((label, row) => centre(label, "y") - centre(document.querySelector(`.board-cell[data-row="${row}"][data-col="0"]`), "y")),
          files: files.map((label, col) => centre(label, "x") - centre(document.querySelector(`.board-cell[data-row="0"][data-col="${col}"]`), "x")),
        };
      });
      for (const [axis, values] of Object.entries(offsets))
        assert(values.every((n) => Math.abs(n) < 1), `${viewport.width}px ${axis} misaligned: ${values}`);
      const square = await page.evaluate(() => {
        const cell = document.querySelector(".board-cell[data-row='0'][data-col='0']");
        const box = cell.getBoundingClientRect();
        return { w: box.width, h: box.height };
      });
      assert.ok(
        Math.abs(square.w - square.h) < 0.6,
        `${viewport.width}px: cells must be square (${square.w} x ${square.h})`,
      );
    }
  });
}
