import test from "node:test";
import assert from "node:assert/strict";
import { engines, fixture } from "./helpers.mjs";

async function demo(t, engine, options = {}) {
  const { hash = "", refuseLinkedRestore = false, ...fixtureOptions } = options;
  const { page, origin } = await fixture(t, engine, { service: false, ...fixtureOptions });
  const played = [];
  const restores = [];
  const newGames = [];
  const searches = [];
  const replies = [];
  let stones = [], positionId = 1, humanColor = "black";
  await page.routeWebSocket("**/api/ws", (socket) => {
    const send = (message) => socket.send(JSON.stringify(message));
    const position = () => send({
      type: "position", position_id: positionId, ply: stones.length,
      human_color: humanColor, to_move: stones.length % 2 ? "white" : "black",
      stones, last_move: stones.at(-1) ?? null, outcome: null, carried_visits: 0,
    });
    socket.onMessage((raw) => {
      const message = JSON.parse(raw);
      if (message.type === "restore_game") {
        restores.push({ human_color: message.human_color, moves: message.moves });
        if (refuseLinkedRestore && message.moves.length > 0) {
          send({
            type: "error", code: "invalid_command",
            message: "restore contains moves after the game ended", recoverable: true,
          });
          return;
        }
        stones = message.moves.map((cell, ply) => ({ ...cell, color: ply % 2 ? "white" : "black" }));
        humanColor = message.human_color;
        positionId++;
        position();
      }
      if (message.type === "new_game") {
        newGames.push(message.human_color);
        humanColor = message.human_color;
        stones = [];
        positionId++;
        position();
      }
      if (message.type === "play") {
        played.push({ row: message.row, column: message.column });
        stones = [...stones, {
          row: message.row, column: message.column,
          color: stones.length % 2 ? "white" : "black",
        }];
        positionId++;
        position();
      }
      if (message.type === "choose_network_move") {
        replies.push(message);
        const free = Array.from({ length: 361 }, (_, i) => ({ row: Math.floor(i / 19), column: i % 19 }))
          .find((cell) => !stones.some((s) => s.row === cell.row && s.column === cell.column));
        stones = [...stones, { ...free, color: stones.length % 2 ? "white" : "black" }];
        positionId++;
        position();
      }
      if (message.type === "start_search") {
        searches.push(message.simulations);
        const legal = Array.from({ length: 361 }, (_, i) => ({
          row: Math.floor(i / 19), column: i % 19,
        })).filter((cell) => !stones.some((s) => s.row === cell.row && s.column === cell.column));
        // Zero simulations is an evaluation only: priors, no visits.
        const visits = Math.max(0, message.simulations - 1);
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
  await page.goto(origin + hash);
  await page.getByText("Complete", { exact: true }).waitFor();
  const cell = (row, col) => page.locator(`.board-cell[data-row="${row}"][data-col="${col}"]`);
  const waitForMoves = async (count) => {
    await page.waitForFunction((n) => document.querySelectorAll(".board-cell .stone").length === n, count);
    assert.equal(played.length, count);
  };
  return { page, played, newGames, searches, replies, restores, cell, waitForMoves };
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
    await page.getByRole("button", { name: "Where to move" }).click();
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

  test(`${engine.name()}: new game starts with the colour picked from its popover`, async (t) => {
    const { page, newGames } = await demo(t, engine);
    const popover = page.locator("#new-game-options");
    await page.getByRole("button", { name: "Choose your colour for the next game" }).click();
    // The radios are visually hidden, so a real press lands on the label text.
    await popover.getByText("White", { exact: true }).click();
    assert.equal(await popover.isVisible(), true, "picking a colour keeps the popover open");
    assert.equal(await popover.getByRole("radio", { name: /White/ }).isChecked(), true);
    await page.getByRole("button", { name: "New game", exact: true }).click();
    await page.getByText("Network turn", { exact: true }).waitFor();
    assert.deepEqual(newGames, ["white"]);
  });

  test(`${engine.name()}: hidden analysis keeps its place and peeks for one move`, async (t) => {
    const { page, cell, waitForMoves } = await demo(t, engine);
    const switchFor = (name) => page.getByRole("button", { name, exact: true });
    const candidates = page.locator(".inspector .concealable");
    const peek = candidates.getByRole("button", { name: "Peek this move" });
    const rows = page.locator(".candidate-table .table-move-button");
    const height = () => candidates.evaluate((el) => el.getBoundingClientRect().height);

    // By default your turn shows who is ahead but not where to move.
    assert.equal(await switchFor("Who's ahead").getAttribute("aria-pressed"), "true");
    assert.equal(await switchFor("Where to move").getAttribute("aria-pressed"), "false");
    await page.locator('.win-bar[role="img"]').waitFor();
    await peek.waitFor();
    assert.equal(await rows.count(), 0, "no candidate figures are rendered while hidden");
    const hiddenHeight = await height();

    await peek.click();
    await rows.first().waitFor();
    assert.equal(await height(), hiddenHeight, "revealing keeps the table's height");
    assert.equal(await switchFor("Where to move").getAttribute("aria-pressed"), "false", "a peek leaves the setting alone");

    // The network's turn shows everything; the next own turn hides the hints again.
    await cell(0, 0).click();
    await waitForMoves(1);
    await rows.first().waitFor();
    await cell(0, 1).click();
    await waitForMoves(2);
    await peek.waitFor();

    await switchFor("Where to move").click();
    await rows.first().waitFor();
    await page.reload();
    await rows.first().waitFor();
    assert.equal(await switchFor("Where to move").getAttribute("aria-pressed"), "true", "the switch is remembered");
  });

  test(`${engine.name()}: square-size metric switches in place and is remembered`, async (t) => {
    const { page } = await demo(t, engine);
    await page.getByRole("button", { name: "Where to move", exact: true }).click();
    const metric = (name) => page.getByRole("radio", { name, exact: true });
    const positions = () =>
      page.locator(".board-legend > *").evaluateAll((items) => items.map((item) => item.getBoundingClientRect().left));
    assert.equal(await metric("Visits").isChecked(), true, "visit fraction is the default");
    const before = await positions();
    await page.locator(".metric-switch").getByText("Probability", { exact: true }).click();
    assert.equal(await metric("Probability").isChecked(), true);
    assert.deepEqual(await positions(), before, "switching the metric moves nothing in the legend");
    await page.reload();
    await page.locator(".metric-switch").waitFor();
    assert.equal(await metric("Probability").isChecked(), true, "the metric is remembered");
  });

  test(`${engine.name()}: a zero budget replies instantly from the network policy`, async (t) => {
    const { page, searches, replies, cell, waitForMoves } = await demo(t, engine);
    await page.locator(".range-control input").first().evaluate((input) => {
      input.value = "0";
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.locator(".search-readout").getByText("Instant", { exact: true }).waitFor();
    assert.equal(await page.locator(".search-actions .button").isDisabled(), true, "nothing to run");

    await cell(9, 9).click();
    await waitForMoves(1);
    await page.getByText("Evaluated", { exact: true }).waitFor();
    assert.equal(searches.at(-1), 0, "the network's turn asks for an evaluation only");
    await page.getByRole("button", { name: "Let network choose" }).click();
    await page.waitForFunction(() => document.querySelectorAll(".board-cell .stone").length === 2);
    assert.deepEqual(replies.map((reply) => reply.instant), [true]);
    assert.equal(await page.locator('[role="alert"]').count(), 0);
  });

  test(`${engine.name()}: a game link restores the game and the address bar follows it`, async (t) => {
    const { page, restores, cell } = await demo(t, engine, { hash: "#moves=K10.K9&as=black" });
    const stoneCount = (n) =>
      page.waitForFunction((count) => document.querySelectorAll(".board-cell .stone").length === count, n);
    assert.deepEqual(restores[0], {
      human_color: "black",
      moves: [{ row: 9, column: 9 }, { row: 10, column: 9 }],
    });
    await stoneCount(2);

    await cell(0, 0).click();
    await stoneCount(3);
    await page.waitForFunction(() => location.hash === "#moves=K10.K9.A19&as=black");

    await page.evaluate(() => {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: { writeText: async (text) => { window.copied = text; } },
      });
    });
    await page.getByRole("button", { name: "Copy link" }).click();
    await page.getByText("Link copied", { exact: true }).waitFor();
    assert.equal(await page.evaluate(() => window.copied), await page.evaluate(() => location.href));

    // A link pasted into the same tab changes only the fragment, without a reload.
    await page.evaluate(() => { location.hash = "#moves=D4&as=white"; });
    await stoneCount(1);
    assert.deepEqual(restores.at(-1), { human_color: "white", moves: [{ row: 15, column: 3 }] });
  });

  test(`${engine.name()}: unusable game links fall back to a new game`, async (t) => {
    const { page, restores } = await demo(t, engine, { hash: "#moves=K10.K10" });
    await page.getByText("This link isn't a playable game: move 2, K10, is played twice. Started a new game instead.").waitFor();
    assert.deepEqual(restores.map((restore) => restore.moves.length), [0]);
    await page.waitForFunction(() => location.hash === "");
  });

  test(`${engine.name()}: a link the server refuses falls back once`, async (t) => {
    const { page, restores } = await demo(t, engine, { hash: "#moves=K10.K9", refuseLinkedRestore: true });
    await page
      .getByText("This link isn't a playable game: restore contains moves after the game ended. Started a new game instead.")
      .waitFor();
    assert.deepEqual(restores.map((restore) => restore.moves.length), [2, 0], "no retry loop");
    assert.equal(await page.locator(".board-cell .stone").count(), 0);
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
