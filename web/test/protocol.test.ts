import assert from 'node:assert/strict';
import test from 'node:test';

import { cellKey, temperatureProbabilities } from '../src/protocol.ts';

function move(row: number, column: number, visits: number) {
  return { row, column, prior: 0.5, visits, mean_value: null };
}

test('temperature probabilities match the Rust power transform', () => {
  const leading = move(0, 0, 9);
  const trailing = move(0, 1, 1);
  const probabilities = temperatureProbabilities([leading, trailing], 0.5);

  assert.ok(Math.abs((probabilities.get(cellKey(leading)) ?? 0) - 81 / 82) < 1e-12);
  assert.ok(Math.abs((probabilities.get(cellKey(trailing)) ?? 0) - 1 / 82) < 1e-12);
});

test('zero temperature uses the same last-wins tie break as Rust', () => {
  const first = move(0, 0, 3);
  const second = move(0, 1, 3);
  const probabilities = temperatureProbabilities([first, second], 0);

  assert.equal(probabilities.get(cellKey(first)), 0);
  assert.equal(probabilities.get(cellKey(second)), 1);
});

test('an unvisited root has zero displayed move probability', () => {
  const moves = [move(0, 0, 0), move(0, 1, 0)];
  const probabilities = temperatureProbabilities(moves, 0.7);
  assert.deepEqual([...probabilities.values()], [0, 0]);
});
