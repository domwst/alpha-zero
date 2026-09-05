import assert from 'node:assert/strict';
import test from 'node:test';

import {
  assignSeriesColorIndexes,
  leadingMoves,
  trackedMoves,
} from '../src/chartSeries.ts';

function move(row: number, column: number, visits: number) {
  return { row, column, prior: 0.1, visits, mean_value: null };
}

test('series colors remain attached to cells when their ranks change', () => {
  const a = move(0, 0, 40);
  const b = move(0, 1, 30);
  const c = move(0, 2, 20);
  const d = move(0, 3, 10);
  const e = move(0, 4, 0);

  let colors = assignSeriesColorIndexes(new Map(), leadingMoves([a, b, c, d, e]));
  assert.deepEqual([...colors.values()], [0, 1, 2, 3, 4]);

  b.visits = 50;
  a.visits = 45;
  colors = assignSeriesColorIndexes(colors, leadingMoves([a, b, c, d, e]));
  assert.equal(colors.get('0:0'), 0);
  assert.equal(colors.get('0:1'), 1);

  e.visits = 60;
  colors = assignSeriesColorIndexes(colors, leadingMoves([a, b, c, d, e]));
  assert.equal(colors.get('0:4'), 4);
  assert.equal(colors.get('0:0'), 0);
  assert.equal(colors.get('0:1'), 1);
});

test('tracked moves keep early leaders that later fall behind', () => {
  const first = [move(0, 0, 30), move(0, 1, 10)];
  const second = [move(0, 0, 35), move(0, 1, 40)];
  const latest = [
    move(0, 2, 50),
    move(0, 1, 45),
    move(0, 0, 5),
    move(0, 3, 3),
    move(0, 4, 1),
    move(0, 5, 0),
  ];

  const tracked = trackedMoves([first, second, latest]);

  assert.deepEqual(
    tracked.map((entry) => `${entry.row}:${entry.column}`),
    ['0:2', '0:1', '0:0', '0:3', '0:4'],
  );
});

test('tracked moves deduplicate repeated leaders and carry latest visit counts', () => {
  const snapshots = [0, 100, 200].map((visits) => [move(3, 7, visits)]);
  const tracked = trackedMoves(snapshots);
  assert.equal(tracked.length, 1);
  assert.equal(tracked[0]!.visits, 200);
  assert.deepEqual(`${tracked[0]!.row}:${tracked[0]!.column}`, '3:7');
});
