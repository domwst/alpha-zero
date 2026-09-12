import assert from 'node:assert/strict';
import { lengthBins, nearestIndex } from '../src/experimentChartData.ts';

const frequencies = [{ moves: 9, count: 300 }, { moves: 20, count: 12 }, { moves: 21, count: 1 }, { moves: 361, count: 1 }];
for (const size of [1, 5, 10, 20]) {
  const bins = lengthBins(frequencies, size, 0, 361);
  assert.equal(bins.reduce((sum, bin) => sum + bin.count, 0), 314);
  assert.equal(bins.at(-1)?.upper, 361);
}
assert.deepEqual(lengthBins(frequencies, 20, 10, 21), [
  { lower: 10, upper: 19, count: 0 }, { lower: 20, upper: 21, count: 13 },
]);
assert.deepEqual(lengthBins(frequencies, 1, 21, 21), [{ lower: 21, upper: 21, count: 1 }]);
assert.deepEqual(lengthBins(frequencies, 1, 25, 24), []);
assert.equal(nearestIndex([55, 110, 165], 154), 2);
assert.equal(nearestIndex([55], -500), 0);
