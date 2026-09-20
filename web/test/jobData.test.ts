import assert from 'node:assert/strict';
import test from 'node:test';
import { mergeEpochSummaries } from '../src/epochSummaries.ts';
import type { Event } from '../src/serviceApi.ts';

const epoch = (id: number, value: number): Event => ({
  id,
  kind: 'epoch_completed',
  payload: { epoch: value },
  time: id,
});

test('continuation history is de-duplicated and sorted by epoch, not insertion id', () => {
  const current = [epoch(10, 3), epoch(11, 4)];
  const inherited = [epoch(12, 0), epoch(13, 1), epoch(14, 2), epoch(11, 4)];
  const merged = mergeEpochSummaries(current, inherited);
  assert.deepEqual(merged.map((event) => event.payload.epoch), [0, 1, 2, 3, 4]);
  assert.equal(merged.length, 5);
});
