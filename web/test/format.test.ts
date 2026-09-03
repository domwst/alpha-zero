import assert from 'node:assert/strict';
import test from 'node:test';

import { favorsText } from '../src/format.ts';

test('favorsText names the side the value favors, from the mover perspective', () => {
  assert.equal(favorsText(0.3, 'Black', 'White'), 'Right now: favors Black.');
  assert.equal(favorsText(-0.3, 'Black', 'White'), 'Right now: favors White.');
});

test('favorsText treats near-zero values as even', () => {
  assert.equal(favorsText(0.04, 'Black', 'White'), 'Right now: close to even.');
  assert.equal(favorsText(-0.04, 'Black', 'White'), 'Right now: close to even.');
});

test('favorsText handles missing evaluations', () => {
  assert.equal(favorsText(null, 'Black', 'White'), 'No evaluation yet.');
});
