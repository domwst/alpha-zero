import assert from 'node:assert/strict';
import test from 'node:test';

import { DEFAULT_DISCLOSURE, parseDisclosure, reveals } from '../src/disclosure.ts';

test('your turn follows the switches and the network turn shows everything', () => {
  assert.equal(reveals(DEFAULT_DISCLOSURE, 'evaluation', true), true);
  assert.equal(reveals(DEFAULT_DISCLOSURE, 'review', true), true);
  assert.equal(reveals(DEFAULT_DISCLOSURE, 'moves', true), false);
  for (const topic of ['evaluation', 'review', 'moves'] as const)
    assert.equal(reveals(DEFAULT_DISCLOSURE, topic, false), true);
});

test('hiding on the network turn too applies the same switches there', () => {
  const strict = { ...DEFAULT_DISCLOSURE, hideOnNetworkTurn: true };
  assert.equal(reveals(strict, 'evaluation', false), true);
  assert.equal(reveals(strict, 'moves', false), false);
});

test('stored settings are validated field by field', () => {
  assert.deepEqual(parseDisclosure(null), DEFAULT_DISCLOSURE);
  assert.deepEqual(parseDisclosure('not json'), DEFAULT_DISCLOSURE);
  assert.deepEqual(parseDisclosure('"full"'), DEFAULT_DISCLOSURE);
  assert.deepEqual(
    parseDisclosure(JSON.stringify({ shown: { moves: true, review: 'yes' }, hideOnNetworkTurn: 1 })),
    { ...DEFAULT_DISCLOSURE, shown: { ...DEFAULT_DISCLOSURE.shown, moves: true } },
  );
  const off = { shown: { evaluation: false, review: false, moves: false }, hideOnNetworkTurn: true };
  assert.deepEqual(parseDisclosure(JSON.stringify(off)), off);
});
