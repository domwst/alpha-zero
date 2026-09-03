import assert from 'node:assert/strict';
import test from 'node:test';

import { revealsMoveGuidance } from '../src/policyVisibility.ts';

test('move guidance visibility follows all three modes', () => {
  assert.equal(revealsMoveGuidance('always', false), true);
  assert.equal(revealsMoveGuidance('always', true), true);
  assert.equal(revealsMoveGuidance('network_turn', false), false);
  assert.equal(revealsMoveGuidance('network_turn', true), true);
  assert.equal(revealsMoveGuidance('never', false), false);
  assert.equal(revealsMoveGuidance('never', true), false);
});
