import assert from 'node:assert/strict';
import test from 'node:test';

import {
  demoPositionUrl,
  gameFragment,
  parseGameFragment,
  parseMoveName,
} from '../src/protocol.ts';

const K10 = { row: 9, column: 9 };
const K9 = { row: 10, column: 9 };
const A19 = { row: 0, column: 0 };
const T1 = { row: 18, column: 18 };

test('moves use the board labels, skipping I', () => {
  assert.deepEqual(parseMoveName('K10'), K10);
  assert.deepEqual(parseMoveName('a19'), A19);
  assert.deepEqual(parseMoveName('T1'), T1);
  assert.deepEqual(parseMoveName('J5'), { row: 14, column: 8 });
  for (const bad of ['I5', 'U1', 'K0', 'K20', 'K', '10', 'K10x']) assert.equal(parseMoveName(bad), null, bad);
});

test('a game round-trips through its fragment', () => {
  const fragment = gameFragment([K10, K9, A19], 'white');
  assert.equal(fragment, 'moves=K10.K9.A19&as=white');
  assert.deepEqual(parseGameFragment(`#${fragment}`), {
    ok: true,
    link: { moves: [K10, K9, A19], humanColor: 'white' },
  });
});

test('a fresh game as Black needs no fragment', () => {
  assert.equal(gameFragment([], 'black'), '');
  assert.equal(parseGameFragment(''), null);
  assert.equal(parseGameFragment('#section'), null);
  assert.deepEqual(parseGameFragment('#as=white'), { ok: true, link: { moves: [], humanColor: 'white' } });
});

test('without a side the visitor plays the side to move', () => {
  assert.deepEqual(parseGameFragment('#moves=K10'), { ok: true, link: { moves: [K10], humanColor: 'white' } });
  assert.deepEqual(parseGameFragment('#moves=k10,k9&as=green'), {
    ok: true,
    link: { moves: [K10, K9], humanColor: 'black' },
  });
});

test('unusable links say why', () => {
  assert.deepEqual(parseGameFragment('#moves=K10.Z3'), { ok: false, reason: 'move 2, “Z3”, is not a board point' });
  assert.deepEqual(parseGameFragment('#moves=K10.K9.k10'), { ok: false, reason: 'move 3, K10, is played twice' });
});

test('dashboard links open the demo at a position for the side to move', () => {
  assert.equal(
    demoPositionUrl('https://gomoku.example/', [K10, K9, A19]),
    'https://gomoku.example/#moves=K10.K9.A19&as=white',
  );
  assert.equal(demoPositionUrl('https://gomoku.example/play?x=1', []), 'https://gomoku.example/play?x=1');
});
