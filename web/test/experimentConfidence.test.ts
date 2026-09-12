import assert from 'node:assert/strict';
import test from 'node:test';
import { confidenceLevels, scoreInterval } from '../src/experimentConfidence.ts';

test('95% matches the existing Rust activation report', () => {
  const interval = scoreInterval({ wins: 156, losses: 144, draws: 0 }, 95)!;
  assert(Math.abs(interval.low - 0.4635710820137406) < 1e-14);
  assert(Math.abs(interval.high - 0.5759231991372775) < 1e-14);
});

test('other levels match independently calculated Wilson intervals', () => {
  const counts = { wins: 156, losses: 144, draws: 0 };
  const low = scoreInterval(counts, 80)!, high = scoreInterval(counts, 99)!;
  assert(Math.abs(low.low - 0.48302611665718403) < 1e-14);
  assert(Math.abs(low.high - 0.5567560924016947) < 1e-14);
  assert(Math.abs(high.low - 0.44607591391444407) < 1e-14);
  assert(Math.abs(high.high - 0.5930585751245312) < 1e-14);
});

test('higher confidence widens intervals at every preset, including sweeps and draws', () => {
  for (const counts of [{ wins: 600, losses: 0, draws: 0 }, { wins: 0, losses: 600, draws: 0 },
                       { wins: 144, losses: 150, draws: 6 }, { wins: 0, losses: 0, draws: 1000 }]) {
    let width = 0;
    for (const level of confidenceLevels) {
      const interval = scoreInterval(counts, level)!;
      assert(interval.low >= 0 && interval.high <= 1);
      assert(interval.high - interval.low > width);
      width = interval.high - interval.low;
    }
  }
});

test('draws count as half a win and both sides have complementary intervals', () => {
  const first = scoreInterval({ wins: 2, losses: 1, draws: 1 }, 95)!;
  const equivalent = scoreInterval({ wins: 5, losses: 3, draws: 0 }, 95)!;
  // Same score rate but different sample sizes: a draw must not become two games.
  assert(first.high - first.low > equivalent.high - equivalent.low);
  const second = scoreInterval({ wins: 1, losses: 2, draws: 1 }, 95)!;
  assert(Math.abs(first.low + second.high - 1) < 1e-14);
  assert(Math.abs(first.high + second.low - 1) < 1e-14);
  const allDraws = scoreInterval({ wins: 0, losses: 0, draws: 20 }, 95)!;
  assert(Math.abs(allDraws.low + allDraws.high - 1) < 1e-14);
});

test('empty or invalid results do not display a fabricated interval', () => {
  for (const counts of [{ wins: 0, losses: 0, draws: 0 }, { wins: -1, losses: 2, draws: 0 },
                       { wins: NaN, losses: 2, draws: 0 }, { wins: 1.5, losses: 2, draws: 0 }]) {
    assert.equal(scoreInterval(counts, 95), null);
  }
});
