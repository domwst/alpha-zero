export const confidenceLevels = [50, 68, 80, 90, 95, 98, 99, 99.9] as const;
export type ConfidenceLevel = typeof confidenceLevels[number];
export type MatchCounts = { wins: number; draws: number; losses: number };

// Two-sided standard-normal critical values: inverse CDF((1 + confidence) / 2).
// Generated with Python statistics.NormalDist; 95% uses the Rust report's exact constant.
const criticalValues: Record<ConfidenceLevel, number> = {
  50: 0.6744897501960817,
  68: 0.9944578832097535,
  80: 1.2815515655446008,
  90: 1.6448536269514715,
  95: 1.959963984540054,
  98: 2.3263478740408408,
  99: 2.5758293035489,
  99.9: 3.2905267314919255,
};

/** Matches src/commands/battle.rs, including its half-win approximation for draws. */
export function scoreInterval(counts: MatchCounts, confidence: ConfidenceLevel): { low: number; high: number } | null {
  if (![counts.wins, counts.draws, counts.losses].every(value => Number.isSafeInteger(value) && value >= 0)) return null;
  const games = counts.wins + counts.draws + counts.losses;
  if (!Number.isSafeInteger(games) || games === 0) return null;
  const z = criticalValues[confidence];
  if (z === undefined) return null;
  const rate = (counts.wins + counts.draws * 0.5) / games;
  const squared = z * z, denominator = 1 + squared / games;
  const center = (rate + squared / (2 * games)) / denominator;
  const margin = z * Math.sqrt((rate * (1 - rate) + squared / (4 * games)) / games) / denominator;
  return { low: Math.max(0, center - margin), high: Math.min(1, center + margin) };
}
