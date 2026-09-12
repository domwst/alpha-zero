export type LengthFrequency = { moves: number; count: number };
export type LengthBin = { lower: number; upper: number; count: number };

export function lengthBins(values: LengthFrequency[], size: number, minimum: number, maximum: number): LengthBin[] {
  if (!Number.isInteger(size) || size < 1 || maximum < minimum) return [];
  const start = Math.floor(minimum / size) * size;
  const bins = Array.from({ length: Math.floor((maximum - start) / size) + 1 }, (_, i) => ({
    lower: Math.max(minimum, start + i * size), upper: Math.min(maximum, start + (i + 1) * size - 1), count: 0,
  }));
  for (const value of values) {
    if (value.moves < minimum || value.moves > maximum) continue;
    const bin = bins[Math.floor((value.moves - start) / size)];
    if (bin) bin.count += value.count;
  }
  return bins;
}

export function nearestIndex(values: number[], target: number): number {
  return values.reduce((best, value, index) => Math.abs(value - target) < Math.abs(values[best]! - target) ? index : best, 0);
}
