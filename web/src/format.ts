export function percent(value: number | null): string {
  if (value === null) return '—';
  if (value > 0 && value < 0.001) return '<0.1%';
  return `${(value * 100).toFixed(1)}%`;
}

export function signed(value: number | null, digits = 2): string {
  if (value === null) return '—';
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

export function formatEta(seconds: number): string {
  if (seconds < 10) return `${seconds.toFixed(1)}s`;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

/** Describes which side a root value (−1…+1, from the mover's perspective) currently favors. */
export function favorsText(
  value: number | null,
  toMove: string,
  opponent: string,
): string {
  if (value === null) return 'No evaluation yet.';
  if (Math.abs(value) < 0.05) return 'Right now: close to even.';
  return `Right now: favors ${value > 0 ? toMove : opponent}.`;
}
