export type Participants = Partial<Record<'first' | 'second', { label: string; model_sha256?: string }>>;

export function networkLabel(participants: Participants | null | undefined, side: 'first' | 'second'): string {
  return participants?.[side]?.label || (side === 'first' ? 'Checkpoint A' : 'Checkpoint B');
}

/** A short alias for tables: the side letter plus the epoch when the label
 *  carries one ("A · E80"); the full label belongs in a title or legend. */
export function shortNetworkLabel(participants: Participants | null | undefined, side: 'first' | 'second'): string {
  const letter = side === 'first' ? 'A' : 'B';
  const label = participants?.[side]?.label;
  if (!label) return `Checkpoint ${letter}`;
  // The checkpoint's own epoch is the last one named; titles such as
  // "continuation to epoch 80 · epoch 73" mention an earlier target first.
  const epochs = [...label.matchAll(/epoch\s+(\d+)/gi)];
  const epoch = epochs.at(-1);
  if (epoch) return `${letter} · E${epoch[1]}`;
  return `${letter} · ${label.length > 26 ? `${label.slice(0, 24)}…` : label}`;
}
