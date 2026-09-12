export type Participants = Partial<Record<'first' | 'second', { label: string; model_sha256?: string }>>;

export function networkLabel(participants: Participants | null | undefined, side: 'first' | 'second'): string {
  return participants?.[side]?.label || (side === 'first' ? 'Checkpoint A' : 'Checkpoint B');
}
