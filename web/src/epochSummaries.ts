type EpochEvent = { id: number; payload: Record<string, unknown> };

export function mergeEpochSummaries<T extends EpochEvent>(current: T[], incoming: T[]) {
  const byId = new Map(current.map((event) => [event.id, event]));
  for (const event of incoming) byId.set(event.id, event);
  return [...byId.values()].sort((a, b) => {
    const aEpoch = Number(a.payload.epoch), bEpoch = Number(b.payload.epoch);
    return aEpoch - bEpoch || a.id - b.id;
  });
}
