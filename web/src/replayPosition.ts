export type RecordedState = { state: number[] };
export type RecordedPly = {
  state: RecordedState;
  action: { x: number; y: number } | null;
};
export function unpackPosition(packed: number[]): number[] {
  return Array.from(
    { length: 361 },
    (_, i) => (packed[Math.floor(i / 4)]! >> ((i % 4) * 2)) & 3,
  );
}
export function terminalPosition(record: {
  plies: RecordedPly[];
  terminal_state?: RecordedState | null;
}): number[] | null {
  if (record.terminal_state?.state?.length === 91) {
    const cells = unpackPosition(record.terminal_state.state);
    if (cells.every((n) => n <= 2)) return cells;
  }
  const last = record.plies.at(-1),
    action = last?.action;
  if (
    !last ||
    !action ||
    !Number.isInteger(action.x) ||
    !Number.isInteger(action.y) ||
    action.x < 0 ||
    action.x >= 19 ||
    action.y < 0 ||
    action.y >= 19 ||
    last.state.state.length !== 91
  )
    return null;
  const cells = unpackPosition(last.state.state),
    index = action.x * 19 + action.y;
  if (cells[index] !== 0 || cells.some((n) => n > 2)) return null;
  return cells.map((n, i) => (i === index ? 2 : n === 1 ? 2 : n === 2 ? 1 : 0));
}
