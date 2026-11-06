/** Analysis the playground reveals, grouped by how much it gives away. */
export type DisclosureTopic = 'evaluation' | 'review' | 'moves';

export interface Disclosure {
  /** Groups shown while it is your turn. */
  shown: Record<DisclosureTopic, boolean>;
  /** Hide the same groups on the network's turn instead of showing everything. */
  hideOnNetworkTurn: boolean;
}

export interface DisclosureTopicInfo {
  id: DisclosureTopic;
  label: string;
  title: string;
  reveals: readonly string[];
  note: string;
  /** Sentence start used where the group is hidden. */
  hidden: string;
}

export const DISCLOSURE_TOPICS: readonly DisclosureTopicInfo[] = [
  {
    id: 'evaluation',
    label: "Who's ahead",
    title: "Who's ahead",
    reveals: ['Win-chance bar above the board', 'Network value V and search value Q'],
    note: 'Says nothing about where to play.',
    hidden: "Who's ahead is hidden",
  },
  {
    id: 'review',
    label: 'Last move',
    title: 'Your last move',
    reveals: [
      "Its rank among the network's candidates",
      'Raw policy and search visit share',
      'The value change it caused',
    ],
    note: 'Looks back only.',
    hidden: 'The rating of your last move is hidden',
  },
  {
    id: 'moves',
    label: 'Where to move',
    title: 'Where to move',
    reveals: [
      'Hint squares and rank badges on the board',
      'Probabilities when hovering a cell',
      'Candidate table',
      'Search history chart',
    ],
    note: "Points at the network's best move.",
    hidden: 'Move hints are hidden',
  },
];

/** Most players want to know who's ahead, but not where they should move. */
export const DEFAULT_DISCLOSURE: Disclosure = {
  shown: { evaluation: true, review: true, moves: false },
  hideOnNetworkTurn: false,
};

/** Whether a topic is shown; a finished game counts as the network's turn. */
export function reveals(
  disclosure: Disclosure,
  topic: DisclosureTopic,
  isHumanTurn: boolean,
): boolean {
  return isHumanTurn || disclosure.hideOnNetworkTurn ? disclosure.shown[topic] : true;
}

/** Reads a stored setting, falling back to the default field by field. */
export function parseDisclosure(stored: string | null): Disclosure {
  let parsed: unknown = null;
  try {
    parsed = stored === null ? null : JSON.parse(stored);
  } catch {
    // Unreadable settings fall back to the default.
  }
  const source = (typeof parsed === 'object' && parsed !== null ? parsed : {}) as Partial<Disclosure>;
  const shown = { ...DEFAULT_DISCLOSURE.shown };
  for (const { id } of DISCLOSURE_TOPICS) {
    const value = source.shown?.[id];
    if (typeof value === 'boolean') shown[id] = value;
  }
  return {
    shown,
    hideOnNetworkTurn: typeof source.hideOnNetworkTurn === 'boolean'
      ? source.hideOnNetworkTurn
      : DEFAULT_DISCLOSURE.hideOnNetworkTurn,
  };
}
