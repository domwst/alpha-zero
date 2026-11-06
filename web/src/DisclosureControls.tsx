import type { JSX } from 'preact';

import { type Disclosure, DISCLOSURE_TOPICS } from './disclosure';

export const EyeIcon = ({ off = false }: { off?: boolean }) => (
  <svg aria-hidden="true" className="eye-icon" viewBox="0 0 16 16">
    <path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z" />
    <circle cx="8" cy="8" r="2" />
    {off && <path d="M2.5 2.5l11 11" />}
  </svg>
);

interface DisclosureSwitchesProps {
  disclosure: Disclosure;
  onChange: (next: Disclosure) => void;
}

/** One switch per analysis group for your turn, each explained by a hover tip. */
export function DisclosureSwitches({ disclosure, onChange }: DisclosureSwitchesProps): JSX.Element {
  const nothingHidden = DISCLOSURE_TOPICS.every(({ id }) => disclosure.shown[id]);
  return (
    <div className="overlay-control-group">
      <b className="control-title" id="disclosure-title">Show on your turn</b>
      <div aria-labelledby="disclosure-title" className="disclosure-switches" role="group">
        {DISCLOSURE_TOPICS.map((topic) => {
          const shown = disclosure.shown[topic.id];
          return (
            <span className="disclosure-switch has-tip" key={topic.id}>
              <button
                aria-describedby={`disclosure-tip-${topic.id}`}
                aria-pressed={shown}
                onClick={() => onChange({
                  ...disclosure,
                  shown: { ...disclosure.shown, [topic.id]: !shown },
                })}
                type="button"
              >
                <EyeIcon off={!shown} />
                {topic.label}
              </button>
              <span className="hover-tip" id={`disclosure-tip-${topic.id}`} role="tooltip">
                <strong>{topic.title}</strong>
                <ul>
                  {topic.reveals.map((item) => <li key={item}>{item}</li>)}
                </ul>
                <span>{topic.note}</span>
              </span>
            </span>
          );
        })}
      </div>
      <label className="disclosure-also">
        <input
          checked={disclosure.hideOnNetworkTurn && !nothingHidden}
          disabled={nothingHidden}
          onChange={(event) => onChange({ ...disclosure, hideOnNetworkTurn: event.currentTarget.checked })}
          type="checkbox"
        />
        Hide on the network's turn too
      </label>
    </div>
  );
}

interface ConcealedProps {
  message: string;
  onPeek: () => void;
}

/** Covers a same-size placeholder of hidden content and offers a one-move peek. */
export function ConcealedVeil({ message, onPeek }: ConcealedProps): JSX.Element {
  return (
    <div className="concealed-veil reveal">
      <EyeIcon off />
      <span>{message}</span>
      <button className="text-button" onClick={onPeek} type="button">
        Peek this move
      </button>
    </div>
  );
}

/** Single-line form of the veil for rows that must keep their height. */
export function ConcealedNote({ message, onPeek }: ConcealedProps): JSX.Element {
  return (
    <span className="concealed-note reveal">
      <EyeIcon off />
      {message}
      <button className="text-button" onClick={onPeek} type="button">
        Peek
      </button>
    </span>
  );
}
