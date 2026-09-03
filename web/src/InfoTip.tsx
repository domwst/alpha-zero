import type { ComponentChildren, JSX } from 'preact';

interface InfoTipProps {
  id: string;
  title: string;
  triggerLabel: string;
  children: ComponentChildren;
}

export function InfoTip({ id, title, triggerLabel, children }: InfoTipProps): JSX.Element {
  return (
    <span class="info-tip">
      <button
        aria-describedby={id}
        aria-label={triggerLabel}
        class="info-tip-trigger"
        type="button"
      >
        i
      </button>
      <span class="info-tip-popover" id={id} role="tooltip">
        <strong>{title}</strong>
        {children}
      </span>
    </span>
  );
}
