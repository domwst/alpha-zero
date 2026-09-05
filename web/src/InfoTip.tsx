import type { ComponentChildren, JSX } from 'preact';

interface InfoTipProps {
  id: string;
  title: string;
  triggerLabel: string;
  /** 'tile' anchors the popover to the surrounding block; 'trigger' hugs the icon. */
  variant?: 'tile' | 'trigger';
  children: ComponentChildren;
}

export function InfoTip({
  id,
  title,
  triggerLabel,
  variant = 'tile',
  children,
}: InfoTipProps): JSX.Element {
  return (
    <span class={variant === 'trigger' ? 'info-tip info-tip--trigger' : 'info-tip'}>
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
