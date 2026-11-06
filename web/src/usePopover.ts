import type { JSX } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';

/**
 * Open state for a click-toggled popover. An outside press, Escape, or focus
 * moving to an element outside the container closes it.
 */
export function usePopover() {
  const [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement | null>(null);
  const toggle = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!container.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setOpen(false);
      if (container.current?.contains(document.activeElement))
        toggle.current?.focus({ preventScroll: true });
    };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', outside);
      document.removeEventListener('keydown', escape);
    };
  }, [open]);

  // Pressing an option's text moves focus to nowhere before its click lands;
  // closing then would swallow the selection. Outside presses are handled by
  // the pointerdown listener instead.
  const onFocusOut = (event: JSX.TargetedFocusEvent<HTMLDivElement>) => {
    const next = event.relatedTarget as Node | null;
    if (next && !event.currentTarget.contains(next)) setOpen(false);
  };

  return { open, setOpen, container, toggle, onFocusOut };
}
