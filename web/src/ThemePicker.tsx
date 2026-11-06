import type { JSX } from 'preact';
import { useEffect, useLayoutEffect, useRef, useState } from 'preact/hooks';

type Theme = 'system' | 'light' | 'dark';

const THEME_KEY = 'alz-playground-theme';
const THEME_TRANSITION_CLASS = 'theme-transition';
const THEME_TRANSITION_FALLBACK_MS = 300;

let themeTransitionTimer: ReturnType<typeof setTimeout> | undefined;

/** The fade duration is owned by CSS; JS only reads it to know when the
 *  transition class can be removed. Reading the computed style also flushes
 *  the newly added class into the before-change style, so the transition
 *  reliably starts when the theme values flip. */
function themeTransitionDurationMs(root: HTMLElement): number {
  const raw = getComputedStyle(root)
    .getPropertyValue('--ds-theme-transition-duration')
    .trim();
  const parsed = Number.parseFloat(raw);
  if (!Number.isFinite(parsed)) return THEME_TRANSITION_FALLBACK_MS;
  return raw.endsWith('ms') ? parsed : parsed * 1000;
}

/** Cross-fades the theme; the class enables color transitions and is removed
 *  once the fade can no longer be in flight. */
function withThemeTransition(apply: () => void): void {
  const root = document.documentElement;
  if (!root.classList.contains('motion-on')) {
    apply();
    return;
  }
  root.classList.add(THEME_TRANSITION_CLASS);
  const durationMs = themeTransitionDurationMs(root);
  apply();
  clearTimeout(themeTransitionTimer);
  themeTransitionTimer = setTimeout(() => {
    root.classList.remove(THEME_TRANSITION_CLASS);
  }, durationMs + 50);
}

function storedTheme(): Theme {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return value === 'light' || value === 'dark' ? value : 'system';
  } catch {
    return 'system';
  }
}

function effectiveTheme(): Exclude<Theme, 'system'> {
  const pinned = document.documentElement.dataset.theme;
  if (pinned === 'light' || pinned === 'dark') return pinned;
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function announceTheme(): void {
  document.dispatchEvent(
    new CustomEvent('themechange', { detail: effectiveTheme() }),
  );
}

function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  if (theme === 'system') delete root.dataset.theme;
  else root.dataset.theme = theme;
  announceTheme();
}

const MonitorIcon = () => (
  <svg aria-hidden="true" viewBox="0 0 16 16">
    <rect height="8.5" rx="1.5" width="12" x="2" y="2.5" />
    <path d="M5.5 14h5M8 11v3" />
  </svg>
);
const SunIcon = () => (
  <svg aria-hidden="true" viewBox="0 0 16 16">
    <circle cx="8" cy="8" r="3" />
    <path d="M8 1v1.5M8 13.5V15M1 8h1.5M13.5 8H15M3.05 3.05l1.06 1.06M11.89 11.89l1.06 1.06M11.89 4.11l1.06-1.06M3.05 12.95l1.06-1.06" />
  </svg>
);
const MoonIcon = () => (
  <svg aria-hidden="true" viewBox="0 0 16 16">
    <path d="M13.5 9.5A6 6 0 0 1 6.5 2.5a6 6 0 1 0 7 7Z" />
  </svg>
);

const OPTIONS: { value: Theme; label: string; Icon: () => JSX.Element }[] = [
  { value: 'system', label: 'System', Icon: MonitorIcon },
  { value: 'light', label: 'Light', Icon: SunIcon },
  { value: 'dark', label: 'Dark', Icon: MoonIcon },
];

/** A compact trigger showing the current theme; hovering or activating it
 *  opens a tray with the three choices. Mouse hover closes when the pointer
 *  leaves the trigger/tray path; touch and keyboard pin it open until
 *  dismissed. Ported from the instructions-reference header. */
export function ThemePicker(): JSX.Element {
  const [theme, setTheme] = useState<Theme>(storedTheme);
  const [open, setOpen] = useState(false);
  const control = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const tray = useRef<HTMLDivElement>(null);
  const pinned = useRef(false);
  const focusOnOpen = useRef(false);
  const selected = OPTIONS.find((option) => option.value === theme) ?? OPTIONS[0]!;

  useEffect(() => {
    const media = matchMedia('(prefers-color-scheme: dark)');
    const syncFromStorage = () => {
      const current = storedTheme();
      setTheme(current);
      applyTheme(current);
    };
    const handleSystemThemeChange = () => {
      if (!document.documentElement.dataset.theme) withThemeTransition(announceTheme);
    };

    syncFromStorage();
    window.addEventListener('pageshow', syncFromStorage);
    media.addEventListener('change', handleSystemThemeChange);
    return () => {
      window.removeEventListener('pageshow', syncFromStorage);
      media.removeEventListener('change', handleSystemThemeChange);
    };
  }, []);

  useLayoutEffect(() => {
    if (open && focusOnOpen.current) {
      focusOnOpen.current = false;
      control.current?.querySelector<HTMLInputElement>('input:checked')?.focus();
    }
  }, [open]);

  function close(restoreFocus = false) {
    pinned.current = false;
    focusOnOpen.current = false;
    if (restoreFocus && control.current?.contains(document.activeElement))
      trigger.current?.focus({ preventScroll: true });
    setOpen(false);
  }

  function withinHoverPath(x: number, y: number) {
    const from = trigger.current?.getBoundingClientRect();
    const to = tray.current?.getBoundingClientRect();
    if (!from || !to) return false;
    if (y >= from.top && y <= from.bottom && x >= from.left && x <= from.right) return true;
    if (y >= to.top && y <= to.bottom && x >= to.left && x <= to.right) return true;
    if (y < from.top || y > to.top) return false;
    // Keep diagonal paths from the whole trigger into the tray open, without
    // intercepting clicks on neighboring header controls with an invisible layer.
    const progress = (y - from.top) / (to.top - from.top);
    const left = from.left + (to.left - from.left) * progress;
    const right = from.right + (to.right - from.right) * progress;
    return x >= left && x <= right;
  }

  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!control.current?.contains(event.target as Node)) close(true);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !event.defaultPrevented) {
        event.preventDefault();
        close(true);
      }
    };
    const move = (event: PointerEvent) => {
      if (
        event.pointerType === 'mouse' &&
        !pinned.current &&
        !control.current?.contains(event.target as Node) &&
        !withinHoverPath(event.clientX, event.clientY)
      )
        close(true);
    };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', escape);
    document.addEventListener('pointermove', move);
    return () => {
      document.removeEventListener('pointerdown', outside);
      document.removeEventListener('keydown', escape);
      document.removeEventListener('pointermove', move);
    };
  }, [open]);

  const selectTheme = (next: Theme) => {
    setTheme(next);
    withThemeTransition(() => applyTheme(next));
    try {
      if (next === 'system') localStorage.removeItem(THEME_KEY);
      else localStorage.setItem(THEME_KEY, next);
    } catch {
      // Theme state still applies when storage is unavailable.
    }
  };

  return (
    <div
      className="theme-control"
      ref={control}
      onPointerEnter={(event) => {
        if (event.pointerType === 'mouse') setOpen(true);
      }}
      onPointerLeave={(event) => {
        if (
          event.pointerType === 'mouse' &&
          !pinned.current &&
          !withinHoverPath(event.clientX, event.clientY)
        )
          close(true);
      }}
      onFocusOut={(event) => {
        const container = event.currentTarget as HTMLElement;
        if (!container.contains(event.relatedTarget as Node | null)) close();
      }}
    >
      <button
        aria-controls="theme-options"
        aria-expanded={open}
        aria-label={`Color theme: ${selected.label}`}
        className="theme-trigger"
        data-value={theme}
        id="theme"
        onClick={() => {
          if (open && pinned.current) close();
          else {
            pinned.current = true;
            setOpen(true);
          }
        }}
        onKeyDown={(event) => {
          if (open) pinned.current = true;
          if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
          event.preventDefault();
          pinned.current = true;
          if (open) control.current?.querySelector<HTMLInputElement>('input:checked')?.focus();
          else {
            focusOnOpen.current = true;
            setOpen(true);
          }
        }}
        ref={trigger}
        title={`Color theme: ${selected.label}`}
        type="button"
      >
        <selected.Icon />
      </button>
      <div className="theme-tray" hidden={!open} id="theme-options" ref={tray}>
        <fieldset
          className="theme-segments"
          onKeyDown={() => {
            pinned.current = true;
          }}
        >
          <legend className="visually-hidden">Color theme</legend>
          {OPTIONS.map(({ value, label, Icon }) => (
            <label className="theme-segment" key={value} title={`${label} theme`}>
              <input
                checked={theme === value}
                className="theme-radio"
                name="theme"
                onChange={() => selectTheme(value)}
                onPointerDown={(event) => {
                  // Mouse selection should still close when the pointer leaves.
                  // Touch and keyboard interactions stay open until dismissed.
                  pinned.current = event.pointerType !== 'mouse';
                }}
                type="radio"
                value={value}
              />
              <Icon />
              <span className="visually-hidden">{label}</span>
            </label>
          ))}
        </fieldset>
      </div>
    </div>
  );
}
