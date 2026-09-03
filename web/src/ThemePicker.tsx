import type { JSX } from 'preact';
import { useEffect, useState } from 'preact/hooks';

type Theme = 'system' | 'light' | 'dark';

const THEME_KEY = 'alz-playground-theme';

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

export function ThemePicker(): JSX.Element {
  const [theme, setTheme] = useState<Theme>(storedTheme);

  useEffect(() => {
    const media = matchMedia('(prefers-color-scheme: dark)');
    const syncFromStorage = () => {
      const current = storedTheme();
      setTheme(current);
      applyTheme(current);
    };
    const handleSystemThemeChange = () => {
      if (!document.documentElement.dataset.theme) announceTheme();
    };

    syncFromStorage();
    window.addEventListener('pageshow', syncFromStorage);
    media.addEventListener('change', handleSystemThemeChange);
    return () => {
      window.removeEventListener('pageshow', syncFromStorage);
      media.removeEventListener('change', handleSystemThemeChange);
    };
  }, []);

  const selectTheme = (next: Theme) => {
    setTheme(next);
    applyTheme(next);
    try {
      if (next === 'system') localStorage.removeItem(THEME_KEY);
      else localStorage.setItem(THEME_KEY, next);
    } catch {
      // Theme state still applies when storage is unavailable.
    }
  };

  return (
    <fieldset className="theme-picker">
      <legend className="visually-hidden">Color theme</legend>
      <label title="Follow system theme">
        <input
          checked={theme === 'system'}
          name="theme"
          onChange={() => selectTheme('system')}
          type="radio"
          value="system"
        />
        <svg aria-hidden="true" viewBox="0 0 16 16">
          <rect height="8.5" rx="1.5" width="12" x="2" y="2.5" />
          <path d="M5.5 14h5M8 11v3" />
        </svg>
        <span className="visually-hidden">System</span>
      </label>
      <label title="Light theme">
        <input
          checked={theme === 'light'}
          name="theme"
          onChange={() => selectTheme('light')}
          type="radio"
          value="light"
        />
        <svg aria-hidden="true" viewBox="0 0 16 16">
          <circle cx="8" cy="8" r="3" />
          <path d="M8 1v1.5M8 13.5V15M1 8h1.5M13.5 8H15M3.05 3.05l1.06 1.06M11.89 11.89l1.06 1.06M11.89 4.11l1.06-1.06M3.05 12.95l1.06-1.06" />
        </svg>
        <span className="visually-hidden">Light</span>
      </label>
      <label title="Dark theme">
        <input
          checked={theme === 'dark'}
          name="theme"
          onChange={() => selectTheme('dark')}
          type="radio"
          value="dark"
        />
        <svg aria-hidden="true" viewBox="0 0 16 16">
          <path d="M13.5 9.5A6 6 0 0 1 6.5 2.5a6 6 0 1 0 7 7Z" />
        </svg>
        <span className="visually-hidden">Dark</span>
      </label>
    </fieldset>
  );
}
