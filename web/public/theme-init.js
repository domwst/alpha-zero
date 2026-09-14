// Apply the saved theme before first paint; kept external for a strict script CSP.
(() => {
  try {
    const saved = localStorage.getItem('alz-playground-theme');
    if (saved === 'light' || saved === 'dark')
      document.documentElement.dataset.theme = saved;
  } catch { /* storage unavailable: follow the host theme */ }
})();
