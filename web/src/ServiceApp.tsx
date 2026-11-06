import { useEffect, useState } from "preact/hooks";
import { ThemePicker } from "./ThemePicker";
import { JobWorkspace } from "./JobWorkspace";
import { GameExplorer } from "./GameExplorer";
import { Overview } from "./Overview";
import { api, type Session } from "./serviceApi";
import "./service.css";

export const SITE_NAME = "AlphaZero Research";
const PAGE_TITLES: Record<string, string> = {
  "/": "Overview",
  "/experiments": "Experiments",
  "/games": "Recorded games",
  "/analyze": "Position analysis",
};

export function ServiceApp({ initialSession }: { initialSession: Session }) {
  const [session, setSession] = useState(initialSession),
    [login, setLogin] = useState(false),
    [password, setPassword] = useState(""),
    [error, setError] = useState("");
  const [analysisAvailable, setAnalysisAvailable] = useState<boolean | null>(null);
  const path = location.pathname;
  useEffect(() => {
    document.title = `${PAGE_TITLES[path] ?? "Page not found"} · ${SITE_NAME}`;
    document
      .querySelector('meta[name="description"]')
      ?.setAttribute(
        "content",
        "Training runs, checkpoint comparisons and recorded self-play games from an AlphaZero Gomoku research programme.",
      );
  }, [path]);
  useEffect(() => {
    const controller = new AbortController();
    api<{ available: boolean }>("analysis", undefined, undefined, controller.signal)
      .then((info) => setAnalysisAvailable(info.available))
      .catch(() => setAnalysisAvailable(false));
    return () => controller.abort();
  }, []);
  async function signIn(event: SubmitEvent) {
    event.preventDefault();
    try {
      setSession({ ...(await api<Session>("login", { password })), demo_url: session.demo_url });
      setPassword("");
      setLogin(false);
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }
  const links: [string, string][] = [
    ["/", "Overview"],
    ["/experiments", "Experiments"],
    ["/games", "Games"],
  ];
  if (analysisAvailable !== false || session.authenticated)
    links.push(["/analyze", "Analysis"]);
  return (
    <>
      <header className="service-header">
        <a className="service-brand" href="/">
          <span className="service-logo" aria-hidden="true" />
          {SITE_NAME}
        </a>
        <nav aria-label="Main navigation">
          {links.map(([url, title]) => (
            <a href={url} aria-current={path === url ? "page" : undefined}>
              {title}
            </a>
          ))}
        </nav>
        <div className="service-actions">
          {session.demo_url && (
            <a className="service-button service-demo-link" href={session.demo_url}>
              Play the demo ↗
            </a>
          )}
          <ThemePicker />
          {session.authenticated ? (
            <button
              onClick={async () => {
                await api("logout", {}, session);
                setSession({
                  authenticated: false,
                  login_enabled: true,
                  demo_url: session.demo_url,
                });
              }}
            >
              Log out
            </button>
          ) : (
            session.login_enabled && (
              <button onClick={() => setLogin(!login)}>Log in</button>
            )
          )}
        </div>
      </header>
      {session.read_only && (
        <p className="service-archive" role="status">
          <strong>Read-only archive.</strong> These experiments have finished;
          nothing on this site is live, and controls that would change jobs are
          disabled.
        </p>
      )}
      {login && (
        <form className="service-login service-panel" onSubmit={signIn}>
          <label>
            Administrator password
            <input
              type="password"
              autocomplete="current-password"
              value={password}
              onInput={(e) => setPassword(e.currentTarget.value)}
              required
            />
          </label>
          <button className="primary">Log in</button>
          {error && <p role="alert">{error}</p>}
        </form>
      )}
      {path === "/" ? (
        <Overview session={session} />
      ) : path === "/experiments" ? (
        <JobWorkspace session={session} />
      ) : path === "/games" || path === "/analyze" ? (
        <GameExplorer session={session} recorded={path === "/games"} />
      ) : (
        <main className="service-main">
          <h1>Page not found</h1>
          <p>This address does not match a dashboard page.</p>
          <a href="/">Open the overview</a>
        </main>
      )}
    </>
  );
}
