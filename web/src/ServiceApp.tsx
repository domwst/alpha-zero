import { useState } from "preact/hooks";
import { ThemePicker } from "./ThemePicker";
import { JobWorkspace } from "./JobWorkspace";
import { GameExplorer } from "./GameExplorer";
import { api, type Session } from "./serviceApi";
import "./service.css";

export function ServiceApp({ initialSession }: { initialSession: Session }) {
  const [session, setSession] = useState(initialSession),
    [login, setLogin] = useState(false),
    [password, setPassword] = useState(""),
    [error, setError] = useState("");
  const path = location.pathname;
  async function signIn(event: SubmitEvent) {
    event.preventDefault();
    try {
      setSession(await api<Session>("login", { password }));
      setPassword("");
      setLogin(false);
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }
  return (
    <>
      <header className="service-header">
        <a className="service-brand" href="/experiments">
          AlphaZero playground
        </a>
        <nav aria-label="Main navigation">
          {[
            ["/experiments", "Experiments"],
            ["/games", "Games"],
            ["/analyze", "Analysis"],
          ].map(([url, title]) => (
            <a href={url} aria-current={path === url ? "page" : undefined}>
              {title}
            </a>
          ))}
        </nav>
        <div className="service-actions">
          <ThemePicker />
          {session.authenticated ? (
            <button
              onClick={async () => {
                await api("logout", {}, session);
                setSession({ authenticated: false, login_enabled: true });
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
      {path === "/experiments" || path === "/" ? (
        <JobWorkspace session={session} />
      ) : path === "/games" || path === "/analyze" ? (
        <GameExplorer session={session} recorded={path === "/games"} />
      ) : (
        <main className="service-main">
          <h1>Page not found</h1>
          <p>This address does not match a playground page.</p>
          <a href="/experiments">Open experiments</a>
        </main>
      )}
    </>
  );
}
