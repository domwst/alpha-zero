import { render } from "preact";

import { App } from "./App";
import { ServiceApp } from "./ServiceApp";
import { Experiments } from "./Experiments";
import "./styles.css";

if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
  document.documentElement.classList.add("motion-on");
}

function legacy() {
  render(
    location.pathname === "/experiments" ? <Experiments /> : <App />,
    document.getElementById("app")!,
  );
}
if (document.querySelector('meta[name="alz-service"]')) {
  if (location.pathname === "/play")
    history.replaceState(
      null,
      "",
      `/analyze${location.search}${location.hash}`,
    );
  fetch("/api/v1/session")
    .then((r) => {
      if (!r.ok) throw new Error("Job service unavailable");
      return r.json();
    })
    .then((session) =>
      render(
        <ServiceApp initialSession={session} />,
        document.getElementById("app")!,
      ),
    )
    .catch(() =>
      render(
        <main className="service-main">
          <h1>Job service unavailable</h1>
          <p>Reload this page after the service recovers.</p>
        </main>,
        document.getElementById("app")!,
      ),
    );
} else legacy();
