import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve, extname, sep } from "node:path";
import { chromium, firefox } from "playwright";

export const engines = [chromium, firefox];
export const gate = () => {
  let release;
  return {
    promise: new Promise((r) => {
      release = r;
    }),
    release: () => release(),
  };
};
export async function fixture(t, engine, { service = true, hasTouch = false } = {}) {
  const root = fileURLToPath(new URL("../../dist/", import.meta.url));
  const server = createServer(async (req, res) => {
    try {
      const path = new URL(req.url, "http://localhost").pathname;
      const file =
        path.startsWith("/assets/") || path === "/theme-init.js"
          ? resolve(root, "." + path)
          : resolve(root, "index.html");
      if (!file.startsWith(root.endsWith(sep) ? root : root + sep))
        throw new Error("Invalid asset");
      let content = await readFile(file);
      if (service && extname(file) === ".html")
        content = Buffer.from(
          content
            .toString()
            .replace("<head>", '<head><meta name="alz-service" content="1">'),
        );
      res.writeHead(200, {
        "Content-Type":
          {
            ".js": "text/javascript",
            ".css": "text/css",
            ".html": "text/html",
          }[extname(file)] || "application/octet-stream",
      });
      res.end(content);
    } catch {
      res.writeHead(404);
      res.end();
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(
    () =>
      new Promise((resolve) => {
        server.closeAllConnections();
        server.close(resolve);
      }),
  );
  const browser = await engine.launch();
  t.after(() => browser.close());
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
    hasTouch,
  });
  page.setDefaultTimeout(8000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error));
  t.after(() => {
    if (errors.length)
      throw new AggregateError(errors, "Browser runtime errors");
  });
  await page.route("**/api/v1/**", (route) =>
    route.fulfill({
      status: 404,
      json: { error: "Unexpected fixture request" },
    }),
  );
  await page.route("**/api/v1/session", (route) =>
    route.fulfill({ json: { authenticated: true, csrf: "fixture" } }),
  );
  return { page, origin: `http://127.0.0.1:${server.address().port}` };
}
export const job = (id, kind = "self_play") => ({
  id,
  title: id,
  state: "paused",
  reason: "pause",
  created: 0,
  spec: {
    kind,
    options: { device: "cpu" },
    inputs: {},
    resources: { slots: 1, host_memory_mb: 8192, gpu_memory_mb: 0 },
  },
  attempts: [],
  dependencies: [],
});
