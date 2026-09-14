# Local dashboard connection

Docker Compose manages the local SSH tunnel to the GPU pod. The web backend,
scheduler, database, and training workers stay on the pod; stopping this container
disconnects the local dashboard without stopping training.

The tunnel binds **127.0.0.1:8765**, forwarding to the pod's **127.0.0.1:8766**.
The existing HTTPS reverse proxy can keep the same upstream. This configuration
uses [Linux host networking](https://docs.docker.com/reference/compose-file/services/#network_mode).

## Setup

From this directory:

```sh
cp .env.example .env
# Edit .env with the pod connection and absolute local SSH paths.
docker compose config --quiet
docker compose build
docker compose up -d --wait --wait-timeout 60
```

The `.env` file is ignored by Git. It contains connection settings and file paths;
the key is mounted read-only at runtime and excluded from the image build context.
Set UID/GID to the private key owner's numeric IDs. Keep the key's existing private
permissions. SSH checks the server against the supplied verified `known_hosts` file.

If migrating from the old user service, first build and test Compose on a spare
port, then stop the old listener and start the normal Compose project:

```sh
DASHBOARD_LOCAL_PORT=8769 docker compose -p alz-dashboard-preview up -d --wait --wait-timeout 60
DASHBOARD_LOCAL_PORT=8769 docker compose -p alz-dashboard-preview down
systemctl --user stop alz-experiment-dashboard.service
docker compose up -d --wait --wait-timeout 60
systemctl --user disable alz-experiment-dashboard.service
```

If startup fails, restore the previous listener with `docker compose down` followed
by `systemctl --user enable --now alz-experiment-dashboard.service`.

## Operations

```sh
docker compose ps
docker compose logs -f
docker compose restart
docker compose down
```

SSH keepalives detect lost connections; Docker restarts an exited tunnel. The HTTP
health check reports whether the pod's dashboard is reachable. An unhealthy HTTP
check alone does not restart the container: a backend outage requires inspecting
the pod service. Docker starts the container again after a daemon restart unless
it was explicitly stopped. Docker itself must be enabled at host boot.

After changing pods, update `.env` and the verified host-key file, then run
`docker compose up -d --force-recreate --wait --wait-timeout 60`. Changes to the public
domain belong in the pod service's exact `--origin` setting and the HTTPS proxy.

The current public origin is `https://dashboard.oleja.dev`. Configure the pod
service with `--origin https://dashboard.oleja.dev`; the old domain is not an
accepted origin. The local tunnel remains bound to `127.0.0.1`.
