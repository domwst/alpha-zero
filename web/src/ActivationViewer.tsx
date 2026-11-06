import { useRef, useState } from "preact/hooks";
import { moveName, cellKey, type PositionMessage } from "./protocol";
import type { Activation } from "./gameViewModel";

export function ActivationViewer({
  activations,
  layerName,
  setLayerName,
  positionPending,
  loadLayer,
  busy,
  board,
}: {
  activations: Activation[];
  layerName: string;
  setLayerName: (name: string) => void;
  positionPending: boolean;
  loadLayer: (name: string) => void;
  busy: boolean;
  board: PositionMessage;
}) {
  const layer = Math.max(
    0,
    activations.findIndex((a) => a.name === layerName),
  );
  const [channel, setChannel] = useState(0),
    [point, setPoint] = useState(0),
    [showBoard, setShowBoard] = useState(true);
  const svg = useRef<SVGSVGElement | null>(null);
  const requestedTensor = activations[layer] || activations[0]!;
  const displayedTensor = useRef(requestedTensor);
  if (requestedTensor.values.length) displayedTensor.current = requestedTensor;
  const tensor = displayedTensor.current,
    loadingLayer = tensor.name !== requestedTensor.name,
    spatial = tensor.axes.includes("row");
  const rows = spatial ? tensor.shape.at(-2)! : 1,
    columns = tensor.shape.at(-1)!,
    channels = tensor.axes.includes("channel") ? tensor.shape[1]! : 1;
  const values = tensor.values.slice(
      Math.min(channel, channels - 1) * rows * columns,
      (Math.min(channel, channels - 1) + 1) * rows * columns,
    ),
    maximum = Math.max(...values.map(Math.abs), 1e-8);
  const compatible = spatial && rows === 19 && columns === 19;
  const coordinate = (i: number) =>
    compatible
      ? moveName({ row: Math.floor(i / columns), column: i % columns })
      : `Row ${Math.floor(i / columns) + 1}, column ${(i % columns) + 1}`;
  return (
    <section className="service-panel">
      <h2>Network activations</h2>
      <div className="service-actions">
        <label>
          Layer
          <select
            aria-label="Layer"
            disabled={busy}
            value={layer}
            onChange={(e) => {
              const i = Number(e.currentTarget.value);
              setLayerName(activations[i]!.name);
              const next = activations[i]!;
              const count = next.axes.includes("channel") ? next.shape[1]! : 1;
              setChannel((old) => Math.min(old, count - 1));
              setPoint(0);
              if (!activations[i]!.values.length)
                loadLayer(activations[i]!.name);
            }}
          >
            {activations.map((v, i) => (
              <option key={v.name} value={i}>
                {v.name} · {v.shape.join(" × ")}
              </option>
            ))}
          </select>
        </label>
        <div className="service-channel-control">
          <label htmlFor="activation-channel">Channel · 0–{channels - 1}</label>
          <div className="service-actions">
            <input
              id="activation-channel"
              aria-label="Channel"
              type="range"
              min="0"
              max={channels - 1}
              step="1"
              disabled={loadingLayer || channels < 2}
              value={Math.min(channel, channels - 1)}
              onInput={(e) => {
                setChannel(Number(e.currentTarget.value));
                setPoint(0);
              }}
            />
            <input
              aria-label="Channel number"
              type="number"
              min="0"
              max={channels - 1}
              step="1"
              disabled={loadingLayer}
              value={Math.min(channel, channels - 1)}
              onInput={(e) => {
                setChannel(
                  Math.max(
                    0,
                    Math.min(
                      channels - 1,
                      Math.trunc(Number(e.currentTarget.value)),
                    ),
                  ),
                );
                setPoint(0);
              }}
            />
          </div>
        </div>
        {compatible && (
          <label className="service-checkbox">
            <input
              type="checkbox"
              checked={showBoard}
              onChange={(e) => setShowBoard(e.currentTarget.checked)}
            />
            Overlay board stones
          </label>
        )}
      </div>
      <p>
        <output>
          {positionPending
            ? busy
              ? "Loading activations for this position…"
              : "Analyze this position to update activations."
            : loadingLayer
              ? busy
                ? `Loading ${requestedTensor.name}…`
                : `Could not load ${requestedTensor.name}. Showing ${tensor.name}.`
              : values.length
                ? `${coordinate(point)} · activation ${values[point]?.toFixed(6)}`
                : "Select this layer to load its activations."}
        </output>
      </p>
      <div className="service-activation-scale" aria-busy={positionPending}>
        <svg
          viewBox="0 0 320 12"
          role="img"
          aria-label={
            positionPending
              ? "Activation scale is loading"
              : `Activation values from −${maximum.toFixed(4)} to +${maximum.toFixed(4)}, with zero at the center. Scale is per channel.`
          }
        >
          <defs>
            <linearGradient
              id="activation-color-scale"
              x1="0%"
              x2="100%"
              y1="0%"
              y2="0%"
              color-interpolation="sRGB"
            >
              <stop offset="0%" stop-color="var(--ds-chart-series-5)" />
              <stop
                offset="47%"
                stop-color="var(--ds-chart-series-5)"
                stop-opacity="0.06"
              />
              <stop
                offset="49.999%"
                stop-color="var(--ds-chart-series-5)"
                stop-opacity="0"
              />
              <stop
                offset="50%"
                stop-color="var(--ds-chart-series-1)"
                stop-opacity="0"
              />
              <stop
                offset="53%"
                stop-color="var(--ds-chart-series-1)"
                stop-opacity="0.06"
              />
              <stop offset="100%" stop-color="var(--ds-chart-series-1)" />
            </linearGradient>
          </defs>
          <rect width="320" height="12" fill="var(--ds-color-surface)" />
          <rect width="320" height="12" fill="url(#activation-color-scale)" />
        </svg>
        <div className="service-activation-scale-ticks">
          <span>{positionPending ? "—" : `−${maximum.toFixed(4)}`}</span>
          <span>0</span>
          <span>{positionPending ? "—" : `+${maximum.toFixed(4)}`}</span>
        </div>
        <small>Per-channel scale</small>
      </div>
      {values.length > 0 && (
        <svg
          ref={svg}
          className={`service-activation${spatial ? "" : " service-activation-vector"}`}
          viewBox={`-1 -1 ${columns + 2} ${rows + 2}`}
          role="grid"
          aria-label={`${tensor.name}, channel ${Math.min(channel, channels - 1)}`}
          aria-busy={loadingLayer || positionPending}
        >
          {Array.from({ length: rows }, (_, r) => (
            <g key={r} role="row">
              {Array.from({ length: columns }, (_, c) => {
                const i = r * columns + c,
                  v = positionPending ? 0 : values[i] || 0;
                return (
                  <rect
                    key={i}
                    data-point={i}
                    role="gridcell"
                    aria-label={`${coordinate(i)}: ${positionPending ? "waiting for activations" : v.toFixed(6)}`}
                    aria-selected={point === i}
                    x={c}
                    y={r}
                    width="1"
                    height="1"
                    tabindex={point === i ? 0 : -1}
                    onFocus={() => setPoint(i)}
                    onPointerEnter={() => setPoint(i)}
                    onClick={() => setPoint(i)}
                    onKeyDown={(e) => {
                      const delta: Record<string, number> = {
                        ArrowLeft: c > 0 ? -1 : 0,
                        ArrowRight: c + 1 < columns ? 1 : 0,
                        ArrowUp: r > 0 ? -columns : 0,
                        ArrowDown: r + 1 < rows ? columns : 0,
                      };
                      if (e.key in delta) {
                        e.preventDefault();
                        const next = i + delta[e.key]!;
                        setPoint(next);
                        svg.current
                          ?.querySelector<SVGElement>(`[data-point="${next}"]`)
                          ?.focus();
                      }
                    }}
                    className={
                      v >= 0 ? "activation-positive" : "activation-negative"
                    }
                    fill-opacity={positionPending ? 0 : Math.abs(v) / maximum}
                    stroke="var(--ds-color-border-strong)"
                    stroke-width=".025"
                  />
                );
              })}
            </g>
          ))}
          {compatible &&
            showBoard &&
            board.stones.map((stone) => (
              <circle
                key={cellKey(stone)}
                pointer-events="none"
                cx={stone.column + 0.5}
                cy={stone.row + 0.5}
                r=".28"
                fill={stone.color === "black" ? "#171717" : "#fafafa"}
                stroke={stone.color === "black" ? "#fafafa" : "#171717"}
                stroke-width=".035"
              />
            ))}
          {compatible &&
            Array.from({ length: 19 }, (_, i) => (
              <g key={i} aria-hidden="true" className="activation-coordinates">
                <text x={i + 0.5} y="-.3">
                  {moveName({ row: 0, column: i }).replace(/[0-9]/g, "")}
                </text>
                <text x="-.45" y={i + 0.6}>
                  {19 - i}
                </text>
              </g>
            ))}
        </svg>
      )}
    </section>
  );
}
