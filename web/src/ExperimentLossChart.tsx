import type { JSX } from "preact";
import { useState } from "preact/hooks";
import { inspectKey, pointerX, usePlotWidth } from "./InteractiveChart";
import { nearestIndex } from "./experimentChartData";

type Metric = "value_loss" | "policy_loss";
type Epoch = {
  epoch: number;
  training: Record<Metric, number>;
  validation: Record<Metric, number> | null;
  self_play?: { first_player_win_rate: number | null } | null;
};
const value = (n: number) => n.toFixed(6);
const dummyMse = (epoch: Epoch): number | null => {
  const a = epoch.self_play?.first_player_win_rate;
  return typeof a === "number" && Number.isFinite(a) && a >= 0 && a <= 1
    ? 1 - (2 * a - 1) ** 2
    : null;
};

export function LossChart({
  metrics,
  metric,
}: {
  metrics: Epoch[];
  metric: Metric;
}): JSX.Element {
  const { ref, width } = usePlotWidth();
  const [cursor, setCursor] = useState<number | null>(null);
  const [focus, setFocus] = useState<
    "training" | "validation" | "dummy" | null
  >(null);
  const last = metrics.at(-1);
  if (!last) return <p>No completed training passes yet.</p>;
  const index = Math.min(cursor ?? metrics.length - 1, metrics.length - 1),
    selected = metrics[index]!;
  const splits = metrics.every((epoch) => epoch.validation)
    ? (["training", "validation"] as const)
    : (["training"] as const);
  const dummy = metrics.map((epoch) =>
    metric === "value_loss" ? dummyMse(epoch) : null,
  );
  const hasDummy = dummy.some((n) => n !== null);
  const selectedDummy = dummy[index];
  const lastDummy = dummy.at(-1);
  const all = metrics
    .flatMap((epoch) => splits.map((split) => epoch[split]![metric]))
    .concat(dummy.filter((n): n is number => n !== null));
  const minimum = Math.min(...all),
    maximum = Math.max(...all),
    padding = Math.max((maximum - minimum) * 0.12, 0.01);
  const low = Math.max(0, minimum - padding),
    high = maximum + padding;
  const left = 55,
    right = width - 18;
  const firstPass = metrics[0]!.epoch;
  const x = (epoch: number) =>
    left +
    ((epoch - firstPass) / Math.max(last.epoch - firstPass, 1)) *
      (right - left);
  const y = (n: number) => 205 - ((n - low) / (high - low)) * 175;
  function inspect(
    event:
      | JSX.TargetedPointerEvent<SVGSVGElement>
      | JSX.TargetedMouseEvent<SVGSVGElement>,
  ) {
    setCursor(
      nearestIndex(
        metrics.map((point) => x(point.epoch)),
        pointerX(event),
      ),
    );
  }
  const dummyPath = dummy
    .map((n, i) =>
      n === null
        ? ""
        : `${i === 0 || dummy[i - 1] === null ? "M" : "L"}${x(metrics[i]!.epoch)},${y(n)}`,
    )
    .join(" ");
  return (
    <figure ref={ref} className="experiment-chart interactive-chart">
      <div className="chart-readout" aria-live="off">
        <strong>Pass {selected.epoch}</strong>
        <span className="chart-training-value">
          Training <b>{value(selected.training[metric])}</b>
        </span>
        {selected.validation && (
          <span className="chart-validation-value">
            Validation <b>{value(selected.validation[metric])}</b>
          </span>
        )}
        {hasDummy && (
          <span className="chart-dummy-value">
            Dummy MSE{" "}
            <b>{selectedDummy == null ? "—" : value(selectedDummy)}</b>
          </span>
        )}
      </div>
      <div className="experiment-legend">
        {splits.map((split) => (
          <button
            key={split}
            className={
              split === "training" ? "train-series" : "validation-series"
            }
            aria-pressed={focus === split}
            onClick={() => setFocus(focus === split ? null : split)}
          >
            {split === "training" ? "Training" : "Validation"}
          </button>
        ))}
        {hasDummy && (
          <button
            className="dummy-series"
            aria-pressed={focus === "dummy"}
            onClick={() => setFocus(focus === "dummy" ? null : "dummy")}
          >
            Dummy MSE
          </button>
        )}
      </div>
      <svg
        className="interactive-plot"
        role="img"
        aria-label={`${metric === "value_loss" ? "Value MSE" : "Policy cross-entropy"} by training pass`}
        viewBox={`0 0 ${width} 245`}
        onPointerMove={inspect}
        onClick={inspect}
        tabindex={0}
        onKeyDown={(event) =>
          inspectKey(event, index, metrics.length, setCursor)
        }
      >
        {[0, 1, 2, 3, 4].map((tick) => {
          const n = low + ((high - low) * tick) / 4;
          return (
            <g key={tick}>
              <line
                className="chart-grid"
                x1={left}
                x2={right}
                y1={y(n)}
                y2={y(n)}
              />
              <text x={left - 10} y={y(n) + 4} text-anchor="end">
                {n.toFixed(2)}
              </text>
            </g>
          );
        })}
        {splits.map((split) => (
          <g
            key={split}
            className={
              focus && (focus !== "dummy" || hasDummy) && focus !== split
                ? "chart-series-muted"
                : ""
            }
          >
            <polyline
              className={
                split === "training" ? "train-line" : "validation-line"
              }
              fill="none"
              points={metrics
                .map((point) => `${x(point.epoch)},${y(point[split]![metric])}`)
                .join(" ")}
            />
            <line
              className={`chart-guide ${split === "training" ? "train-line" : "validation-line"}`}
              x1={left}
              x2={right}
              y1={y(selected[split]![metric])}
              y2={y(selected[split]![metric])}
            />
            <circle
              className={
                split === "training" ? "train-point" : "validation-point"
              }
              cx={x(selected.epoch)}
              cy={y(selected[split]![metric])}
              r="4"
            />
          </g>
        ))}
        {hasDummy && (
          <g className={focus && focus !== "dummy" ? "chart-series-muted" : ""}>
            <path className="dummy-line" fill="none" d={dummyPath} />
            {dummy.map(
              (n, i) =>
                n !== null && (
                  <circle
                    className="dummy-point"
                    cx={x(metrics[i]!.epoch)}
                    cy={y(n)}
                    r="2"
                  />
                ),
            )}
            {selectedDummy != null && (
              <>
                <line
                  className="chart-guide dummy-line"
                  x1={left}
                  x2={right}
                  y1={y(selectedDummy)}
                  y2={y(selectedDummy)}
                />
                <circle
                  className="dummy-point"
                  cx={x(selected.epoch)}
                  cy={y(selectedDummy)}
                  r="4"
                />
              </>
            )}
          </g>
        )}
        <line
          className="chart-crosshair"
          x1={x(selected.epoch)}
          x2={x(selected.epoch)}
          y1="30"
          y2="205"
        />
        <text x={left} y="229">
          Pass {firstPass}
        </text>
        <text x={right} y="229" text-anchor="end">
          Pass {last.epoch}
        </text>
        <rect
          className="chart-hit-area"
          x={left}
          y="25"
          width={right - left}
          height="185"
        />
      </svg>
      <p className="chart-help">
        Hover, tap, or focus the plot and use arrow keys to inspect a pass.
      </p>
      <figcaption>
        <strong>Reading:</strong> After pass {last.epoch}: training{" "}
        {last.training[metric].toFixed(4)},{" "}
        {last.validation
          ? `validation ${last.validation[metric].toFixed(4)}.`
          : "no held-out validation split."}
        {hasDummy && (
          <>
            {" "}
            Dummy MSE:{" "}
            {lastDummy == null
              ? "unavailable for this pass"
              : lastDummy.toFixed(4)}
            ; baseline 1 − (2a − 1)², where a is that epoch’s first-player win
            rate. This is a game-outcome reference, not measured MSE over replay
            positions.
          </>
        )}{" "}
        Lower is better; playing strength is measured separately.
      </figcaption>
      <details className="chart-data">
        <summary>Exact loss values</summary>
        <div className="chart-data-scroll">
          <table className="experiment-table">
            <thead>
              <tr>
                <th>Pass</th>
                <th
                  className={focus === "training" ? "chart-focused-cell" : ""}
                >
                  Training
                </th>
                <th
                  className={focus === "validation" ? "chart-focused-cell" : ""}
                >
                  Validation
                </th>
                {hasDummy && (
                  <th className={focus === "dummy" ? "chart-focused-cell" : ""}>
                    Dummy MSE
                  </th>
                )}
              </tr>
            </thead>
            <tbody>
              {metrics.map((point, i) => (
                <tr key={point.epoch}>
                  <td>{point.epoch}</td>
                  <td
                    className={focus === "training" ? "chart-focused-cell" : ""}
                  >
                    {value(point.training[metric])}
                  </td>
                  <td
                    className={
                      focus === "validation" ? "chart-focused-cell" : ""
                    }
                  >
                    {point.validation ? value(point.validation[metric]) : "—"}
                  </td>
                  {hasDummy && (
                    <td
                      className={focus === "dummy" ? "chart-focused-cell" : ""}
                    >
                      {dummy[i] == null ? "—" : value(dummy[i]!)}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
