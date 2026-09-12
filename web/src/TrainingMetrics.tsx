import type { JSX } from 'preact';
import { useState } from 'preact/hooks';
import { LossChart } from './ExperimentLossChart';
import { SelfPlayMetrics, selfPlayMetricLabels } from './SelfPlayMetrics';
import type { SelfPlayMetric, SelfPlayStats } from './SelfPlayMetrics';

type Loss = { value_loss: number; policy_loss: number; samples_per_second?: number; samples?: number; batches?: number; duration_seconds?: number };
type Epoch = { epoch: number; training: Loss; validation: Loss | null; self_play?: SelfPlayStats | null; self_play_seconds?: number | null; scheduled_learning_rate?: number | null };
type Metric = 'value_loss' | 'policy_loss' | SelfPlayMetric;

export function TrainingMetrics({ metrics }: { metrics: Epoch[] }): JSX.Element {
  const [metric, setMetric] = useState<Metric>('value_loss');
  const selfPlay = metrics.some(row => row.self_play);
  const labels = { value_loss: 'Value MSE', policy_loss: 'Policy cross-entropy', ...(selfPlay ? selfPlayMetricLabels : {}) };
  const loss = metric === 'value_loss' || metric === 'policy_loss';
  const throughput = metrics.at(-1)?.training.samples_per_second;
  return <section aria-label="Epoch metrics">
    <div className="experiment-chart-controls" role="group" aria-label="Epoch metric">
      {Object.entries(labels).map(([key, label]) => <button key={key} aria-pressed={metric === key}
        onClick={() => setMetric(key as Metric)}>{label}</button>)}
      {throughput != null && <span>Latest: {throughput.toLocaleString(undefined, { maximumFractionDigits: 0 })} samples/s</span>}
    </div>
    {loss && <LossChart key={metric} metrics={metrics} metric={metric} />}
    {selfPlay && <SelfPlayMetrics metrics={metrics} metric={loss ? undefined : metric} />}
  </section>;
}
