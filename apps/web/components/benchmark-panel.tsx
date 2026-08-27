"use client";

import * as React from "react";
import { Loader2 } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ApiError, getBenchmarkReport, type AttackTypeMetrics, type BenchmarkReport } from "@/lib/api";

const SERIES_COLORS: Record<string, string> = {
  exact_mutation: "var(--series-1)",
  payload_injection: "var(--series-2)",
  semantic_drift: "var(--series-3)",
};

const ATTACK_LABELS: Record<string, string> = {
  exact_mutation: "Exact mutation",
  payload_injection: "Payload injection",
  semantic_drift: "Semantic drift",
};

function StatTile({ label, value, tone }: { label: string; value: string; tone?: "good" | "muted" }) {
  return (
    <div className="rounded-lg border border-border p-4">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p
        className="mt-1 text-2xl font-semibold"
        style={tone === "good" ? { color: "var(--status-good)" } : undefined}
      >
        {value}
      </p>
    </div>
  );
}

function TprBar({ metric }: { metric: AttackTypeMetrics }) {
  const pct = metric.true_positive_rate !== null ? metric.true_positive_rate * 100 : 0;
  const color = SERIES_COLORS[metric.attack_type] ?? "var(--series-1)";
  return (
    <div
      className="group grid grid-cols-[9rem_1fr_3.5rem] items-center gap-3"
      title={`${metric.counts.true_positive}/${metric.sample_count} attacked chunks flagged non-VALID`}
    >
      <span className="text-sm">{ATTACK_LABELS[metric.attack_type] ?? metric.attack_type}</span>
      <span
        className="h-3 rounded-[4px] bg-[var(--chart-grid)]"
        style={{
          background: `linear-gradient(to right, ${color} ${pct}%, var(--chart-grid) ${pct}%)`,
        }}
      />
      <span className="text-right text-xs tabular-nums text-muted-foreground">
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

export function BenchmarkPanel() {
  const [report, setReport] = React.useState<BenchmarkReport | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getBenchmarkReport()
      .then((result) => {
        if (!cancelled) setReport(result);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : "Failed to load the benchmark report.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
        Loading benchmark report...
      </div>
    );
  }

  if (error || !report) {
    return <p className="text-sm text-[var(--status-critical)]">{error ?? "No report available."}</p>;
  }

  const attackMetrics = report.per_attack_type.filter((m) => m.attack_type !== "none");
  const benignMetric = report.per_attack_type.find((m) => m.attack_type === "none");
  const fprPct = report.overall_false_positive_rate * 100;

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <CardTitle>FEVER benchmark</CardTitle>
          <CardDescription>
            {report.dataset} ({report.split}) -- {report.sample_size} real claim/evidence
            pairs, similarity threshold {report.similarity_threshold}. Generated{" "}
            {new Date(report.generated_at).toLocaleString()}.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatTile
              label="Overall TPR"
              value={`${(report.overall_true_positive_rate * 100).toFixed(0)}%`}
              tone="good"
            />
            <StatTile
              label="Overall FPR"
              value={`${fprPct.toFixed(0)}%`}
              tone={fprPct === 0 ? "good" : undefined}
            />
            <StatTile
              label="Mean overhead"
              value={`${report.verification_overhead_ms_mean.toFixed(2)}ms`}
            />
            <StatTile
              label="P95 overhead"
              value={`${report.verification_overhead_ms_p95.toFixed(2)}ms`}
            />
          </div>

          <div className="flex flex-col gap-2">
            <p className="text-xs font-medium text-muted-foreground">
              Detection rate by attack type
              {benignMetric && ` (benign baseline: ${benignMetric.sample_count} untouched chunks)`}
            </p>
            {attackMetrics.map((metric) => (
              <TprBar key={metric.attack_type} metric={metric} />
            ))}
          </div>

          <p className="text-xs text-muted-foreground">
            TPR = 1.0 and FPR = 0.0 aren&apos;t estimates here -- they&apos;re a designed
            invariant of hash-based verification (any byte change fails the hash check;
            identical content always passes). The real signal in this benchmark is the
            HASH_MISMATCH vs SEMANTIC_DRIFT classification split and the latency numbers,
            not a probabilistic detection rate.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
