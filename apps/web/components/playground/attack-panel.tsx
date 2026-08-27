"use client";

import * as React from "react";
import { Loader2, RefreshCw, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { StatusBadge } from "@/components/status-badge";
import { ApiError, launchAttack, type AttackResponse, type AttackType } from "@/lib/api";

const ATTACK_OPTIONS: { value: Exclude<AttackType, "none">; label: string; hint: string }[] = [
  {
    value: "exact_mutation",
    label: "Exact mutation",
    hint: "Flips a handful of characters -- a direct database/vector-store edit.",
  },
  {
    value: "payload_injection",
    label: "Payload injection",
    hint: "Appends an indirect prompt-injection payload onto the chunk.",
  },
  {
    value: "semantic_drift",
    label: "Semantic drift",
    hint: "Swaps the chunk wholesale for a different, topically-similar passage -- disguised poisoning.",
  },
];

export function AttackPanel({
  availableChunkIds,
  onAttacked,
}: {
  availableChunkIds: string[];
  onAttacked: (response: AttackResponse) => void;
}) {
  const [chunkId, setChunkId] = React.useState("");
  const [attackType, setAttackType] = React.useState<Exclude<AttackType, "none">>(
    "exact_mutation"
  );
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<AttackResponse | null>(null);

  React.useEffect(() => {
    if (!chunkId && availableChunkIds.length > 0) {
      setChunkId(availableChunkIds[0]);
    }
  }, [availableChunkIds, chunkId]);

  async function handleAttack() {
    if (!chunkId) return;
    setLoading(true);
    setError(null);
    try {
      const response = await launchAttack(chunkId, attackType);
      setResult(response);
      onAttacked(response);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Attack failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-4 w-4" aria-hidden />
          3. Attack simulator
        </CardTitle>
        <CardDescription>
          Tampers a chunk that&apos;s already live in the store -- its stored fingerprint is
          left untouched, exactly like a real poisoning attack.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-col gap-2 sm:flex-row">
          <Select
            value={chunkId}
            onChange={(event) => setChunkId(event.target.value)}
            disabled={availableChunkIds.length === 0}
            aria-label="Target chunk"
            className="sm:flex-1"
          >
            {availableChunkIds.length === 0 && <option value="">Ingest chunks first</option>}
            {availableChunkIds.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </Select>
          <Select
            value={attackType}
            onChange={(event) => setAttackType(event.target.value as Exclude<AttackType, "none">)}
            aria-label="Attack type"
            className="sm:flex-1"
          >
            {ATTACK_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </Select>
        </div>
        <p className="text-xs text-muted-foreground">
          {ATTACK_OPTIONS.find((option) => option.value === attackType)?.hint}
        </p>

        <div>
          <Button
            variant="destructive"
            onClick={handleAttack}
            disabled={loading || !chunkId}
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
            Launch attack
          </Button>
          {error && (
            <span className="ml-3 text-xs text-[var(--status-critical)]">{error}</span>
          )}
        </div>

        {result && (
          <div className="rounded-md border border-border p-3 text-sm">
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="font-mono text-xs text-muted-foreground">{result.chunk_id}</span>
              <StatusBadge status={result.resulting_status} />
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              <div>
                <p className="mb-1 text-xs font-medium text-muted-foreground">Before</p>
                <p className="text-xs">{result.original_excerpt}</p>
              </div>
              <div>
                <p className="mb-1 text-xs font-medium text-muted-foreground">After</p>
                <p className="text-xs">{result.tampered_excerpt}</p>
              </div>
            </div>
            {result.resulting_similarity !== null && (
              <p className="mt-2 text-xs text-muted-foreground">
                resulting similarity: {result.resulting_similarity.toFixed(4)}
              </p>
            )}
            <p className="mt-2 flex items-center gap-1 text-xs text-muted-foreground">
              <RefreshCw className="h-3 w-3" aria-hidden />
              Re-run the query above to see the guard react.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
