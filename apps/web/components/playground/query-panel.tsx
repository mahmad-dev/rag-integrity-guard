"use client";

import * as React from "react";
import { AlertOctagon, Loader2, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { StatusBadge } from "@/components/status-badge";
import { ApiError, runQuery, type QueryResponse, type StrictnessMode } from "@/lib/api";

const STRICTNESS_OPTIONS: { value: StrictnessMode; label: string; hint: string }[] = [
  {
    value: "log_only",
    label: "Standard -- show everything",
    hint: "No guard applied: every retrieved chunk is returned, annotated with its integrity status.",
  },
  {
    value: "filter",
    label: "Guarded -- filter",
    hint: "Tampered chunks are silently dropped; only VALID chunks reach the response.",
  },
  {
    value: "strict",
    label: "Guarded -- strict (block)",
    hint: "Any tampered chunk blocks the entire query with an IntegrityViolationError.",
  },
];

export function QueryPanel({
  hasIngested,
  externalQuery,
  onQueried,
}: {
  hasIngested: boolean;
  externalQuery: { text: string; strictness: StrictnessMode } | null;
  onQueried: (response: QueryResponse) => void;
}) {
  const [text, setText] = React.useState("Where is the Eiffel Tower?");
  const [k, setK] = React.useState(3);
  const [strictness, setStrictness] = React.useState<StrictnessMode>("filter");
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [response, setResponse] = React.useState<QueryResponse | null>(null);

  const runIt = React.useCallback(
    async (queryText: string, queryStrictness: StrictnessMode) => {
      setLoading(true);
      setError(null);
      try {
        const result = await runQuery(queryText, { k, strictness: queryStrictness });
        setResponse(result);
        onQueried(result);
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Query failed.");
      } finally {
        setLoading(false);
      }
    },
    [k, onQueried]
  );

  // Lets the Attack panel trigger a one-click re-query with the same text/mode.
  React.useEffect(() => {
    if (externalQuery) {
      setText(externalQuery.text);
      setStrictness(externalQuery.strictness);
      void runIt(externalQuery.text, externalQuery.strictness);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [externalQuery]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="h-4 w-4" aria-hidden />
          2. Query
        </CardTitle>
        <CardDescription>
          Compare standard retrieval against the guard by switching modes below.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-col gap-2 sm:flex-row">
          <Input
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Ask a question..."
            className="sm:flex-1"
          />
          <Input
            type="number"
            min={1}
            max={20}
            value={k}
            onChange={(event) => setK(Number(event.target.value) || 1)}
            className="sm:w-20"
            aria-label="Number of chunks to retrieve"
          />
        </div>
        <Select
          value={strictness}
          onChange={(event) => setStrictness(event.target.value as StrictnessMode)}
          aria-label="Strictness mode"
        >
          {STRICTNESS_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </Select>
        <p className="text-xs text-muted-foreground">
          {STRICTNESS_OPTIONS.find((option) => option.value === strictness)?.hint}
        </p>

        <div>
          <Button onClick={() => runIt(text, strictness)} disabled={loading || !hasIngested}>
            {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
            Run query
          </Button>
          {!hasIngested && (
            <span className="ml-3 text-xs text-muted-foreground">Ingest some chunks first.</span>
          )}
        </div>

        {error && <p className="text-xs text-[var(--status-critical)]">{error}</p>}

        {response?.blocked && (
          <div className="flex items-start gap-2 rounded-md border border-[var(--status-critical)]/30 bg-[var(--status-critical)]/5 p-3 text-sm">
            <AlertOctagon
              className="mt-0.5 h-4 w-4 shrink-0 text-[var(--status-critical)]"
              aria-hidden
            />
            <div>
              <p className="font-medium text-[var(--status-critical)]">Query blocked</p>
              <ul className="mt-1 list-inside list-disc text-xs text-muted-foreground">
                {response.violations.map((violation) => (
                  <li key={violation}>{violation}</li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {response && !response.blocked && (
          <ul className="flex flex-col gap-2">
            {response.results.map((result) => (
              <li
                key={result.chunk_id}
                className="rounded-md border border-border p-3 text-sm"
              >
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <span className="font-mono text-xs text-muted-foreground">
                    {result.chunk_id}
                  </span>
                  <StatusBadge status={result.integrity_status} />
                </div>
                <p>{result.text}</p>
                {result.integrity_similarity !== null && (
                  <p className="mt-1 text-xs text-muted-foreground">
                    similarity: {result.integrity_similarity.toFixed(4)}
                  </p>
                )}
              </li>
            ))}
            {response.results.length === 0 && (
              <li className="text-xs text-muted-foreground">
                No chunks returned (all were filtered out).
              </li>
            )}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
