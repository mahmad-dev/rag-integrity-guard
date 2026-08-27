"use client";

import * as React from "react";
import { IngestPanel } from "@/components/playground/ingest-panel";
import { QueryPanel } from "@/components/playground/query-panel";
import { AttackPanel } from "@/components/playground/attack-panel";
import type { AttackResponse, IngestedChunk, QueryResponse, StrictnessMode } from "@/lib/api";

export function Playground() {
  const [documentId] = React.useState(
    () => `demo-${Math.random().toString(36).slice(2, 8)}`
  );
  const [ingestedChunks, setIngestedChunks] = React.useState<IngestedChunk[]>([]);
  const [lastQueryParams, setLastQueryParams] = React.useState<{
    text: string;
    strictness: StrictnessMode;
  } | null>(null);
  const [reQueryTrigger, setReQueryTrigger] = React.useState<{
    text: string;
    strictness: StrictnessMode;
  } | null>(null);

  function handleIngested(chunks: IngestedChunk[]) {
    setIngestedChunks((prev) => [...prev, ...chunks]);
  }

  function handleQueried(response: QueryResponse) {
    setLastQueryParams({ text: response.query, strictness: response.strictness });
  }

  function handleAttacked(_response: AttackResponse) {
    if (lastQueryParams) {
      // New object reference each time so QueryPanel's effect re-fires even
      // if the text/strictness are identical to the previous attack.
      setReQueryTrigger({ ...lastQueryParams });
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-muted-foreground">
        Demo state (the vector store and fingerprints) is shared in memory by whichever API
        instance is running -- it isn&apos;t per-browser-session.
      </p>
      <IngestPanel documentId={documentId} onIngested={handleIngested} />
      <QueryPanel
        hasIngested={ingestedChunks.length > 0}
        externalQuery={reQueryTrigger}
        onQueried={handleQueried}
      />
      <AttackPanel
        availableChunkIds={ingestedChunks.map((chunk) => chunk.chunk_id)}
        onAttacked={handleAttacked}
      />
    </div>
  );
}
