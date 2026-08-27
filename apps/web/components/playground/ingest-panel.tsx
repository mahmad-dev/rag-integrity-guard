"use client";

import * as React from "react";
import { DatabaseZap, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ApiError, ingestDocument, type IngestedChunk } from "@/lib/api";

const SAMPLE_TEXT = [
  "The Eiffel Tower is located in Paris, France.",
  "Mount Everest is the tallest mountain above sea level.",
  "The Great Wall of China stretches over 13,000 miles.",
].join("\n");

export function IngestPanel({
  documentId,
  onIngested,
}: {
  documentId: string;
  onIngested: (chunks: IngestedChunk[]) => void;
}) {
  const [text, setText] = React.useState(SAMPLE_TEXT);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [ingested, setIngested] = React.useState<IngestedChunk[]>([]);

  async function handleIngest() {
    const chunks = text
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => ({ text: line }));

    if (chunks.length === 0) {
      setError("Add at least one line of text to ingest.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await ingestDocument(documentId, chunks);
      setIngested(response.ingested);
      onIngested(response.ingested);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ingest failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <DatabaseZap className="h-4 w-4" aria-hidden />
          1. Ingest
        </CardTitle>
        <CardDescription>
          One chunk per line. Each is fingerprinted (hash + embedding signature) and added
          to the in-memory vector store.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <Textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={4}
          spellCheck={false}
        />
        <div className="flex items-center gap-3">
          <Button onClick={handleIngest} disabled={loading}>
            {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
            Ingest chunks
          </Button>
          {error && <span className="text-xs text-[var(--status-critical)]">{error}</span>}
        </div>
        {ingested.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {ingested.map((chunk) => (
              <Badge key={chunk.chunk_id} title={`content hash ${chunk.content_hash}`}>
                {chunk.chunk_id}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
