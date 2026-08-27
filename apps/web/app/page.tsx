import { BarChart3, FlaskConical, ShieldCheck } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Playground } from "@/components/playground";
import { BenchmarkPanel } from "@/components/benchmark-panel";

export default function Home() {
  return (
    <main className="container flex min-h-screen flex-col gap-6 py-10">
      <header className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-6 w-6 text-primary" aria-hidden />
          <h1 className="text-xl font-semibold tracking-tight">rag-integrity-guard</h1>
        </div>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Fingerprints RAG context at ingestion and re-verifies it before it reaches an
          LLM, catching retrieval poisoning and tampering. Ingest a few facts, run a query,
          then attack a chunk and watch the guard react.
        </p>
      </header>

      <Tabs defaultValue="playground">
        <TabsList>
          <TabsTrigger value="playground">
            <FlaskConical className="h-4 w-4" aria-hidden />
            Live playground
          </TabsTrigger>
          <TabsTrigger value="benchmark">
            <BarChart3 className="h-4 w-4" aria-hidden />
            Benchmark
          </TabsTrigger>
        </TabsList>

        <TabsContent value="playground">
          <Playground />
        </TabsContent>
        <TabsContent value="benchmark">
          <BenchmarkPanel />
        </TabsContent>
      </Tabs>
    </main>
  );
}
