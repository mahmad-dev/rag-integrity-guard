import { ShieldCheck } from "lucide-react";

export default function Home() {
  return (
    <main className="container flex min-h-screen flex-col items-center justify-center gap-4 text-center">
      <ShieldCheck className="h-10 w-10 text-primary" />
      <h1 className="text-3xl font-semibold tracking-tight">rag-integrity-guard</h1>
      <p className="max-w-md text-muted-foreground">
        Live playground, attack simulator, and FEVER benchmark dashboard land in
        Step 6 of the build.
      </p>
    </main>
  );
}
