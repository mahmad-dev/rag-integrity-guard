import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "rag-integrity-guard",
  description:
    "Security middleware that fingerprints RAG context at ingestion and re-verifies it before it reaches an LLM.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
