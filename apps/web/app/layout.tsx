import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "rag-integrity-guard",
  description:
    "Security middleware that fingerprints RAG context at ingestion and re-verifies it before it reaches an LLM.",
};

const THEME_SCRIPT = `
  try {
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      document.documentElement.classList.add("dark");
    }
  } catch (_) {}
`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        {/* Runs before paint so there's no flash of the wrong theme. */}
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
