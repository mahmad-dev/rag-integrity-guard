import { AlertTriangle, CheckCircle2, HelpCircle, type LucideIcon, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { ValidationStatus } from "@/lib/api";

const STATUS_META: Record<
  ValidationStatus,
  { label: string; Icon: LucideIcon; color: string }
> = {
  valid: { label: "Valid", Icon: CheckCircle2, color: "var(--status-good)" },
  hash_mismatch: { label: "Hash mismatch", Icon: XCircle, color: "var(--status-critical)" },
  semantic_drift: {
    label: "Semantic drift",
    Icon: AlertTriangle,
    color: "var(--status-warning)",
  },
  missing_fingerprint: {
    label: "No fingerprint",
    Icon: HelpCircle,
    color: "var(--status-serious)",
  },
};

export function StatusBadge({ status }: { status: ValidationStatus }) {
  const meta = STATUS_META[status];
  return (
    <Badge style={{ borderColor: meta.color, color: meta.color }}>
      <meta.Icon className="h-3.5 w-3.5" aria-hidden />
      {meta.label}
    </Badge>
  );
}
