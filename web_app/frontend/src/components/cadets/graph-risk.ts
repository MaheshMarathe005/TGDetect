export const RISK_COLORS = { high: "#CF6B79", medium: "#C89A5B", low: "#499E89", unknown: "#94A3B8" } as const;
export const WARNING_RATIO = 0.8;

/** Orange is a display band within the model's below-threshold class. */
export function nodeRisk(node: { score: number; alert: boolean }, threshold: number, thresholdBased = true) {
  if (!Number.isFinite(node.score)) return { level: "unknown", label: "Unscored", color: RISK_COLORS.unknown } as const;
  if (!thresholdBased) return node.alert
    ? { level: "high", label: "Predicted malicious", color: RISK_COLORS.high } as const
    : { level: "low", label: "Predicted benign", color: RISK_COLORS.low } as const;
  if (!Number.isFinite(threshold) || threshold <= 0) return { level: "unknown", label: "Threshold unavailable", color: RISK_COLORS.unknown } as const;
  if (node.score > threshold) return { level: "high", label: "Predicted malicious", color: RISK_COLORS.high } as const;
  if (node.score >= threshold * WARNING_RATIO) return { level: "medium", label: "Medium · near threshold", color: RISK_COLORS.medium } as const;
  return { level: "low", label: "Predicted benign", color: RISK_COLORS.low } as const;
}
