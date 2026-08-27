import type { Evidence } from "./runState";

/** Select only an explicitly validated HTTP(S) URL for outbound navigation. */
export function safeSourceUrl(item: Pick<Evidence, "canonical_url" | "final_url">): string | null {
  const candidates = [item.final_url, item.canonical_url];
  for (const candidate of candidates) {
    if (!candidate) continue;
    // Require an explicit authority separator; `http:foo` is ambiguous across URL implementations.
    if (!candidate.includes("://")) continue;
    try {
      const parsed = new URL(candidate);
      if (
        (parsed.protocol === "https:" || parsed.protocol === "http:") &&
        parsed.hostname &&
        !parsed.username &&
        !parsed.password
      ) {
        return parsed.href;
      }
    } catch {
      // Runtime data may be stale or hostile; render a non-link state.
    }
  }
  return null;
}
