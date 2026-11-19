// Type guards for untrusted JSON: the app's HTTP responses and the plugin's
// own state file. One home, so every reader narrows a parsed value the same
// way.

/** True for a plain JSON object; arrays and null are not records. */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
