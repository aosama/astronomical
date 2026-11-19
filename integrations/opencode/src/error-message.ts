// A thrown value can be anything in JavaScript; diagnostics and toasts need a
// sentence. One conversion, so the log and the terminal word a failure alike.

/** The message of an Error, or the thrown value as text. */
export function describeError(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}
