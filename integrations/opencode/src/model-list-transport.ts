// One GET of a channel's /v1/models, and nothing else: the transport knows
// HTTP status, timeouts and JSON, not what a chat model is.

import { describeError } from "./error-message.js";

export type ModelListPayloadResult =
  | { readonly kind: "success"; readonly payload: unknown }
  | { readonly kind: "failure"; readonly reason: string };

/**
 * Fetches one channel's model list. A stopped channel, a non-2xx answer, a
 * timeout or a body that is not JSON all come back as a failure with a
 * readable reason, never as a throw.
 */
export async function fetchModelListPayload(input: {
  readonly baseUrl: string;
  readonly fetchImpl: typeof fetch;
  readonly timeoutMs: number;
}): Promise<ModelListPayloadResult> {
  try {
    const response: Response = await input.fetchImpl(
      `${input.baseUrl}/models`,
      { signal: AbortSignal.timeout(input.timeoutMs) },
    );
    if (!response.ok) {
      return { kind: "failure", reason: `responded ${response.status}` };
    }
    return { kind: "success", payload: await response.json() };
  } catch (cause: unknown) {
    return { kind: "failure", reason: describeError(cause) };
  }
}
