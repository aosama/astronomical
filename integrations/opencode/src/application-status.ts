// The app's version, read from a channel's GET /v1/status. Only the status
// command shows it, and a version is never worth failing that command over,
// so every problem comes back as "version unknown".

import { ASTRONOMICAL_PROBE_TIMEOUT_MS } from "./constants.ts";
import { isRecord } from "./wire-values.ts";

/** `application.version` from a /v1/status payload, when it is a string. */
export function readApplicationVersion(
  statusPayload: unknown,
): string | undefined {
  if (!isRecord(statusPayload)) return undefined;
  const application: unknown = statusPayload["application"];
  if (!isRecord(application)) return undefined;
  const version: unknown = application["version"];
  return typeof version === "string" ? version : undefined;
}

/** The running app's version, or undefined when the status read fails. */
export async function fetchApplicationVersion(
  baseUrl: string,
  fetchImpl: typeof fetch,
): Promise<string | undefined> {
  try {
    const statusResponse: Response = await fetchImpl(`${baseUrl}/status`, {
      signal: AbortSignal.timeout(ASTRONOMICAL_PROBE_TIMEOUT_MS),
    });
    if (!statusResponse.ok) return undefined;
    return readApplicationVersion(await statusResponse.json());
  } catch {
    return undefined;
  }
}
