// Read the live Astronomical model catalogue from the running daemon(s) and
// return the chat models each reachable channel can actually serve.
//
// This module owns the transport and the capability decision. Row parsing lives
// in astronomer-model-row.ts, so "this row is malformed" and "this model is not
// chat-capable" stay separate concerns.
//
// Every channel is probed independently and all successes are returned: the
// stable and dev daemons are separate installations serving separate model
// sets, so one answering must not hide the other. Any per-channel failure —
// channel down, non-2xx, malformed payload — excludes only that channel; a
// total failure resolves to a failure result rather than an empty or stale
// list, so opencode simply omits the providers until a channel is reachable
// again.

import type { AstronomicalChannel } from "./astronomer-base-url.js";
import type { AstronomicalChannelInstance } from "./constants.js";
import type { AstronomicalModelRow } from "./astronomer-model.js";
import { parseAstronomicalModelRow } from "./astronomer-model-row.js";
import {
  ASTRONOMICAL_CHAT_ENDPOINT,
  ASTRONOMICAL_CATALOGUE_TIMEOUT_MS,
} from "./constants.js";

export type FetchAstronomicalCatalogueOptions = {
  readonly channels: readonly AstronomicalChannel[];
  readonly fetchImpl?: typeof fetch;
  readonly timeoutMs?: number;
  readonly onDiagnostic?: (message: string) => void;
};

// The live catalogue of one reachable channel. A channel that answered HTTP
// 200 appears here even with zero chat models, so "daemon up, nothing
// chat-capable loaded" stays distinguishable from "daemon down" for whoever
// reads the result.
export interface AstronomicalChannelCatalogue {
  readonly instance: AstronomicalChannelInstance;
  readonly baseUrl: string;
  readonly models: readonly AstronomicalModelRow[];
}

export type FetchAstronomicalCatalogueResult =
  | {
      readonly kind: "success";
      readonly channels: readonly AstronomicalChannelCatalogue[];
    }
  | { readonly kind: "failure" };

// The result of probing a single channel. A failure carries the human-readable
// reason so the orchestrator can fold every channel's reason into one summary
// diagnostic instead of one noisy message per channel.
type FetchChannelResult =
  | {
      readonly kind: "success";
      readonly models: readonly AstronomicalModelRow[];
    }
  | { readonly kind: "failure"; readonly reason: string };

export async function fetchAstronomicalCatalogue(
  options: FetchAstronomicalCatalogueOptions,
): Promise<FetchAstronomicalCatalogueResult> {
  // Probe every channel, not just the first that answers: with stable and dev
  // running side by side, first-answer-wins made the dev catalogue permanently
  // invisible, which is the exact bug this fan-in replaced.
  const catalogues: AstronomicalChannelCatalogue[] = [];
  const failures: string[] = [];

  for (const channel of options.channels) {
    const outcome = await fetchChannel({
      baseUrl: channel.baseUrl,
      fetchImpl: options.fetchImpl,
      timeoutMs: options.timeoutMs ?? ASTRONOMICAL_CATALOGUE_TIMEOUT_MS,
    });
    if (outcome.kind === "success") {
      catalogues.push({
        instance: channel.instance,
        baseUrl: channel.baseUrl,
        models: outcome.models,
      });
    } else {
      failures.push(`${channel.baseUrl} (${outcome.reason})`);
    }
  }

  if (failures.length > 0 && failures.length === options.channels.length) {
    options.onDiagnostic?.(
      `no Astronomical channel is reachable — ${failures.join(
        "; ",
      )}; start Astronomical to register its models`,
    );
    return { kind: "failure" };
  }

  // A partial outage still serves the answering channels, but the skipped ones
  // are worth one diagnostic line so a missing dev channel is explainable from
  // the log instead of looking like discovery never tried it.
  if (failures.length > 0) {
    options.onDiagnostic?.(
      `skipped unreachable Astronomical channel(s) — ${failures.join("; ")}`,
    );
  }

  return { kind: "success", channels: catalogues };
}

async function fetchChannel(input: {
  readonly baseUrl: string;
  readonly fetchImpl?: typeof fetch;
  readonly timeoutMs: number;
}): Promise<FetchChannelResult> {
  const { baseUrl } = input;
  const fetchImpl = input.fetchImpl ?? fetch;

  try {
    const response = await fetchImpl(`${baseUrl}/models`, {
      signal: AbortSignal.timeout(input.timeoutMs),
    });

    if (!response.ok) {
      return { kind: "failure", reason: `responded ${response.status}` };
    }

    const payload: unknown = await response.json();
    return { kind: "success", models: selectChatModels(payload) };
  } catch (cause) {
    const errorMessage = cause instanceof Error ? cause.message : String(cause);
    return { kind: "failure", reason: errorMessage };
  }
}

// A model is offered only when the daemon lists the chat-completions route for
// it, because that is the only route opencode's openai-compatible adapter
// serves. Image-generation and embeddings models list no such route (and omit
// the chat limits during parsing), so they are naturally excluded.
function selectChatModels(payload: unknown): readonly AstronomicalModelRow[] {
  if (!isRecord(payload)) return [];
  const rows = payload["data"];
  if (!Array.isArray(rows)) return [];

  const models: AstronomicalModelRow[] = [];
  for (const row of rows) {
    const model = parseAstronomicalModelRow(row);
    if (model && servesChatCompletions(model)) {
      models.push(model);
    }
  }
  return models;
}

// Matched exactly because the route name is a fixed wire value; a differently
// cased route would otherwise silently drop a model the daemon is serving.
function servesChatCompletions(model: AstronomicalModelRow): boolean {
  return model.supported_endpoints.some(
    (endpoint) => endpoint === ASTRONOMICAL_CHAT_ENDPOINT,
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
