// One /v1/models row: its parsed shape, and the parser that turns untrusted
// wire data into that shape or rejects it.
//
// The parser deliberately does NOT decide whether a model is chat-capable;
// that belongs to chat-models.ts, so a rejected row and an unservable row stay
// distinguishable to whoever reads the result.
//
// A row is rejected when it cannot be registered without inventing a value: no
// id, or no context window, or no output budget. Image, embedding, speech and
// decision models omit the two limits, so they fall through as rejected here
// rather than registering as bogus chat models.

import { isRecord } from "./wire-values.ts";

// Every field the plugin reads is required and already resolved, so
// downstream code never re-applies a fallback or invents a default. Fields the
// app publishes but the plugin never reads (created, owned_by,
// output_modalities, reasoning_format, tool_call_format, structured output
// support) are deliberately absent: carrying unread wire facts invites
// someone to act on a value nothing validated.
export type AstronomicalModelRow = {
  readonly id: string;
  readonly context_window: number;
  readonly max_output_tokens: number;
  readonly input_modalities: readonly string[];
  readonly supports_reasoning: boolean;
  readonly supports_tool_calls: boolean;
  readonly supported_endpoints: readonly string[];
};

export function parseAstronomicalModelRow(
  entry: unknown,
): AstronomicalModelRow | null {
  if (!isRecord(entry)) return null;
  const id = readNonEmptyString(entry["id"]);
  const contextWindow = readPositiveInteger(entry["context_window"]);
  const maxOutputTokens = readPositiveInteger(entry["max_output_tokens"]);
  if (id === undefined || contextWindow === undefined) return null;
  if (maxOutputTokens === undefined) return null;

  // A missing or empty modality list means text-only input, a real (if
  // unadvertised) capability; a corrupt one rejects the row.
  const inputModalities = readStringArray(entry["input_modalities"]);
  // A corrupt supported_endpoints rejects the row, because the chat filter
  // depends on it and an empty list would silently drop a chat model.
  const supportedEndpoints = readStringArray(entry["supported_endpoints"]);
  if (inputModalities === undefined || supportedEndpoints === undefined) {
    return null;
  }

  return {
    id,
    context_window: contextWindow,
    max_output_tokens: maxOutputTokens,
    input_modalities: inputModalities,
    supports_reasoning: entry["supports_reasoning"] === true,
    supports_tool_calls: entry["supports_tool_calls"] === true,
    supported_endpoints: supportedEndpoints,
  };
}

function readNonEmptyString(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

// A positive integer limit is the smallest value that can truncate a request,
// so both zero and negative values are rejected as malformed rather than
// registering a model that silently cuts every reply short.
function readPositiveInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? Math.floor(value)
    : undefined;
}

// A missing or empty array means "advertises no values of this kind", while a
// present-but-wrongly-typed array is a parse failure. This keeps an absent
// optional fact distinct from a corrupt one.
function readStringArray(value: unknown): readonly string[] | undefined {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) return undefined;
  for (const item of value) {
    if (typeof item !== "string") return undefined;
  }
  return value as readonly string[];
}
