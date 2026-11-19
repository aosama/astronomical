// Parse one raw /v1/models row into a normalized chat-model row, or reject it.
//
// This module owns exactly one job: turning untrusted wire data into a typed
// row or rejecting it. It deliberately does NOT decide whether a model is
// chat-capable — that capability filter belongs to the caller, so a rejected
// row and an unservable row stay distinguishable to whoever reads the result.
//
// A row is rejected when it cannot be registered without inventing a value: no
// id, or no context window, or no output budget. Image-generation and
// embeddings models omit the two limits, so they fall through as rejected here
// rather than registering as bogus chat models.

import type { AstronomicalModelRow } from "./astronomer-model.js";

export function parseAstronomicalModelRow(
  entry: unknown,
): AstronomicalModelRow | null {
  if (!isRecord(entry)) return null;

  const id = readNonEmptyString(entry["id"]);
  if (id === undefined) return null;

  const contextWindow = readPositiveInteger(entry["context_window"]);
  if (contextWindow === undefined) return null;

  const maxOutputTokens = readPositiveInteger(entry["max_output_tokens"]);
  if (maxOutputTokens === undefined) return null;

  // readStringArray already collapses a missing/empty list to [], so the
  // `?? []` only narrows the type; an absent modality list means text-only
  // input, which is a real (if unadvertised) capability, not a failure.
  const inputModalities = readStringArray(entry["input_modalities"]) ?? [];

  // A corrupt supported_endpoints is a parse failure (reject the row), because
  // the capability filter depends on it and an empty list would silently drop
  // a chat model.
  const supportedEndpoints = readStringArray(entry["supported_endpoints"]);
  if (supportedEndpoints === undefined) return null;

  return {
    id,
    context_window: contextWindow,
    max_output_tokens: maxOutputTokens,
    input_modalities: inputModalities,
    supports_reasoning: readBoolean(entry["supports_reasoning"]),
    supports_tool_calls: readBoolean(entry["supports_tool_calls"]),
    supports_structured_outputs: readBoolean(
      entry["supports_structured_outputs"],
    ),
    supported_endpoints: supportedEndpoints,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
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

function readBoolean(value: unknown): boolean {
  return value === true;
}

// A missing or empty array means "advertises no values of this kind", while a
// present-but-wrongly-typed array is a parse failure. This keeps an absent
// optional fact distinct from a corrupt one.
function readStringArray(value: unknown): readonly string[] | undefined {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) return undefined;
  return value.every((item) => typeof item === "string")
    ? (value as readonly string[])
    : undefined;
}
