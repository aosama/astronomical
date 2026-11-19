// Which models OpenCode can use, and what each one accepts. OpenCode's
// OpenAI-compatible runtime speaks only Chat Completions, so a model is offered
// exactly when the app lists that route for it. This is the one place that
// decision lives; the provider definition asks the same functions.

import {
  parseAstronomicalModelRow,
  type AstronomicalModelRow,
} from "./model-row.js";
import { ASTRONOMICAL_CHAT_ENDPOINT } from "./constants.js";
import { isRecord } from "./wire-values.js";

/**
 * Matched exactly because the route name is a fixed wire value; a loose match
 * could admit a model whose only route OpenCode cannot call.
 */
export function servesChatCompletions(model: AstronomicalModelRow): boolean {
  return model.supported_endpoints.includes(ASTRONOMICAL_CHAT_ENDPOINT);
}

/** True when the model accepts pictures in its input. */
export function acceptsImageInput(model: AstronomicalModelRow): boolean {
  for (const modality of model.input_modalities) {
    if (modality.toLowerCase() === "image") return true;
  }
  return false;
}

/**
 * The chat models in one /v1/models payload, in the app's order. A payload
 * that is not the expected `{ data: [...] }` envelope yields no models, and a
 * malformed row is skipped rather than failing the whole list.
 */
export function selectChatModels(
  payload: unknown,
): readonly AstronomicalModelRow[] {
  if (!isRecord(payload) || !Array.isArray(payload["data"])) return [];
  const chatModels: AstronomicalModelRow[] = [];
  for (const wireRow of payload["data"]) {
    const model = parseAstronomicalModelRow(wireRow);
    if (model !== null && servesChatCompletions(model)) chatModels.push(model);
  }
  return chatModels;
}
