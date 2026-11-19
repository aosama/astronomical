// Translate the validated Astronomical catalogue into OpenCode V2 source
// definitions. The catalogue is filtered again here so this adapter cannot
// accidentally publish a non-chat model if another caller bypasses discovery.

import { Model, Provider } from "@opencode/plugin";
import {
  ASTRONOMICAL_CHANNEL_PROVIDER_IDS,
  ASTRONOMICAL_CHANNEL_PROVIDER_NAMES,
  ASTRONOMICAL_CHAT_ENDPOINT,
  ASTRONOMICAL_INFERENCE_TIMEOUT_MS,
  ASTRONOMICAL_OPENAI_COMPATIBLE_PACKAGE,
  type AstronomicalChannelInstance,
} from "./constants.js";
import type { AstronomicalModelRow } from "./astronomer-model.js";

export interface AstronomicalProviderDefinition {
  readonly info: Provider.Info;
  readonly models: readonly Model.Info[];
}

function servesChatCompletions(model: AstronomicalModelRow): boolean {
  return model.supported_endpoints.some(
    (endpoint: string): boolean => endpoint === ASTRONOMICAL_CHAT_ENDPOINT,
  );
}

function isVisionModel(model: AstronomicalModelRow): boolean {
  return model.input_modalities.some(
    (modality: string): boolean => modality.toLowerCase() === "image",
  );
}

function toModelInfo(
  providerId: Provider.ID,
  model: AstronomicalModelRow,
): Model.Info {
  const inputModalities: string[] = isVisionModel(model)
    ? ["text", "image"]
    : ["text"];
  const zeroCost: Model.Cost = {
    input: Model.Cost.fields.input.zero,
    output: Model.Cost.fields.output.zero,
    cache: {
      read: Model.Cost.fields.cache.fields.read.zero,
      write: Model.Cost.fields.cache.fields.write.zero,
    },
  };

  return {
    ...Model.Info.default(providerId, Model.ID.make(model.id)),
    name: model.id,
    capabilities: {
      ...Model.Capabilities.default(),
      tools: model.supports_tool_calls,
      input: inputModalities,
      output: ["text"],
    },
    ...(model.supports_reasoning
      ? { compatibility: { reasoningField: "reasoning_content" } }
      : {}),
    cost: [zeroCost],
    limit: {
      context: model.context_window,
      output: model.max_output_tokens,
    },
  };
}

/**
 * Builds the keyless provider source and chat model definitions for one live
 * Astronomical channel. Provider identity (id and display name) comes from the
 * channel instance, so stable and dev register as sibling providers instead of
 * fighting over one id.
 */
export function buildAstronomicalProviderDefinition(input: {
  readonly instance: AstronomicalChannelInstance;
  readonly baseUrl: string;
  readonly models: readonly AstronomicalModelRow[];
}): AstronomicalProviderDefinition {
  const providerId: Provider.ID = Provider.ID.make(
    ASTRONOMICAL_CHANNEL_PROVIDER_IDS[input.instance],
  );
  const models: Model.Info[] = input.models
    .filter((model: AstronomicalModelRow): boolean =>
      servesChatCompletions(model),
    )
    .map((model: AstronomicalModelRow): Model.Info =>
      toModelInfo(providerId, model),
    );
  const info: Provider.Info = {
    ...Provider.Info.empty(providerId),
    name: ASTRONOMICAL_CHANNEL_PROVIDER_NAMES[input.instance],
    activation: "enabled",
    package: ASTRONOMICAL_OPENAI_COMPATIBLE_PACKAGE,
    settings: {
      baseURL: input.baseUrl,
      // Cold local model loads can delay response headers beyond the V2 default.
      headerTimeout: ASTRONOMICAL_INFERENCE_TIMEOUT_MS,
      timeout: ASTRONOMICAL_INFERENCE_TIMEOUT_MS,
    },
  };

  return { info, models };
}
