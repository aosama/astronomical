// Translate one channel's chat models into an OpenCode provider definition.
// The models are checked against the chat filter again here, so this adapter
// cannot publish a non-chat model even if a caller skips discovery.

import { Model, Provider } from "@opencode/plugin";
import { acceptsImageInput, servesChatCompletions } from "./chat-models.ts";
import {
  ASTRONOMICAL_CHANNEL_PROVIDER_IDS,
  ASTRONOMICAL_CHANNEL_PROVIDER_NAMES,
  ASTRONOMICAL_INFERENCE_TIMEOUT_MS,
  ASTRONOMICAL_OPENAI_COMPATIBLE_PACKAGE,
  type AstronomicalChannelInstance,
} from "./constants.ts";
import type { AstronomicalModelRow } from "./model-row.ts";

export interface AstronomicalProviderDefinition {
  readonly info: Provider.Info;
  readonly models: readonly Model.Info[];
}

// The local runner does not bill per token, so every price is zero.
function zeroCost(): Model.Cost {
  return {
    input: Model.Cost.fields.input.zero,
    output: Model.Cost.fields.output.zero,
    cache: {
      read: Model.Cost.fields.cache.fields.read.zero,
      write: Model.Cost.fields.cache.fields.write.zero,
    },
  };
}

function toModelInfo(
  providerId: Provider.ID,
  model: AstronomicalModelRow,
): Model.Info {
  return {
    ...Model.Info.default(providerId, Model.ID.make(model.id)),
    name: model.id,
    capabilities: {
      ...Model.Capabilities.default(),
      tools: model.supports_tool_calls,
      input: acceptsImageInput(model) ? ["text", "image"] : ["text"],
      output: ["text"],
    },
    ...(model.supports_reasoning
      ? { compatibility: { reasoningField: "reasoning_content" } }
      : {}),
    cost: [zeroCost()],
    limit: {
      context: model.context_window,
      output: model.max_output_tokens,
    },
  };
}

/**
 * Builds the keyless provider and its chat models for one live channel.
 * Provider identity (id and display name) comes from the channel instance, so
 * Stable and Development register as sibling providers instead of fighting
 * over one id.
 */
export function buildAstronomicalProviderDefinition(input: {
  readonly instance: AstronomicalChannelInstance;
  readonly baseUrl: string;
  readonly models: readonly AstronomicalModelRow[];
}): AstronomicalProviderDefinition {
  const providerId: Provider.ID = Provider.ID.make(
    ASTRONOMICAL_CHANNEL_PROVIDER_IDS[input.instance],
  );
  const models: Model.Info[] = [];
  for (const model of input.models) {
    if (servesChatCompletions(model))
      models.push(toModelInfo(providerId, model));
  }
  const info: Provider.Info = {
    ...Provider.Info.empty(providerId),
    name: ASTRONOMICAL_CHANNEL_PROVIDER_NAMES[input.instance],
    activation: "enabled",
    package: ASTRONOMICAL_OPENAI_COMPATIBLE_PACKAGE,
    settings: {
      baseURL: input.baseUrl,
      // A cold model load can delay response headers beyond OpenCode's default.
      headerTimeout: ASTRONOMICAL_INFERENCE_TIMEOUT_MS,
      timeout: ASTRONOMICAL_INFERENCE_TIMEOUT_MS,
    },
  };
  return { info, models };
}
