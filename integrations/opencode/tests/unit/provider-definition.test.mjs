// Provider-definition tests cover the source shape, advertised capabilities,
// the keyless local runtime settings, and the per-channel provider identity.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { buildAstronomicalProviderDefinition } from "../../src/provider-definition.ts";
import { STABLE, representativeModels } from "../support/test-helpers.mjs";

describe("the provider definition reflects the live model list", () => {
  it("should_use_the_native_openai_compatible_runtime_and_have_no_credential_fields", () => {
    const definition = buildAstronomicalProviderDefinition({
      instance: "stable",
      baseUrl: STABLE,
      models: representativeModels,
    });

    assert.equal(definition.info.name, "Astronomical");
    assert.equal(
      definition.info.package,
      "@opencode/ai/providers/openai-compatible",
    );
    assert.equal(definition.info.settings.baseURL, STABLE);
    assert.equal(definition.info.settings.timeout, 600_000);
    assert.equal(definition.info.settings.headerTimeout, 600_000);
    assert.equal(Object.hasOwn(definition.info, "env"), false);
    assert.equal(Object.hasOwn(definition.info.settings, "apiKey"), false);
  });

  it("should_give_stable_and_dev_distinct_provider_ids_so_both_channels_can_coexist", () => {
    const stableDefinition = buildAstronomicalProviderDefinition({
      instance: "stable",
      baseUrl: STABLE,
      models: representativeModels,
    });
    const devDefinition = buildAstronomicalProviderDefinition({
      instance: "development",
      baseUrl: "http://127.0.0.1:6733/v1",
      models: representativeModels,
    });

    assert.equal(stableDefinition.info.id, "astronomical");
    assert.equal(stableDefinition.info.name, "Astronomical");
    assert.equal(devDefinition.info.id, "astronomical-development");
    assert.equal(devDefinition.info.name, "Astronomical Development");
    assert.notEqual(stableDefinition.info.id, devDefinition.info.id);
  });

  it("should_map_a_custom_base_url_override_onto_the_primary_provider_identity", () => {
    // A "baseUrl" option replaces stable as "the" Astronomical
    // endpoint, so it inherits the primary id rather than inventing a third.
    const definition = buildAstronomicalProviderDefinition({
      instance: "custom",
      baseUrl: "http://localhost:9999/v1",
      models: representativeModels,
    });

    assert.equal(definition.info.id, "astronomical");
    assert.equal(definition.info.name, "Astronomical");
  });

  it("should_include_chat_and_vision_models_but_exclude_image_generation_and_embedding_models", () => {
    const definition = buildAstronomicalProviderDefinition({
      instance: "stable",
      baseUrl: STABLE,
      models: representativeModels,
    });

    assert.deepEqual(
      definition.models.map((model) => model.id),
      ["text-only-model", "vision-model"],
    );
  });

  it("should_map_vision_reasoning_tools_zero_cost_and_limits_into_opencode_fields", () => {
    const definition = buildAstronomicalProviderDefinition({
      instance: "stable",
      baseUrl: STABLE,
      models: representativeModels,
    });
    const visionModel = definition.models.find(
      (model) => model.id === "vision-model",
    );

    assert.ok(visionModel);
    assert.deepEqual(visionModel.capabilities, {
      tools: true,
      input: ["text", "image"],
      output: ["text"],
    });
    assert.deepEqual(visionModel.compatibility, {
      reasoningField: "reasoning_content",
    });
    assert.deepEqual(visionModel.cost, [
      { input: 0, output: 0, cache: { read: 0, write: 0 } },
    ]);
    assert.deepEqual(visionModel.limit, {
      context: 131_072,
      output: 65_535,
    });
  });

  it("should_mark_text_only_models_accurately_and_leave_non_reasoning_compatibility_unset", () => {
    const nonReasoningChatModel = {
      ...representativeModels[0],
      id: "plain-chat-model",
      supports_reasoning: false,
      supports_tool_calls: false,
    };
    const definition = buildAstronomicalProviderDefinition({
      instance: "stable",
      baseUrl: STABLE,
      models: [representativeModels[0], nonReasoningChatModel],
    });
    const textOnlyModel = definition.models.find(
      (model) => model.id === "text-only-model",
    );
    const plainChatModel = definition.models.find(
      (model) => model.id === "plain-chat-model",
    );

    assert.ok(textOnlyModel);
    assert.deepEqual(textOnlyModel.capabilities.input, ["text"]);
    assert.ok(plainChatModel);
    assert.equal(plainChatModel.capabilities.tools, false);
    assert.equal(plainChatModel.compatibility, undefined);
  });
});
