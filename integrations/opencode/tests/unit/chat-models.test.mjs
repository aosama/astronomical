// Chat model choice tests: which rows OpenCode is offered, and what each row
// is said to accept.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  acceptsImageInput,
  selectChatModels,
  servesChatCompletions,
} from "../../src/chat-models.ts";
import { representativeModels } from "../support/test-helpers.mjs";

function parsedRow(overrides = {}) {
  return {
    id: "text-only-model",
    context_window: 262_144,
    max_output_tokens: 65_535,
    input_modalities: ["text"],
    supports_reasoning: false,
    supports_tool_calls: false,
    supported_endpoints: ["/v1/chat/completions"],
    ...overrides,
  };
}

describe("servesChatCompletions matches the chat route exactly", () => {
  it("should_accept_a_row_that_lists_the_chat_route", () => {
    assert.equal(servesChatCompletions(parsedRow()), true);
  });

  it("should_reject_a_row_with_a_lookalike_or_missing_route", () => {
    for (const supportedEndpoints of [
      [],
      ["/v1/responses"],
      ["/v1/chat/completions/"],
      ["/V1/CHAT/COMPLETIONS"],
    ]) {
      assert.equal(
        servesChatCompletions(
          parsedRow({ supported_endpoints: supportedEndpoints }),
        ),
        false,
      );
    }
  });
});

describe("acceptsImageInput reads the input modalities", () => {
  it("should_accept_image_in_any_letter_case", () => {
    assert.equal(
      acceptsImageInput(parsedRow({ input_modalities: ["text", "IMAGE"] })),
      true,
    );
  });

  it("should_reject_text_only_and_empty_modality_lists", () => {
    assert.equal(acceptsImageInput(parsedRow()), false);
    assert.equal(acceptsImageInput(parsedRow({ input_modalities: [] })), false);
  });
});

describe("selectChatModels keeps chat models in the app's order", () => {
  it("should_keep_the_two_chat_models_of_the_shared_listing", () => {
    assert.deepEqual(
      selectChatModels({ data: representativeModels }).map((model) => model.id),
      ["text-only-model", "vision-model"],
    );
  });

  it("should_skip_a_malformed_row_without_dropping_its_neighbours", () => {
    const listing = {
      data: [{ id: "" }, "not a row", null, representativeModels[1]],
    };
    assert.deepEqual(
      selectChatModels(listing).map((model) => model.id),
      ["vision-model"],
    );
  });

  it("should_return_nothing_for_a_payload_that_is_not_the_list_envelope", () => {
    for (const payload of [null, [], "text", {}, { data: {} }]) {
      assert.deepEqual(selectChatModels(payload), []);
    }
  });
});
