// Row-level parsing tests: what a well-formed /v1/models row becomes, and what
// is rejected rather than registered on an invented value.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { parseAstronomicalModelRow } from "../../dist/astronomer-model-row.js";

function row(overrides = {}) {
  return {
    id: "vision-model",
    object: "model",
    context_window: 131_072,
    max_output_tokens: 65_535,
    input_modalities: ["text", "image"],
    output_modalities: ["text"],
    supports_reasoning: true,
    supports_tool_calls: true,
    supports_structured_outputs: true,
    supported_endpoints: ["/v1/chat/completions"],
    ...overrides,
  };
}

describe("a well-formed wire row becomes a fully resolved chat row", () => {
  it("carries every field the plugin reads", () => {
    assert.deepEqual(parseAstronomicalModelRow(row()), {
      id: "vision-model",
      context_window: 131_072,
      max_output_tokens: 65_535,
      input_modalities: ["text", "image"],
      supports_reasoning: true,
      supports_tool_calls: true,
      supports_structured_outputs: true,
      supported_endpoints: ["/v1/chat/completions"],
    });
  });

  it("trims a padded id, so a padded id cannot become a second provider key", () => {
    assert.equal(
      parseAstronomicalModelRow(row({ id: "  spaced  " })).id,
      "spaced",
    );
  });

  it("drops wire facts nothing reads (created, owned_by, output_modalities, reasoning_format)", () => {
    const parsed = parseAstronomicalModelRow(
      row({
        created: 1_792_257_274,
        owned_by: "astronomical",
        output_modalities: ["text"],
        reasoning_format: "openai_chat_reasoning_content",
      }),
    );
    assert.equal(Object.hasOwn(parsed, "created"), false);
    assert.equal(Object.hasOwn(parsed, "owned_by"), false);
    assert.equal(Object.hasOwn(parsed, "output_modalities"), false);
    assert.equal(Object.hasOwn(parsed, "reasoning_format"), false);
  });
});

describe("a row is rejected rather than registered on an invented value", () => {
  it("rejects a value that is not an object at all", () => {
    assert.equal(parseAstronomicalModelRow(null), null);
    assert.equal(parseAstronomicalModelRow("vision-model"), null);
    assert.equal(parseAstronomicalModelRow([]), null);
  });

  it("rejects a missing, blank, or non-string id", () => {
    assert.equal(parseAstronomicalModelRow(row({ id: "" })), null);
    assert.equal(parseAstronomicalModelRow(row({ id: "   " })), null);
    assert.equal(parseAstronomicalModelRow(row({ id: 42 })), null);
    assert.equal(parseAstronomicalModelRow(row({ id: undefined })), null);
  });

  it("rejects a missing or non-positive context window", () => {
    assert.equal(
      parseAstronomicalModelRow(row({ context_window: undefined })),
      null,
    );
    assert.equal(parseAstronomicalModelRow(row({ context_window: 0 })), null);
    assert.equal(
      parseAstronomicalModelRow(row({ context_window: -100 })),
      null,
    );
  });

  it("rejects a missing or non-positive output budget", () => {
    assert.equal(
      parseAstronomicalModelRow(row({ max_output_tokens: undefined })),
      null,
    );
    assert.equal(
      parseAstronomicalModelRow(row({ max_output_tokens: 0 })),
      null,
    );
  });

  it("rejects a corrupt supported_endpoints, which the capability filter needs", () => {
    assert.equal(
      parseAstronomicalModelRow(row({ supported_endpoints: "nope" })),
      null,
    );
    assert.equal(
      parseAstronomicalModelRow(row({ supported_endpoints: [1, 2] })),
      null,
    );
  });

  it("does not claim tool support unless the gateway advertises it", () => {
    assert.equal(
      parseAstronomicalModelRow(row({ supports_tool_calls: undefined }))
        .supports_tool_calls,
      false,
    );
    assert.equal(
      parseAstronomicalModelRow(row({ supports_tool_calls: true }))
        .supports_tool_calls,
      true,
    );
  });

  it("treats a missing modality list as text-only input", () => {
    assert.deepEqual(
      parseAstronomicalModelRow(row({ input_modalities: undefined }))
        .input_modalities,
      [],
    );
  });
});
