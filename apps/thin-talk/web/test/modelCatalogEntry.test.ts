import assert from "node:assert/strict";
import { test } from "node:test";
import { ModelCatalogEntry } from "../src/ModelCatalogEntry";

const BASE_ENTRY = { id: "chat-model", input_modalities: ["text"], supported_endpoints: ["/v1/chat/completions"] };

test("should_read_tool_support_only_from_an_explicit_true", () => {
  assert.equal(ModelCatalogEntry.fromPlain({ ...BASE_ENTRY, supports_tool_calls: true })?.supportsToolCalls, true);
  for (const notTrue of [false, "true", 1, null, undefined]) {
    assert.equal(ModelCatalogEntry.fromPlain({ ...BASE_ENTRY, supports_tool_calls: notTrue })?.supportsToolCalls, false, String(notTrue));
  }
});

test("should_still_reject_an_entry_without_an_id", () => {
  assert.equal(ModelCatalogEntry.fromPlain({ supports_tool_calls: true }), null);
});

test("should_recognise_image_models_by_their_generation_endpoint", () => {
  assert.equal(ModelCatalogEntry.fromPlain({ id: "FLUX", supported_endpoints: ["/v1/images/generations"] })?.generatesImages, true);
  assert.equal(ModelCatalogEntry.fromPlain(BASE_ENTRY)?.generatesImages, false);
  assert.equal(ModelCatalogEntry.fromPlain({ id: "no-endpoints" })?.generatesImages, false);
});
