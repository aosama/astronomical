import assert from "node:assert/strict";
import { test } from "node:test";
import { isLibraryModelReady, parseLibraryStarterPicks } from "../src/LibraryStarterPicks";

function catalogEntry(overrides: Record<string, unknown>): Record<string, unknown> {
  return {
    huggingface_id: "example-org/model",
    role: "chat",
    get_started: false,
    approximate_size_bytes: 1_000,
    ready_on_this_mac: false,
    ...overrides,
  };
}

test("should_pick_the_get_started_chat_and_image_models", () => {
  const catalogText = JSON.stringify({
    entries: [
      catalogEntry({ huggingface_id: "example-org/big-chat", role: "chat" }),
      catalogEntry({ huggingface_id: "example-org/starter-chat", role: "chat", get_started: true, approximate_size_bytes: 5_970_332_199 }),
      catalogEntry({ huggingface_id: "example-org/coder", role: "coding", get_started: true }),
      catalogEntry({ huggingface_id: "example-org/starter-image", role: "image_generation", get_started: true, ready_on_this_mac: true }),
    ],
  });
  const starterPicks = parseLibraryStarterPicks(catalogText);
  assert.deepEqual(starterPicks?.brain, { huggingfaceID: "example-org/starter-chat", approximateSizeBytes: 5_970_332_199 });
  assert.equal(starterPicks?.sketchbook?.huggingfaceID, "example-org/starter-image");
});

test("should_answer_null_picks_when_the_catalog_recommends_none", () => {
  const starterPicks = parseLibraryStarterPicks(JSON.stringify({ entries: [catalogEntry({ role: "coding", get_started: true })] }));
  assert.deepEqual(starterPicks, { brain: null, sketchbook: null });
});

test("should_skip_a_malformed_pick_and_use_the_next_valid_one", () => {
  const starterPicks = parseLibraryStarterPicks(JSON.stringify({
    entries: [
      null,
      catalogEntry({ get_started: true, approximate_size_bytes: "big" }),
      catalogEntry({ huggingface_id: "example-org/second", get_started: true }),
    ],
  }));
  assert.equal(starterPicks?.brain?.huggingfaceID, "example-org/second");
});

test("should_refuse_text_that_is_not_a_catalog", () => {
  assert.equal(parseLibraryStarterPicks("not json"), null);
  assert.equal(parseLibraryStarterPicks(JSON.stringify({ models: [] })), null);
});

test("should_report_a_model_ready_only_when_its_entry_says_so", () => {
  const catalogText = JSON.stringify({ entries: [catalogEntry({ huggingface_id: "example-org/ready", ready_on_this_mac: true }), catalogEntry({ huggingface_id: "example-org/not-yet" })] });
  assert.equal(isLibraryModelReady(catalogText, "example-org/ready"), true);
  assert.equal(isLibraryModelReady(catalogText, "example-org/not-yet"), false);
  assert.equal(isLibraryModelReady(catalogText, "example-org/absent"), false);
  assert.equal(isLibraryModelReady("broken", "example-org/ready"), false);
});
