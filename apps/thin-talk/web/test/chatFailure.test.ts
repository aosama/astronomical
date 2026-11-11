import assert from "node:assert/strict";
import { test } from "node:test";
import { ChatFailure } from "../src/ChatFailure";
import { ChatFailureClassifier } from "../src/ChatFailureClassifier";
import { ChatFailureKind } from "../src/ChatFailureKind";

const ENGINE_FATAL_SENTENCE =
  "the local worker stopped after a fatal model execution error: model execution failed inside the local worker; inspect the worker log for the request-specific native error";

test("should_classify_a_streamed_error_by_the_code_the_engine_sends_beside_its_sentence", () => {
  assert.equal(ChatFailureClassifier.classifyErrorChunk("chat_out_of_memory", "this Mac ran out of graphics memory"), ChatFailureKind.OUT_OF_MEMORY);
  assert.equal(ChatFailureClassifier.classifyErrorChunk("chat_worker_unavailable", ENGINE_FATAL_SENTENCE), ChatFailureKind.WORKER_UNAVAILABLE);
  assert.equal(ChatFailureClassifier.classifyErrorChunk("chat_engine_busy", "already processing another request"), ChatFailureKind.ENGINE_BUSY);
});

test("should_fall_back_to_the_sentence_when_the_code_is_missing_or_unknown", () => {
  assert.equal(ChatFailureClassifier.classifyErrorChunk(null, "chat_engine_busy: try later"), ChatFailureKind.ENGINE_BUSY);
  assert.equal(ChatFailureClassifier.classifyErrorChunk("some_new_code", "chat_worker_unavailable: worker exited"), ChatFailureKind.WORKER_UNAVAILABLE);
  assert.equal(ChatFailureClassifier.classifyErrorChunk(null, null), ChatFailureKind.UNKNOWN);
  assert.equal(ChatFailureClassifier.classifyErrorChunk("", ""), ChatFailureKind.UNKNOWN);
});

test("should_never_show_the_engines_wording_about_workers_and_logs_as_the_headline", () => {
  for (const kind of [ChatFailureKind.WORKER_UNAVAILABLE, ChatFailureKind.OUT_OF_MEMORY, ChatFailureKind.UNKNOWN, ChatFailureKind.STALL]) {
    const failure = new ChatFailure(kind, ENGINE_FATAL_SENTENCE);
    assert.doesNotMatch(failure.headline, /worker|log/i, kind);
    assert.equal(failure.reason, ENGINE_FATAL_SENTENCE, "the engine's sentence is kept for the tooltip");
  }
});

test("should_tell_someone_out_of_memory_what_frees_it", () => {
  const failure = new ChatFailure(ChatFailureKind.OUT_OF_MEMORY, "this Mac ran out of graphics memory");
  assert.equal(failure.headline, "This Mac ran out of memory for that one.");
  assert.match(failure.nextAction, /Close other apps/);
  assert.match(failure.nextAction, /smaller model/);
});

// The engine goes quiet two ways: a crashed worker comes back by itself, and an
// engine that was stopped does not. The advice has to be true for both.
test("should_offer_a_retry_and_then_the_menu_bar_restart_when_the_engine_stops_answering", () => {
  const failure = new ChatFailure(ChatFailureKind.WORKER_UNAVAILABLE, ENGINE_FATAL_SENTENCE);
  assert.equal(failure.headline, "The engine stopped answering.");
  assert.match(failure.nextAction, /^Try again in a few seconds\./);
  assert.match(failure.nextAction, /Restart server/);
  assert.doesNotMatch(failure.nextAction, /by itself/);
});

test("should_keep_the_engines_sentence_where_it_is_the_specific_thing_to_know", () => {
  const overLength = new ChatFailure(ChatFailureKind.INVALID_REQUEST, "requested context uses 300000 tokens, exceeding the 262144-token model context window");
  assert.match(overLength.headline, /300000 tokens/);
  assert.equal(new ChatFailure(ChatFailureKind.INVALID_REQUEST).headline, "That message could not be sent.");
  assert.equal(new ChatFailure(ChatFailureKind.NO_USABLE_MODEL).headline, "No chat model is ready.");
});

test("should_give_every_failure_kind_a_headline_and_a_next_action", () => {
  for (const kind of Object.values(ChatFailureKind)) {
    const failure = new ChatFailure(kind);
    assert.ok(failure.headline.length > 0, kind);
    assert.ok(failure.nextAction.length > 0, kind);
  }
});
