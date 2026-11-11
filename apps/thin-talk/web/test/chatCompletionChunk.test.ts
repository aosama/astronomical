import assert from "node:assert/strict";
import { test } from "node:test";
import { ChatCompletionChunk } from "../src/ChatCompletionChunk";
import { ChatFailure } from "../src/ChatFailure";
import { ChatFailureKind } from "../src/ChatFailureKind";

test("should_decode_text_reasoning_and_finish_reason_from_a_data_line", () => {
  const chunk = ChatCompletionChunk.fromServerSentEventLine(
    'data: {"choices":[{"delta":{"reasoning_content":"thinks","content":"says"},"finish_reason":"stop"}]}',
  );
  assert.ok(chunk);
  assert.equal(chunk.reasoningDelta, "thinks");
  assert.equal(chunk.textDelta, "says");
  assert.equal(chunk.finishReason, "stop");
});

test("should_ignore_lines_that_carry_no_chunk", () => {
  for (const lineWithoutChunk of ["", ": keep-alive", "event: ping", "data:", "data: [DONE]", "data: {not json", "data: 42"]) {
    assert.equal(ChatCompletionChunk.fromServerSentEventLine(lineWithoutChunk), null, lineWithoutChunk);
  }
});

test("should_treat_an_empty_or_null_finish_reason_as_still_streaming", () => {
  for (const finishReason of ['""', "null"]) {
    const chunk = ChatCompletionChunk.fromServerSentEventLine(
      `data: {"choices":[{"delta":{"content":"a"},"finish_reason":${finishReason}}]}`,
    );
    assert.equal(chunk?.finishReason, null, finishReason);
  }
});

test("should_fold_every_choice_in_a_chunk_in_order", () => {
  const chunk = ChatCompletionChunk.fromServerSentEventLine(
    'data: {"choices":[{"delta":{"content":"one "}},{"delta":{"content":"two"},"finish_reason":"length"}]}',
  );
  assert.equal(chunk?.textDelta, "one two");
  assert.equal(chunk?.finishReason, "length");
});

test("should_return_an_empty_chunk_for_a_usage_only_line", () => {
  const chunk = ChatCompletionChunk.fromServerSentEventLine(
    'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":5}}',
  );
  assert.ok(chunk);
  assert.equal(chunk.textDelta, "");
  assert.equal(chunk.reasoningDelta, "");
  assert.equal(chunk.finishReason, null);
});

test("should_throw_a_classified_failure_for_a_streamed_error_object", () => {
  assert.throws(
    () => ChatCompletionChunk.fromServerSentEventLine('data: {"error":{"message":"chat_engine_busy: try later"}}'),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.ENGINE_BUSY,
  );
});

test("should_classify_an_error_object_without_a_message_as_unknown", () => {
  assert.throws(
    () => ChatCompletionChunk.fromServerSentEventLine('data: {"error":{"code":500}}'),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.UNKNOWN && error.reason === null,
  );
});

test("should_ignore_an_error_field_that_is_not_an_object", () => {
  const chunk = ChatCompletionChunk.fromServerSentEventLine('data: {"error":"text","choices":[{"delta":{"content":"still here"}}]}');
  assert.equal(chunk?.textDelta, "still here");
});

test("should_decode_a_whole_tool_call_and_a_later_arguments_piece", () => {
  const firstPiece = ChatCompletionChunk.fromServerSentEventLine(
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"get_current_time","arguments":"{\\"zo"}}]}}]}',
  );
  assert.deepEqual(firstPiece?.toolCallFragments, [{ toolCallIndex: 0, toolCallID: "call-1", toolName: "get_current_time", argumentsFragment: '{"zo' }]);
  const laterPiece = ChatCompletionChunk.fromServerSentEventLine(
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ne\\":1}"}}]}}]}',
  );
  assert.deepEqual(laterPiece?.toolCallFragments, [{ toolCallIndex: 0, toolCallID: null, toolName: null, argumentsFragment: 'ne":1}' }]);
});

test("should_skip_a_tool_call_piece_without_a_usable_index", () => {
  for (const badIndex of ['"0"', "-1", "1.5", "null"]) {
    const chunk = ChatCompletionChunk.fromServerSentEventLine(
      `data: {"choices":[{"delta":{"tool_calls":[{"index":${badIndex},"id":"call-1","function":{"name":"x","arguments":"{}"}}]}}]}`,
    );
    assert.deepEqual(chunk?.toolCallFragments, [], badIndex);
  }
});

test("should_classify_an_error_chunk_by_its_code_field_as_the_engine_really_sends_it", () => {
  const engineErrorLine =
    'data: {"error":{"message":"this Mac ran out of graphics memory while running the model; close other apps that use a lot of memory, or use a smaller model","type":"service_unavailable","code":"chat_out_of_memory"}}';
  assert.throws(
    () => ChatCompletionChunk.fromServerSentEventLine(engineErrorLine),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.OUT_OF_MEMORY && error.reason !== null && error.reason.startsWith("this Mac ran out"),
  );
  assert.throws(
    () => ChatCompletionChunk.fromServerSentEventLine('data: {"error":{"message":"the local worker stopped after a fatal model execution error: x","code":"chat_worker_unavailable"}}'),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.WORKER_UNAVAILABLE,
  );
});
