import assert from "node:assert/strict";
import { test } from "node:test";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { CONTEXT_BUDGET_SHARE, estimatedTokens, trimToContextBudget } from "../src/AgentContextTrimmer";
import { daemonModel, emptyAssistantMessage } from "../src/DaemonModelDescription";

const MODEL = daemonModel("chat-model");

function question(text: string): AgentMessage {
  return { role: "user", content: text, timestamp: 0 };
}

function answer(text: string): AgentMessage {
  const assistantMessage = emptyAssistantMessage(MODEL);
  assistantMessage.content = [{ type: "text", text }];
  assistantMessage.timestamp = 0;
  return assistantMessage;
}

const INSTRUCTIONS: AgentMessage = { role: "system", content: "Be brief.", timestamp: 0 } as AgentMessage;
const LONG_TEXT = "w".repeat(4_000);

test("should_leave_a_conversation_inside_the_budget_untouched", () => {
  const transcript = [INSTRUCTIONS, question("Who wrote Hamlet?"), answer("Shakespeare."), question("When?")];
  assert.deepEqual(trimToContextBudget(transcript, 100_000), transcript);
});

test("should_drop_the_oldest_whole_turns_and_keep_the_instructions", () => {
  const transcript = [INSTRUCTIONS, question(LONG_TEXT), answer(LONG_TEXT), question("Second"), answer("Two"), question("Third?")];
  const contextWindowTokens = Math.ceil(estimatedTokens([INSTRUCTIONS, question("Second"), answer("Two"), question("Third?")]) / CONTEXT_BUDGET_SHARE) + 10;
  const trimmedTranscript = trimToContextBudget(transcript, contextWindowTokens);
  assert.deepEqual(trimmedTranscript, [INSTRUCTIONS, question("Second"), answer("Two"), question("Third?")]);
});

test("should_keep_a_tool_call_with_its_result", () => {
  const toolCallAnswer = emptyAssistantMessage(MODEL);
  toolCallAnswer.content = [{ type: "toolCall", id: "call-1", name: "get_current_time", arguments: {} }];
  const toolResult = { role: "toolResult", toolCallId: "call-1", toolName: "get_current_time", content: [{ type: "text", text: LONG_TEXT }], isError: false, timestamp: 0 } as AgentMessage;
  const transcript = [question("Time?"), toolCallAnswer, toolResult, answer("Two PM."), question("Thanks")];
  const trimmedTranscript = trimToContextBudget(transcript, 50);
  assert.deepEqual(trimmedTranscript, [question("Thanks")], "the whole first turn goes, call and result together");
});

test("should_keep_the_latest_turn_even_when_it_alone_is_too_long", () => {
  const transcript = [question("Hi"), answer("Hello"), question(LONG_TEXT)];
  assert.deepEqual(trimToContextBudget(transcript, 10), [question(LONG_TEXT)]);
});

test("should_return_a_transcript_without_questions_as_it_is", () => {
  assert.deepEqual(trimToContextBudget([INSTRUCTIONS], 1), [INSTRUCTIONS]);
});

test("should_keep_a_transcript_exactly_at_the_budget_and_trim_one_token_over", () => {
  const transcript = [question("a".repeat(40)), answer("b".repeat(40)), question("c".repeat(40))];
  const totalTokens = estimatedTokens(transcript);
  const exactWindow = Math.ceil(totalTokens / CONTEXT_BUDGET_SHARE);
  assert.equal(trimToContextBudget(transcript, exactWindow).length, 3);
  const smallerWindow = Math.floor((totalTokens - 1) / CONTEXT_BUDGET_SHARE);
  assert.deepEqual(trimToContextBudget(transcript, smallerWindow), [question("c".repeat(40))]);
});

test("should_treat_a_window_of_zero_or_less_as_room_for_only_the_latest_turn", () => {
  const transcript = [INSTRUCTIONS, question("Hi"), answer("Hello"), question("Bye")];
  for (const unusableWindow of [0, -100]) {
    assert.deepEqual(trimToContextBudget(transcript, unusableWindow), [INSTRUCTIONS, question("Bye")], String(unusableWindow));
  }
});

test("should_keep_long_instructions_and_the_latest_turn_whatever_the_budget", () => {
  const longInstructions = { role: "system", content: LONG_TEXT, timestamp: 0 } as AgentMessage;
  const transcript = [longInstructions, question("Hi"), answer("Hello"), question("Bye")];
  assert.deepEqual(trimToContextBudget(transcript, 100), [longInstructions, question("Bye")]);
});

test("should_count_what_the_model_reads_not_bookkeeping_fields", () => {
  assert.equal(estimatedTokens([answer("abcd")]), 1, "usage, model and timestamps are not counted");
  assert.equal(estimatedTokens([question("abcdefgh")]), 2);
});

test("should_count_an_image_as_a_fixed_block_of_tokens_not_by_its_base64_length", () => {
  const hugeBase64 = "A".repeat(400_000);
  const imageTokens = estimatedTokens([{
    role: "toolResult", toolCallId: "call-1", toolName: "generate_image", isError: false, timestamp: 0,
    content: [{ type: "image", data: hugeBase64, mimeType: "image/png" }],
  } as AgentMessage]);
  assert.equal(imageTokens, 2_000);
});
