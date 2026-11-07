import assert from "node:assert/strict";
import { test } from "node:test";
import { ChatMessageState } from "../src/ChatMessageState";
import { ConversationLog } from "../src/ConversationLog";
import { ToolStep, ToolStepState } from "../src/ToolStep";

test("should_leave_the_turn_being_answered_out_of_the_request", () => {
  const log = new ConversationLog();
  log.appendUserTurn("Who wrote Hamlet?");
  log.openAssistantTurn();
  assert.deepEqual(log.requestMessages().map((message) => message.content), ["Who wrote Hamlet?"]);
});

test("should_send_a_finished_answer_with_the_next_question", () => {
  const log = new ConversationLog();
  log.appendUserTurn("Who wrote Hamlet?");
  log.openAssistantTurn();
  log.applyTextDelta("Shakespeare.");
  log.markAssistantTurn(ChatMessageState.COMPLETE);
  log.assistantMessageID = null;
  log.appendUserTurn("When?");
  log.openAssistantTurn();
  assert.deepEqual(log.requestMessages().map((message) => message.content), ["Who wrote Hamlet?", "Shakespeare.", "When?"]);
});

test("should_leave_failed_turns_out_of_the_request", () => {
  const log = new ConversationLog();
  log.appendUserTurn("First");
  log.openAssistantTurn();
  log.applyTextDelta("partial");
  log.markAssistantTurn(ChatMessageState.FAILED);
  log.assistantMessageID = null;
  log.appendUserTurn("Second");
  assert.deepEqual(log.requestMessages().map((message) => message.content), ["First", "Second"]);
});

test("should_send_an_empty_history_when_nothing_was_asked", () => {
  assert.deepEqual(new ConversationLog().requestMessages(), []);
});

test("should_leave_both_the_failed_attempt_and_the_retry_out_of_the_request", () => {
  const log = new ConversationLog();
  log.appendUserTurn("Who wrote Hamlet?");
  log.openAssistantTurn();
  log.markAssistantTurn(ChatMessageState.FAILED);
  log.assistantMessageID = null;
  log.openAssistantTurn();
  assert.deepEqual(log.requestMessages().map((message) => message.content), ["Who wrote Hamlet?"]);
});

test("should_append_new_tool_steps_and_replace_one_by_its_call_id", () => {
  const log = new ConversationLog();
  log.appendUserTurn("What time is it?");
  log.openAssistantTurn();
  const timeStep = (toolCallID: string, state: ToolStepState): ToolStep =>
    new ToolStep({ toolCallID, toolName: "get_current_time", label: "Current time", argumentsJSON: "{}", resultText: "", state });
  log.applyToolStep(timeStep("call-1", ToolStepState.RUNNING));
  log.applyToolStep(timeStep("call-2", ToolStepState.RUNNING));
  log.applyToolStep(timeStep("call-1", ToolStepState.SUCCEEDED));
  assert.deepEqual(log.messages[1]?.toolSteps.map((toolStep) => [toolStep.toolCallID, toolStep.state]), [
    ["call-1", ToolStepState.SUCCEEDED],
    ["call-2", ToolStepState.RUNNING],
  ]);
});

test("should_keep_a_turn_that_only_called_tools_when_removing_empty_turns", () => {
  const log = new ConversationLog();
  log.appendUserTurn("What time is it?");
  log.openAssistantTurn();
  log.applyToolStep(new ToolStep({ toolCallID: "call-1", toolName: "get_current_time", label: "Current time", argumentsJSON: "{}", resultText: "", state: ToolStepState.RUNNING }));
  log.removeEmptyAssistantTurn();
  assert.equal(log.messages.length, 2);
});

test("should_answer_queued_messages_in_order_with_each_answer_under_its_message", () => {
  const log = new ConversationLog();
  log.appendUserTurn("First");
  log.openAssistantTurn();
  log.applyTextDelta("One.");
  const secondQueued = log.appendQueuedUserTurn("Second");
  const thirdQueued = log.appendQueuedUserTurn("Third");
  assert.deepEqual(log.requestMessages().map((message) => message.content), ["First"], "queued messages wait");
  assert.equal(log.beginAnsweringQueuedTurn(secondQueued.id), true);
  log.applyTextDelta("Two.");
  assert.deepEqual(log.messages.map((message) => [message.content, message.state]), [
    ["First", ChatMessageState.COMPLETE],
    ["One.", ChatMessageState.COMPLETE],
    ["Second", ChatMessageState.COMPLETE],
    ["Two.", ChatMessageState.COMPLETE],
    ["Third", ChatMessageState.QUEUED],
  ]);
  assert.equal(log.firstQueuedUserTurn()?.id, thirdQueued.id);
});

test("should_refuse_to_answer_a_message_that_is_not_queued", () => {
  const log = new ConversationLog();
  const sentMessage = log.appendUserTurn("Sent");
  assert.equal(log.beginAnsweringQueuedTurn(sentMessage.id), false);
  assert.equal(log.beginAnsweringQueuedTurn("unknown"), false);
});

test("should_mark_unanswered_queued_messages_not_sent", () => {
  const log = new ConversationLog();
  log.appendUserTurn("First");
  log.appendQueuedUserTurn("Never mind");
  log.markQueuedTurnsNotSent();
  assert.equal(log.messages[1]?.state, ChatMessageState.FAILED);
  assert.equal(log.firstQueuedUserTurn(), null);
  assert.deepEqual(log.requestMessages().map((message) => message.content), ["First"]);
});

test("should_keep_the_log_in_answer_order_when_a_later_message_is_answered_first", () => {
  const log = new ConversationLog();
  log.appendUserTurn("U1");
  log.openAssistantTurn();
  log.applyTextDelta("A1");
  const thirdQueued = log.appendQueuedUserTurn("Q3");
  const fourthQueued = log.appendQueuedUserTurn("Q4");
  log.beginAnsweringQueuedTurn(fourthQueued.id);
  log.applyTextDelta("A4");
  log.markAssistantTurn(ChatMessageState.COMPLETE);
  log.assistantMessageID = null;
  assert.deepEqual(log.messages.map((message) => message.content), ["U1", "A1", "Q4", "A4", "Q3"]);
  log.beginAnsweringQueuedTurn(thirdQueued.id);
  assert.deepEqual(log.requestMessages().map((message) => message.content), ["U1", "A1", "Q4", "A4", "Q3"], "the request ends with the question");
});

test("should_place_a_retried_answer_under_its_question_ahead_of_unsent_messages", () => {
  const log = new ConversationLog();
  log.appendUserTurn("U1");
  log.openAssistantTurn();
  log.markAssistantTurn(ChatMessageState.FAILED);
  log.assistantMessageID = null;
  log.appendQueuedUserTurn("Q2");
  log.markQueuedTurnsNotSent();
  assert.equal(log.lastUserTurn()?.content, "U1", "a message never sent is not retried");
  log.openAssistantTurn();
  log.applyTextDelta("Retried answer");
  assert.deepEqual(log.messages.map((message) => message.content), ["U1", "", "Retried answer", "Q2"]);
});

test("should_find_no_question_to_retry_when_every_message_is_unsent", () => {
  const log = new ConversationLog();
  log.appendQueuedUserTurn("Q1");
  log.markQueuedTurnsNotSent();
  assert.equal(log.lastUserTurn(), null);
});
