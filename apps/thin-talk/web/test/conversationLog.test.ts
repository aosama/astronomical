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
