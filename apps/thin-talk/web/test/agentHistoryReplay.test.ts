import assert from "node:assert/strict";
import { test } from "node:test";
import { agentHistory } from "../src/AgentHistoryReplay";
import { ChatMessage } from "../src/ChatMessage";
import { ChatRole } from "../src/ChatRole";
import { daemonModel } from "../src/DaemonModelDescription";
import { ToolStep, ToolStepState } from "../src/ToolStep";

const MODEL = daemonModel("fake-chat-model");

function answerWithTimeStep(state: ToolStepState, answerText: string): ChatMessage {
  return new ChatMessage({
    id: "answer-1",
    role: ChatRole.ASSISTANT,
    content: answerText,
    toolSteps: [new ToolStep({
      toolCallID: "call-1",
      toolName: "get_current_time",
      label: "Current time",
      argumentsJSON: "{}",
      resultText: "Saturday, 2 PM",
      state,
    })],
  });
}

const QUESTION = new ChatMessage({ id: "question-1", role: ChatRole.USER, content: "What time is it?" });

test("should_replay_a_tool_answer_as_call_then_result_then_text", () => {
  const history = agentHistory([QUESTION, answerWithTimeStep(ToolStepState.SUCCEEDED, "It is 2 PM.")], MODEL, true);
  assert.deepEqual(history.map((agentMessage) => agentMessage.role), ["user", "assistant", "toolResult", "assistant"]);
  const toolCallMessage = history[1] as { content: { type: string; id?: string }[] };
  assert.equal(toolCallMessage.content[0]?.type, "toolCall");
  assert.equal(toolCallMessage.content[0]?.id, "call-1");
  const toolResult = history[2] as { isError: boolean; content: { text: string }[] };
  assert.equal(toolResult.isError, false);
  assert.equal(toolResult.content[0]?.text, "Saturday, 2 PM");
});

test("should_leave_tool_steps_out_when_the_turn_offers_no_tools", () => {
  const history = agentHistory([QUESTION, answerWithTimeStep(ToolStepState.SUCCEEDED, "It is 2 PM.")], MODEL, false);
  assert.deepEqual(history.map((agentMessage) => agentMessage.role), ["user", "assistant"]);
});

test("should_replay_an_unfinished_step_as_a_failed_call", () => {
  const history = agentHistory([QUESTION, answerWithTimeStep(ToolStepState.RUNNING, "")], MODEL, true);
  assert.deepEqual(history.map((agentMessage) => agentMessage.role), ["user", "assistant", "toolResult"]);
  const toolResult = history[2] as { isError: boolean; content: { text: string }[] };
  assert.equal(toolResult.isError, true);
  assert.match(toolResult.content[0]?.text ?? "", /stopped before it finished/);
});

test("should_replay_damaged_saved_arguments_as_none", () => {
  const damagedStep = answerWithTimeStep(ToolStepState.SUCCEEDED, "ok");
  damagedStep.toolSteps = [new ToolStep({ ...damagedStep.toolSteps[0]!, argumentsJSON: "{not json" })];
  const history = agentHistory([QUESTION, damagedStep], MODEL, true);
  const toolCallMessage = history[1] as { content: { arguments: unknown }[] };
  assert.deepEqual(toolCallMessage.content[0]?.arguments, {});
});
