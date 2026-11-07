import assert from "node:assert/strict";
import { test } from "node:test";
import { AgentTurnRunner } from "../src/AgentTurnRunner";
import { AppConfig } from "../src/AppConfig";
import { ChatFailure } from "../src/ChatFailure";
import { ChatFailureKind } from "../src/ChatFailureKind";
import { ChatMessage } from "../src/ChatMessage";
import { ChatRole } from "../src/ChatRole";
import type { ChatStreamRequest } from "../src/ChatStreamer";
import { DaemonClient } from "../src/DaemonClient";
import { ThinkingEffort } from "../src/ThinkingEffort";
import { DAEMON_BASE_URL, ScriptedChatFetch, chunkLine, waitUntil } from "./support/scriptedChatFetch";

interface ForwardedDeltas {
  text: string;
  reasoning: string;
}

function turnRunner(): AgentTurnRunner {
  return new AgentTurnRunner(
    new DaemonClient(
      new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" }),
    ),
  );
}

function turnRequest(messages: ChatMessage[], forwardedDeltas: ForwardedDeltas, signal: AbortSignal): ChatStreamRequest {
  return {
    modelID: "fake-chat-model",
    messages,
    thinkingEffort: ThinkingEffort.BALANCED,
    onText: (delta) => { forwardedDeltas.text += delta; },
    onReasoning: (delta) => { forwardedDeltas.reasoning += delta; },
    signal,
  };
}

function userMessage(content: string): ChatMessage {
  return new ChatMessage({ id: `user-${content}`, role: ChatRole.USER, content });
}

test("should_stream_reasoning_and_text_of_one_turn_to_the_conversation", async () => {
  new ScriptedChatFetch([{
    lines: [chunkLine({ reasoning_content: "Thinking." }), chunkLine({ content: "Shakespeare." }), chunkLine({}, "stop")],
  }]).install();
  const forwardedDeltas: ForwardedDeltas = { text: "", reasoning: "" };
  await turnRunner().streamChat(turnRequest([userMessage("Who wrote Hamlet?")], forwardedDeltas, new AbortController().signal));
  assert.deepEqual(forwardedDeltas, { text: "Shakespeare.", reasoning: "Thinking." });
});

test("should_replay_earlier_turns_and_ask_the_latest_question_once", async () => {
  const scriptedFetch = new ScriptedChatFetch([{ lines: [chunkLine({ content: "1606." }, "stop")] }]);
  scriptedFetch.install();
  const history = [
    userMessage("Who wrote Macbeth?"),
    new ChatMessage({ id: "answer-1", role: ChatRole.ASSISTANT, content: "Shakespeare.", reasoning: "Known." }),
    userMessage("When?"),
  ];
  await turnRunner().streamChat(turnRequest(history, { text: "", reasoning: "" }, new AbortController().signal));
  assert.equal(scriptedFetch.requests.length, 1);
  assert.deepEqual(scriptedFetch.requests[0]?.body["messages"], [
    { role: "user", content: "Who wrote Macbeth?" },
    { role: "assistant", content: "Shakespeare." },
    { role: "user", content: "When?" },
  ]);
  assert.equal(scriptedFetch.requests[0]?.body["thinking_budget"], ThinkingEffort.thinkingBudgetTokens(ThinkingEffort.BALANCED));
});

test("should_refuse_a_turn_whose_history_does_not_end_with_a_question", async () => {
  const scriptedFetch = new ScriptedChatFetch([]);
  scriptedFetch.install();
  const endsWithAnswer = [userMessage("Hi"), new ChatMessage({ id: "answer", role: ChatRole.ASSISTANT, content: "Hello." })];
  for (const history of [[], endsWithAnswer]) {
    await assert.rejects(
      turnRunner().streamChat(turnRequest(history, { text: "", reasoning: "" }, new AbortController().signal)),
      (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.INVALID_REQUEST,
    );
  }
  assert.equal(scriptedFetch.requests.length, 0);
});

test("should_end_quietly_when_the_user_stops_mid_answer", async () => {
  new ScriptedChatFetch([{ lines: [chunkLine({ content: "Once upon a time" })], holdOpenUntilAborted: true }]).install();
  const forwardedDeltas: ForwardedDeltas = { text: "", reasoning: "" };
  const stopController = new AbortController();
  const turnFinished = turnRunner().streamChat(turnRequest([userMessage("Tell a story")], forwardedDeltas, stopController.signal));
  await waitUntil(() => forwardedDeltas.text.length > 0, "the first words of the answer");
  stopController.abort();
  await turnFinished;
  assert.equal(forwardedDeltas.text, "Once upon a time");
});

test("should_send_nothing_when_stopped_before_the_turn_starts", async () => {
  const scriptedFetch = new ScriptedChatFetch([]);
  scriptedFetch.install();
  const stopController = new AbortController();
  stopController.abort();
  await turnRunner().streamChat(turnRequest([userMessage("Hi")], { text: "", reasoning: "" }, stopController.signal));
  assert.equal(scriptedFetch.requests.length, 0);
});

test("should_rethrow_the_daemons_classified_failure", async () => {
  new ScriptedChatFetch([{ lines: ['data: {"error":{"message":"chat_engine_busy: another request is running"}}'] }]).install();
  await assert.rejects(
    turnRunner().streamChat(turnRequest([userMessage("Hi")], { text: "", reasoning: "" }, new AbortController().signal)),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.ENGINE_BUSY,
  );
});

test("should_report_an_unreachable_daemon_as_unavailable", async () => {
  new ScriptedChatFetch([{ networkFailure: true }]).install();
  await assert.rejects(
    turnRunner().streamChat(turnRequest([userMessage("Hi")], { text: "", reasoning: "" }, new AbortController().signal)),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.WORKER_UNAVAILABLE,
  );
});

test("should_fail_a_turn_whose_stream_was_cut_off", async () => {
  new ScriptedChatFetch([{ lines: [chunkLine({ content: "Half" })] }]).install();
  await assert.rejects(
    turnRunner().streamChat(turnRequest([userMessage("Hi")], { text: "", reasoning: "" }, new AbortController().signal)),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.MALFORMED_MODEL_OUTPUT,
  );
});

test("should_finish_a_turn_cut_at_the_length_limit_without_failing", async () => {
  new ScriptedChatFetch([{ lines: [chunkLine({ content: "A long answer that" }, "length")] }]).install();
  const forwardedDeltas: ForwardedDeltas = { text: "", reasoning: "" };
  await turnRunner().streamChat(turnRequest([userMessage("Explain")], forwardedDeltas, new AbortController().signal));
  assert.equal(forwardedDeltas.text, "A long answer that");
});

test("should_cancel_the_daemon_request_when_the_conversation_listener_throws", async () => {
  const scriptedFetch = new ScriptedChatFetch([{ lines: [chunkLine({ content: "Hello" })], holdOpenUntilAborted: true }]);
  scriptedFetch.install();
  const throwingRequest: ChatStreamRequest = {
    ...turnRequest([userMessage("Hi")], { text: "", reasoning: "" }, new AbortController().signal),
    onText: () => {
      throw new Error("the transcript failed to draw");
    },
  };
  await assert.rejects(
    turnRunner().streamChat(throwingRequest),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.UNKNOWN,
  );
  await waitUntil(() => scriptedFetch.requests[0]?.wasCancelled === true, "the daemon request to be cancelled");
});
