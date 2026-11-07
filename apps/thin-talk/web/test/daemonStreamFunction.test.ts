import assert from "node:assert/strict";
import { test } from "node:test";
import type { AssistantMessage, Message, TranscriptContext } from "@earendil-works/pi-ai";
import { AppConfig } from "../src/AppConfig";
import { ChatFailureKind } from "../src/ChatFailureKind";
import { DaemonClient } from "../src/DaemonClient";
import { daemonModel, emptyAssistantMessage } from "../src/DaemonModelDescription";
import { DaemonStreamFunction } from "../src/DaemonStreamFunction";
import { DAEMON_BASE_URL, ScriptedChatFetch, chunkLine } from "./support/scriptedChatFetch";

const DAEMON_MODEL = daemonModel("fake-chat-model");

function daemonClient(): DaemonClient {
  return new DaemonClient(
    new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" }),
  );
}

function transcript(messages: Message[]): TranscriptContext {
  return { messages } as unknown as TranscriptContext;
}

function assistantReply(content: AssistantMessage["content"]): AssistantMessage {
  const reply = emptyAssistantMessage(DAEMON_MODEL);
  reply.content = content;
  return reply;
}

test("should_send_only_the_fields_and_header_the_daemon_reads", async () => {
  const scriptedFetch = new ScriptedChatFetch([{ lines: [chunkLine({ content: "Hi." }, "stop"), "data: [DONE]"] }]);
  scriptedFetch.install();
  const streamFunction = new DaemonStreamFunction(daemonClient(), 512, new AbortController().signal);
  const finalMessage = await streamFunction
    .stream(DAEMON_MODEL, transcript([{ role: "user", content: "Hello", timestamp: 0 }]))
    .result();
  assert.equal(finalMessage.stopReason, "stop");
  const sentRequest = scriptedFetch.requests[0];
  assert.ok(sentRequest);
  assert.deepEqual(sentRequest.body, {
    model: "fake-chat-model",
    stream: true,
    messages: [{ role: "user", content: "Hello" }],
    thinking_budget: 512,
  });
  assert.deepEqual(sentRequest.headerNames, ["content-type"]);
});

test("should_send_history_as_plain_text_without_earlier_reasoning", () => {
  const requestMessages = DaemonStreamFunction.requestMessages([
    { role: "system", content: "", timestamp: 0 },
    { role: "system", content: [{ type: "text", text: "Be brief." }], timestamp: 0 },
    { role: "user", content: [{ type: "text", text: "Who wrote " }, { type: "text", text: "Hamlet?" }], timestamp: 0 },
    assistantReply([{ type: "thinking", thinking: "Easy." }, { type: "text", text: "Shakespeare." }]),
    { role: "user", content: "And Macbeth?", timestamp: 0 },
  ] as Message[]);
  assert.deepEqual(requestMessages, [
    { role: "system", content: "Be brief." },
    { role: "user", content: "Who wrote Hamlet?" },
    { role: "assistant", content: "Shakespeare." },
    { role: "user", content: "And Macbeth?" },
  ]);
});

test("should_refuse_a_tool_result_while_no_tools_are_offered", () => {
  assert.throws(() =>
    DaemonStreamFunction.requestMessages([
      { role: "toolResult", toolCallId: "call-1", toolName: "clock", content: [], isError: false, timestamp: 0 },
    ] as unknown as Message[]),
  );
});

test("should_keep_the_classified_failure_behind_an_http_error", async () => {
  const scriptedFetch = new ScriptedChatFetch([{ status: 503, errorBody: '{"error":{"message":"not ready"}}' }]);
  scriptedFetch.install();
  const streamFunction = new DaemonStreamFunction(daemonClient(), 256, new AbortController().signal);
  const finalMessage = await streamFunction
    .stream(DAEMON_MODEL, transcript([{ role: "user", content: "Hello", timestamp: 0 }]))
    .result();
  assert.equal(finalMessage.stopReason, "error");
  assert.equal(streamFunction.lastFailure?.kind, ChatFailureKind.WORKER_UNAVAILABLE);
});

test("should_end_as_aborted_when_the_turn_is_stopped_mid_stream", async () => {
  const scriptedFetch = new ScriptedChatFetch([{ lines: [chunkLine({ content: "Once upon" })], holdOpenUntilAborted: true }]);
  scriptedFetch.install();
  const streamFunction = new DaemonStreamFunction(daemonClient(), 256, new AbortController().signal);
  const stopController = new AbortController();
  const eventStream = streamFunction.stream(
    DAEMON_MODEL,
    transcript([{ role: "user", content: "Tell a story", timestamp: 0 }]),
    { signal: stopController.signal },
  );
  for await (const streamEvent of eventStream) {
    if (streamEvent.type === "text_delta") {
      stopController.abort();
    }
  }
  const finalMessage = await eventStream.result();
  assert.equal(finalMessage.stopReason, "aborted");
  assert.deepEqual(finalMessage.content, [{ type: "text", text: "Once upon" }]);
  assert.equal(streamFunction.lastFailure, null);
});

test("should_report_an_unexpected_error_as_an_unknown_failure", async () => {
  const scriptedFetch = new ScriptedChatFetch([]);
  scriptedFetch.install();
  const streamFunction = new DaemonStreamFunction(daemonClient(), 256, new AbortController().signal);
  const finalMessage = await streamFunction
    .stream(DAEMON_MODEL, transcript([
      { role: "toolResult", toolCallId: "call-1", toolName: "clock", content: [], isError: false, timestamp: 0 },
    ] as unknown as Message[]))
    .result();
  assert.equal(finalMessage.stopReason, "error");
  assert.equal(streamFunction.lastFailure?.kind, ChatFailureKind.UNKNOWN);
  assert.equal(scriptedFetch.requests.length, 0);
});

test("should_cancel_the_request_when_its_turn_ends_first", async () => {
  const scriptedFetch = new ScriptedChatFetch([{ lines: [chunkLine({ content: "Still going" })], holdOpenUntilAborted: true }]);
  scriptedFetch.install();
  const turnLifetime = new AbortController();
  const streamFunction = new DaemonStreamFunction(daemonClient(), 256, turnLifetime.signal);
  const eventStream = streamFunction.stream(DAEMON_MODEL, transcript([{ role: "user", content: "Hello", timestamp: 0 }]));
  for await (const streamEvent of eventStream) {
    if (streamEvent.type === "text_delta") {
      turnLifetime.abort();
    }
  }
  assert.equal((await eventStream.result()).stopReason, "aborted");
  assert.equal(scriptedFetch.requests[0]?.wasCancelled, true);
});
