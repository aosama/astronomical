import assert from "node:assert/strict";
import { test } from "node:test";
import { createInitialSystemMessage, type AssistantMessage, type Message, type TranscriptContext } from "@earendil-works/pi-ai";
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
  const streamFunction = new DaemonStreamFunction(daemonClient(), new AbortController().signal);
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
  });
  assert.equal("thinking_budget" in sentRequest.body, false, "the model thinks as long as it needs");
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

test("should_send_tool_calls_and_their_results_in_the_daemon_shape", () => {
  const requestMessages = DaemonStreamFunction.requestMessages([
    { role: "user", content: "What time is it?", timestamp: 0 },
    assistantReply([{ type: "toolCall", id: "call-1", name: "get_current_time", arguments: {} }]),
    { role: "toolResult", toolCallId: "call-1", toolName: "get_current_time", content: [{ type: "text", text: "Saturday 2 PM" }], isError: false, timestamp: 0 },
    assistantReply([{ type: "text", text: "It is 2 PM." }]),
  ] as Message[]);
  assert.deepEqual(requestMessages, [
    { role: "user", content: "What time is it?" },
    { role: "assistant", content: "", tool_calls: [{ id: "call-1", type: "function", function: { name: "get_current_time", arguments: "{}" } }] },
    { role: "tool", tool_call_id: "call-1", content: "Saturday 2 PM" },
    { role: "assistant", content: "It is 2 PM." },
  ]);
});

test("should_keep_the_classified_failure_behind_an_http_error", async () => {
  const scriptedFetch = new ScriptedChatFetch([{ status: 503, errorBody: '{"error":{"message":"not ready"}}' }]);
  scriptedFetch.install();
  const streamFunction = new DaemonStreamFunction(daemonClient(), new AbortController().signal);
  const finalMessage = await streamFunction
    .stream(DAEMON_MODEL, transcript([{ role: "user", content: "Hello", timestamp: 0 }]))
    .result();
  assert.equal(finalMessage.stopReason, "error");
  assert.equal(streamFunction.lastFailure?.kind, ChatFailureKind.WORKER_UNAVAILABLE);
});

test("should_end_as_aborted_when_the_turn_is_stopped_mid_stream", async () => {
  const scriptedFetch = new ScriptedChatFetch([{ lines: [chunkLine({ content: "Once upon" })], holdOpenUntilAborted: true }]);
  scriptedFetch.install();
  const streamFunction = new DaemonStreamFunction(daemonClient(), new AbortController().signal);
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
  const streamFunction = new DaemonStreamFunction(daemonClient(), new AbortController().signal);
  const finalMessage = await streamFunction
    .stream(DAEMON_MODEL, transcript([
      // A BigInt cannot be written as JSON, so building the request throws a
      // plain TypeError rather than a classified failure.
      assistantReply([{ type: "toolCall", id: "call-1", name: "clock", arguments: { tick: BigInt(1) } as never }]),
    ] as Message[]))
    .result();
  assert.equal(finalMessage.stopReason, "error");
  assert.equal(streamFunction.lastFailure?.kind, ChatFailureKind.UNKNOWN);
  assert.equal(scriptedFetch.requests.length, 0);
});

test("should_cancel_the_request_when_its_turn_ends_first", async () => {
  const scriptedFetch = new ScriptedChatFetch([{ lines: [chunkLine({ content: "Still going" })], holdOpenUntilAborted: true }]);
  scriptedFetch.install();
  const turnLifetime = new AbortController();
  const streamFunction = new DaemonStreamFunction(daemonClient(), turnLifetime.signal);
  const eventStream = streamFunction.stream(DAEMON_MODEL, transcript([{ role: "user", content: "Hello", timestamp: 0 }]));
  for await (const streamEvent of eventStream) {
    if (streamEvent.type === "text_delta") {
      turnLifetime.abort();
    }
  }
  assert.equal((await eventStream.result()).stopReason, "aborted");
  assert.equal(scriptedFetch.requests[0]?.wasCancelled, true);
});

test("should_declare_the_transcripts_tools_and_omit_tools_when_there_are_none", async () => {
  const scriptedFetch = new ScriptedChatFetch([
    { lines: [chunkLine({ content: "ok" }, "stop")] },
    { lines: [chunkLine({ content: "ok" }, "stop")] },
  ]);
  scriptedFetch.install();
  const streamFunction = new DaemonStreamFunction(daemonClient(), new AbortController().signal);
  const clockTool = { name: "get_current_time", description: "Reads the clock.", parameters: { type: "object", properties: {} } };
  await streamFunction
    .stream(DAEMON_MODEL, transcript([
      createInitialSystemMessage(undefined, [clockTool as never]),
      { role: "user", content: "What time is it?", timestamp: 0 },
    ] as unknown as Message[]))
    .result();
  await streamFunction.stream(DAEMON_MODEL, transcript([{ role: "user", content: "Hi", timestamp: 0 }])).result();
  assert.deepEqual(scriptedFetch.requests[0]?.body["tools"], [
    { type: "function", function: { name: "get_current_time", description: "Reads the clock.", parameters: { type: "object", properties: {} } } },
  ]);
  assert.equal("tools" in (scriptedFetch.requests[1]?.body ?? {}), false);
});

test("should_follow_a_tool_result_with_its_image_as_a_user_message", () => {
  const requestMessages = DaemonStreamFunction.requestMessages([
    { role: "user", content: "Draw a rose.", timestamp: 0 },
    assistantReply([{ type: "toolCall", id: "call-1", name: "generate_image", arguments: { prompt: "A rose" } }]),
    {
      role: "toolResult", toolCallId: "call-1", toolName: "generate_image", isError: false, timestamp: 0,
      content: [{ type: "text", text: "Generated." }, { type: "image", data: "iVBORw0KGgo=", mimeType: "image/png" }],
    },
  ] as Message[]);
  assert.deepEqual(requestMessages.slice(2), [
    { role: "tool", tool_call_id: "call-1", content: "Generated." },
    {
      role: "user",
      content: [
        { type: "text", text: "The images the tools returned:" },
        { type: "image_url", image_url: { url: "data:image/png;base64,iVBORw0KGgo=" } },
      ],
    },
  ]);
});

test("should_send_a_user_message_with_an_image_as_parts_and_one_without_as_text", () => {
  const requestMessages = DaemonStreamFunction.requestMessages([
    { role: "user", content: [{ type: "text", text: "What is this?" }, { type: "image", data: "/9j/4AAQ", mimeType: "image/jpeg" }], timestamp: 0 },
    { role: "user", content: [{ type: "image", data: "/9j/4AAQ", mimeType: "image/jpeg" }], timestamp: 0 },
    { role: "user", content: "Plain words.", timestamp: 0 },
  ] as Message[]);
  assert.deepEqual(requestMessages, [
    { role: "user", content: [{ type: "text", text: "What is this?" }, { type: "image_url", image_url: { url: "data:image/jpeg;base64,/9j/4AAQ" } }] },
    { role: "user", content: [{ type: "image_url", image_url: { url: "data:image/jpeg;base64,/9j/4AAQ" } }] },
    { role: "user", content: "Plain words." },
  ]);
});

test("should_keep_every_tool_result_together_and_send_their_images_after_them", () => {
  const imageResult = (callID: string, data: string) => ({
    role: "toolResult", toolCallId: callID, toolName: "generate_image", isError: false, timestamp: 0,
    content: [{ type: "text", text: `Drew ${callID}.` }, { type: "image", data, mimeType: "image/png" }],
  });
  const requestMessages = DaemonStreamFunction.requestMessages([
    { role: "user", content: "Draw Romeo and Juliet.", timestamp: 0 },
    assistantReply([
      { type: "toolCall", id: "romeo", name: "generate_image", arguments: { prompt: "Romeo" } },
      { type: "toolCall", id: "juliet", name: "generate_image", arguments: { prompt: "Juliet" } },
    ]),
    imageResult("romeo", "ROMEO"),
    imageResult("juliet", "JULIET"),
    assistantReply([{ type: "text", text: "Both drawn." }]),
  ] as Message[]);
  assert.deepEqual(requestMessages.map((requestMessage) => requestMessage.role), ["user", "assistant", "tool", "tool", "user", "assistant"]);
  assert.deepEqual(requestMessages[4], {
    role: "user",
    content: [
      { type: "text", text: "The images the tools returned:" },
      { type: "image_url", image_url: { url: "data:image/png;base64,ROMEO" } },
      { type: "image_url", image_url: { url: "data:image/png;base64,JULIET" } },
    ],
  });
});
