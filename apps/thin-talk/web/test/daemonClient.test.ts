import assert from "node:assert/strict";
import { test } from "node:test";
import { AppConfig } from "../src/AppConfig";
import { ChatCompletionChunk } from "../src/ChatCompletionChunk";
import type { ChatCompletionRequestBody } from "../src/ChatCompletionRequestBody";
import { ChatFailure } from "../src/ChatFailure";
import { ChatFailureKind } from "../src/ChatFailureKind";
import { DaemonClient } from "../src/DaemonClient";
import { DAEMON_BASE_URL, ScriptedChatFetch, chunkLine } from "./support/scriptedChatFetch";

const QUESTION_BODY: ChatCompletionRequestBody = {
  model: "fake-chat-model",
  stream: true,
  messages: [{ role: "user", content: "Who wrote The Tempest?" }],
  thinking_budget: 256,
};

function daemonClient(): DaemonClient {
  return new DaemonClient(
    new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" }),
  );
}

async function collectChunks(client: DaemonClient): Promise<ChatCompletionChunk[]> {
  const deliveredChunks: ChatCompletionChunk[] = [];
  await client.streamChatCompletion(QUESTION_BODY, new AbortController().signal, (chunk) => deliveredChunks.push(chunk));
  return deliveredChunks;
}

test("should_reassemble_a_line_split_across_reads", async () => {
  new ScriptedChatFetch([{
    lines: [chunkLine({ content: "Shakespeare wrote it." }), chunkLine({}, "stop"), "data: [DONE]"],
    bytesPerRead: 7,
  }]).install();
  const deliveredChunks = await collectChunks(daemonClient());
  assert.deepEqual(deliveredChunks.map((chunk) => chunk.textDelta).join(""), "Shakespeare wrote it.");
  assert.equal(deliveredChunks[deliveredChunks.length - 1]?.finishReason, "stop");
});

test("should_deliver_a_final_line_that_has_no_newline", async () => {
  new ScriptedChatFetch([{ lines: [chunkLine({ content: "Last words." }, "stop")], omitFinalNewline: true }]).install();
  const deliveredChunks = await collectChunks(daemonClient());
  assert.equal(deliveredChunks.length, 1);
  assert.equal(deliveredChunks[0]?.finishReason, "stop");
});

test("should_deliver_text_before_a_streamed_error_and_then_fail", async () => {
  const scriptedFetch = new ScriptedChatFetch([{
    lines: [chunkLine({ content: "Half" }), 'data: {"error":{"message":"chat_worker_unavailable: worker exited"}}'],
    holdOpenUntilAborted: true,
  }]);
  scriptedFetch.install();
  const deliveredChunks: ChatCompletionChunk[] = [];
  await assert.rejects(
    daemonClient().streamChatCompletion(QUESTION_BODY, new AbortController().signal, (chunk) => deliveredChunks.push(chunk)),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.WORKER_UNAVAILABLE,
  );
  assert.equal(deliveredChunks[0]?.textDelta, "Half");
  assert.equal(scriptedFetch.requests[0]?.wasCancelled, true, "the unread rest of the stream is cancelled");
});

test("should_classify_an_http_refusal_by_its_reason", async () => {
  new ScriptedChatFetch([{ status: 429, errorBody: '{"error":{"message":"server_capacity: busy"}}' }]).install();
  await assert.rejects(
    collectChunks(daemonClient()),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.ENGINE_BUSY && error.reason === "server_capacity: busy",
  );
});

test("should_return_quietly_when_stopped_before_the_reply_arrives", async () => {
  new ScriptedChatFetch([{ networkFailure: true }]).install();
  const stopController = new AbortController();
  stopController.abort();
  await daemonClient().streamChatCompletion(QUESTION_BODY, stopController.signal, () => assert.fail("no chunk expected"));
});
