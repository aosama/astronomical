import assert from "node:assert/strict";
import { test } from "node:test";
import { AppConfig } from "../src/AppConfig";
import { ChatCompletionChunk } from "../src/ChatCompletionChunk";
import type { ChatCompletionRequestBody } from "../src/ChatCompletionRequestBody";
import { ChatFailure } from "../src/ChatFailure";
import { ChatFailureKind } from "../src/ChatFailureKind";
import { DaemonClient, LibraryDownloadRefusal } from "../src/DaemonClient";
import { DAEMON_BASE_URL, ScriptedChatFetch, chunkLine } from "./support/scriptedChatFetch";

const QUESTION_BODY: ChatCompletionRequestBody = {
  model: "fake-chat-model",
  stream: true,
  messages: [{ role: "user", content: "Who wrote The Tempest?" }],
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

/** Answers every request with one scripted response, recording what was sent. */
function answerEveryRequest(answer: () => Response | Promise<Response>): { method: string; url: string; body: string | null }[] {
  const sentRequests: { method: string; url: string; body: string | null }[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    sentRequests.push({ method: init?.method ?? "GET", url: String(input), body: typeof init?.body === "string" ? init.body : null });
    return answer();
  }) as typeof fetch;
  return sentRequests;
}

test("should_start_a_library_download_and_read_its_starting_status", async () => {
  const sentRequests = answerEveryRequest(() => Response.json({ state: "checking_disk", huggingface_id: "example-org/chat-model", bytes_completed: 0, bytes_total: 0, error_code: null }, { status: 202 }));
  const startingStatus = await daemonClient().startLibraryDownload("example-org/chat-model");
  assert.equal(startingStatus.state, "checking_disk");
  assert.deepEqual(sentRequests, [{ method: "POST", url: `${DAEMON_BASE_URL}/v1/library/download`, body: JSON.stringify({ huggingface_id: "example-org/chat-model" }) }]);
});

test("should_carry_the_daemons_refusal_code_or_none_when_a_download_cannot_start", async () => {
  answerEveryRequest(() => Response.json({ error: { code: "library_busy", message: "busy" } }, { status: 409 }));
  await assert.rejects(daemonClient().startLibraryDownload("example-org/chat-model"), (refusal: LibraryDownloadRefusal) => refusal.refusalCode === "library_busy");
  answerEveryRequest(() => new Response("<html>gateway</html>", { status: 502 }));
  await assert.rejects(daemonClient().startLibraryDownload("example-org/chat-model"), (refusal: LibraryDownloadRefusal) => refusal.refusalCode === null);
  answerEveryRequest(() => Promise.reject(new TypeError("connection refused")));
  await assert.rejects(daemonClient().startLibraryDownload("example-org/chat-model"), (refusal: LibraryDownloadRefusal) => refusal.refusalCode === null);
});

test("should_resume_the_paused_download_and_report_a_refusal", async () => {
  const sentRequests = answerEveryRequest(() => Response.json({ state: "resuming" }, { status: 202 }));
  await daemonClient().resumeLibraryDownload();
  assert.deepEqual(sentRequests.map((request) => [request.method, request.url]), [["POST", `${DAEMON_BASE_URL}/v1/library/download/resume`]]);
  answerEveryRequest(() => Response.json({ error: { code: "download_failed", message: "There is no model download to control." } }, { status: 404 }));
  await assert.rejects(daemonClient().resumeLibraryDownload(), (refusal: LibraryDownloadRefusal) => refusal.refusalCode === "download_failed");
});

test("should_read_the_current_library_download", async () => {
  answerEveryRequest(() => Response.json({ state: "downloading", huggingface_id: "example-org/chat-model", bytes_completed: 5, bytes_total: 10, error_code: null }));
  const currentStatus = await daemonClient().fetchLibraryDownloadStatus();
  assert.deepEqual([currentStatus.state, currentStatus.bytesCompleted, currentStatus.bytesTotal], ["downloading", 5, 10]);
});

test("should_report_a_model_load_only_while_the_status_names_one", async () => {
  const statusBodies = ['{"activity":"idle","model_load_in_flight":{"model_id":"Example-9B"}}', '{"activity":"idle","model_load_in_flight":null}', '{"activity":"idle"}', "[]"];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response(statusBodies.shift() ?? "{}", { status: 200 })) as typeof fetch;
  try {
    const client = daemonClient();
    assert.equal(await client.isModelLoading(), true);
    assert.equal(await client.isModelLoading(), false);
    assert.equal(await client.isModelLoading(), false, "an engine that does not report loads");
    assert.equal(await client.isModelLoading(), false, "a status that is not an object");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
