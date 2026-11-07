import assert from "node:assert/strict";
import { test } from "node:test";
import { ReadChosenFileTool } from "../src/ReadChosenFileTool";
import type { SessionBridgeRequest } from "../src/SessionBridgeRequest";
import { scriptedBridge } from "./support/scriptedBridge";

const TURN_CONTEXT = { conversationID: "conversation-1", modelID: "chat-model" };

function readTool(hostAnswer: unknown, sentRequests: SessionBridgeRequest[] = []): ReadChosenFileTool {
  return new ReadChosenFileTool(scriptedBridge({ useHostTool: hostAnswer, cancelHostToolCall: null }, sentRequests));
}

test("should_return_the_chosen_files_contents_in_one_host_call", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const toolResult = await readTool({ outcome: "chosen", fileName: "sonnet.txt", text: "Shall I compare thee" }, sentRequests)
    .execute({}, new AbortController().signal, TURN_CONTEXT);
  assert.equal(toolResult, "Contents of sonnet.txt:\n\nShall I compare thee");
  assert.deepEqual(sentRequests.map((request) => [request.op, request.payload]), [
    ["useHostTool", { toolName: "read_file", conversationID: "conversation-1" }],
  ]);
});

test("should_tell_a_refusal_apart_from_a_tool_the_user_never_saw", async () => {
  await assert.rejects(readTool({ outcome: "refused" }).execute({}, new AbortController().signal, TURN_CONTEXT), /did not allow "Read a file"/);
  await assert.rejects(
    readTool({ outcome: "unavailable", reason: "the conversation window is closed" }).execute({}, new AbortController().signal, TURN_CONTEXT),
    /not available right now \(the conversation window is closed\)\. The user was not asked/,
  );
  await assert.rejects(readTool({ outcome: "nothingChosen" }).execute({}, new AbortController().signal, TURN_CONTEXT), /did not choose anything/);
});

test("should_not_call_the_host_without_an_open_conversation", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  await assert.rejects(
    readTool({ outcome: "approved" }, sentRequests).execute({}, new AbortController().signal, { conversationID: null, modelID: "m" }),
    /no conversation is open/,
  );
  assert.equal(sentRequests.length, 0);
});

test("should_cancel_the_host_call_when_the_turn_stops", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const neverAnsweringBridge = scriptedBridge({ cancelHostToolCall: null }, sentRequests);
  // The scripted host has no answer for useHostTool and rejects it; replace
  // the post so the use stays open until the stop.
  const stopController = new AbortController();
  const tool = new ReadChosenFileTool(neverAnsweringBridge);
  const pendingUse = tool.execute({}, stopController.signal, TURN_CONTEXT);
  stopController.abort();
  await assert.rejects(pendingUse, /was stopped/);
  const cancelRequest = sentRequests.find((request) => request.op === "cancelHostToolCall");
  assert.ok(cancelRequest, "the host is told to close its sheet");
  assert.equal((cancelRequest.payload as { callId: string }).callId, sentRequests[0]?.callId);
});
