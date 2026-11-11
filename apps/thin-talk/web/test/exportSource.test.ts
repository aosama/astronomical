import assert from "node:assert/strict";
import { test } from "node:test";
import { isExportableMessage } from "../src/ExportSource";
import { TranscriptMessage } from "../src/TranscriptMessage";
import { ChatRole } from "../src/ChatRole";
import { ChatMessageState } from "../src/ChatMessageState";

function message(role: ChatRole, state: ChatMessageState): TranscriptMessage {
  return new TranscriptMessage({ id: `${role}-${state}`, role, markdown: "text", reasoning: "", state });
}

test("should_export_finished_questions_and_answers_and_answers_the_person_stopped", () => {
  assert.equal(isExportableMessage(message(ChatRole.USER, ChatMessageState.COMPLETE)), true);
  assert.equal(isExportableMessage(message(ChatRole.ASSISTANT, ChatMessageState.COMPLETE)), true);
  assert.equal(isExportableMessage(message(ChatRole.ASSISTANT, ChatMessageState.STOPPED)), true, "what arrived is what they read");
});

test("should_leave_out_replies_still_arriving_and_questions_never_sent", () => {
  assert.equal(isExportableMessage(message(ChatRole.ASSISTANT, ChatMessageState.STREAMING)), false);
  assert.equal(isExportableMessage(message(ChatRole.ASSISTANT, ChatMessageState.FAILED)), false);
  assert.equal(isExportableMessage(message(ChatRole.USER, ChatMessageState.QUEUED)), false);
  assert.equal(isExportableMessage(message(ChatRole.USER, ChatMessageState.FAILED)), false);
});
