import assert from "node:assert/strict";
import { test } from "node:test";
import { SessionDocument } from "../src/SessionDocument";

test("session document decodes a plain host document", () => {
  const document = SessionDocument.fromPlain({
    id: "abc-123",
    title: "Romeo and Juliet notes",
    updatedAt: "2026-02-14T10:00:00.000Z",
    messages: [
      { id: "m1", role: "user", content: "Who wrote it?", reasoning: null, state: "complete" },
      { id: "m2", role: "assistant", content: "Shakespeare.", reasoning: "recall", state: "complete" },
    ],
  });
  assert.ok(document);
  assert.equal(document.id, "abc-123");
  assert.equal(document.messages.length, 2);
  assert.equal(document.messages[1].reasoning, "recall");
});

test("session document rejects non-objects and empty ids", () => {
  assert.throws(() => SessionDocument.fromPlain(null));
  assert.throws(() => SessionDocument.fromPlain("nope"));
  assert.throws(() => SessionDocument.fromPlain({ id: "", title: "t", updatedAt: "u", messages: [] }));
  assert.throws(() => SessionDocument.fromPlain({ id: 7, title: "t", updatedAt: "u", messages: [] }));
});

test("session document round-trips through toPlain", () => {
  const document = SessionDocument.fromPlain({
    id: "abc-123",
    title: "t",
    updatedAt: "u",
    messages: [{ id: "m1", role: "user", content: "hi", reasoning: null, state: "complete" }],
  });
  assert.ok(document);
  const plain = document.toPlain();
  const decoded = SessionDocument.fromPlain(plain);
  assert.ok(decoded);
  assert.deepEqual(decoded.toPlain(), plain);
});

test("should_round_trip_tool_steps_in_a_version_2_document", () => {
  const savedPlain = {
    schemaVersion: 2,
    id: "tools-1",
    title: "What time is it?",
    createdAt: "2026-11-07T10:00:00.000Z",
    updatedAt: "2026-11-07T10:00:00.000Z",
    messages: [
      { id: "m1", role: "user", content: "What time is it?", reasoning: "", state: "complete", toolSteps: [] },
      {
        id: "m2", role: "assistant", content: "It is 2 PM.", reasoning: "", state: "complete",
        toolSteps: [{ toolCallID: "call-1", toolName: "get_current_time", label: "Current time", argumentsJSON: "{}", resultText: "2 PM", state: "succeeded", roundIndex: 0, generatedImageName: null }],
      },
    ],
  };
  const reloadedPlain = SessionDocument.fromPlain(savedPlain).toPlain() as typeof savedPlain;
  assert.equal(reloadedPlain.schemaVersion, 2);
  assert.deepEqual(reloadedPlain.messages[1]?.toolSteps, savedPlain.messages[1]?.toolSteps);
});

test("should_load_a_version_1_document_with_no_tool_steps", () => {
  const versionOneDocument = SessionDocument.fromPlain({
    schemaVersion: 1,
    id: "old-1",
    messages: [{ id: "m1", role: "assistant", content: "Hello.", reasoning: "", state: "complete" }],
  });
  assert.deepEqual(versionOneDocument.messages[0]?.toolSteps, []);
});

test("should_drop_only_the_damaged_tool_step", () => {
  const document = SessionDocument.fromPlain({
    id: "mixed-1",
    messages: [{
      id: "m1", role: "assistant", content: "", reasoning: "", state: "complete",
      toolSteps: [{ toolCallID: "", toolName: "x" }, { toolCallID: "call-2", toolName: "get_current_time", state: "failed" }],
    }],
  });
  assert.deepEqual(document.messages[0]?.toolSteps.map((toolStep) => toolStep.toolCallID), ["call-2"]);
});

test("should_keep_a_turn_that_only_called_tools_when_saving", () => {
  const document = SessionDocument.fromPlain({
    id: "steps-only",
    messages: [{
      id: "m1", role: "assistant", content: "", reasoning: "", state: "streaming",
      toolSteps: [{ toolCallID: "call-1", toolName: "get_current_time", state: "running" }],
    }],
  });
  const persistable = SessionDocument.persistableMessages(document.messages);
  assert.equal(persistable.length, 1);
  assert.equal(persistable[0]?.toolSteps.length, 1);
});
