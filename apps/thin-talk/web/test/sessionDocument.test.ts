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
