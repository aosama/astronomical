import assert from "node:assert/strict";
import { test } from "node:test";
import { SessionBridge } from "../src/SessionBridge";
import { SessionBridgeTimeoutError } from "../src/SessionBridgeError";
import { SessionDocument } from "../src/SessionDocument";

function harness(): {
  bridge: SessionBridge;
  posted: { callId: string; op: string; payload: unknown }[];
  resolve: (callId: string, payload: unknown) => void;
  reject: (callId: string, message: string) => void;
} {
  const posted: { callId: string; op: string; payload: unknown }[] = [];
  const bridge = new SessionBridge((request) => {
    posted.push({ callId: request.callId, op: request.op, payload: request.payload });
  });
  return {
    bridge,
    posted,
    resolve: (callId, payload) => window.__thintalkBridge?.resolve(callId, payload),
    reject: (callId, message) => window.__thintalkBridge?.reject(callId, message),
  };
}

test("session bridge posts call envelopes with the op and payload", async () => {
  const { bridge, posted } = harness();
  const document = new SessionDocument({
    id: "abc",
    title: "t",
    createdAt: "c",
    updatedAt: "u",
    messages: [],
  });
  const pending = bridge.save(document);
  assert.equal(posted.length, 1);
  assert.match(posted[0].callId, /^call-[0-9a-z]+-1$/);
  assert.equal(posted[0].op, "save");
  assert.deepEqual(posted[0].payload, {
    schemaVersion: document.schemaVersion,
    id: "abc",
    title: "t",
    createdAt: "c",
    updatedAt: "u",
    messages: [],
  });
  window.__thintalkBridge?.resolve(posted[0].callId, null);
  await pending;
});

test("session bridge call ids increment across calls and carry one page tag", async () => {
  const { bridge, posted } = harness();
  const first = bridge.list();
  const second = bridge.list();
  const [firstTag, firstNumber] = posted[0].callId.split("-").slice(1);
  const [secondTag, secondNumber] = posted[1].callId.split("-").slice(1);
  assert.equal(firstTag, secondTag);
  assert.equal(Number(secondNumber), Number(firstNumber) + 1);
  window.__thintalkBridge?.resolve(posted[0].callId, []);
  window.__thintalkBridge?.resolve(posted[1].callId, []);
  await Promise.all([first, second]);
});

test("should_give_two_pages_different_call_ids_so_late_answers_cannot_cross", () => {
  const firstPage = harness();
  void firstPage.bridge.list().catch(() => undefined);
  const secondPage = harness();
  void secondPage.bridge.list().catch(() => undefined);
  assert.notEqual(firstPage.posted[0].callId, secondPage.posted[0].callId);
});

test("session bridge rejects with the host's message", async () => {
  const { bridge, reject, posted } = harness();
  const pending = bridge.load("missing");
  reject(posted[0].callId, "unreadable document");
  await assert.rejects(pending, (error: Error) => {
    assert.match(error.message, /unreadable document/);
    return true;
  });
});

test("session bridge times out when the host never answers", async () => {
  const { bridge } = harness();
  const pending = bridge.list();
  await assert.rejects(pending, SessionBridgeTimeoutError);
}, { timeout: 15_000 });
