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

test("should_post_call_envelopes_with_the_op_and_payload", async () => {
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
    skill: "chat",
    messages: [],
  });
  window.__thintalkBridge?.resolve(posted[0].callId, null);
  await pending;
});

test("should_increment_call_ids_across_calls_and_carry_one_page_tag", async () => {
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

test("should_reject_with_the_hosts_message", async () => {
  const { bridge, reject, posted } = harness();
  const pending = bridge.load("missing");
  reject(posted[0].callId, "unreadable document");
  await assert.rejects(pending, (error: Error) => {
    assert.match(error.message, /unreadable document/);
    return true;
  });
});

test("should_time_out_when_the_host_never_answers", async () => {
  const shortCallTimeoutMilliseconds = 20;
  const bridge = new SessionBridge(() => undefined, shortCallTimeoutMilliseconds);
  await assert.rejects(bridge.list(), SessionBridgeTimeoutError);
});
