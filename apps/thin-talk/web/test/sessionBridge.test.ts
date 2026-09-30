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

test("session bridge posts call-N envelopes with the op and payload", async () => {
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
  assert.equal(posted[0].callId, "call-1");
  assert.equal(posted[0].op, "save");
  assert.deepEqual(posted[0].payload, {
    schemaVersion: document.schemaVersion,
    id: "abc",
    title: "t",
    createdAt: "c",
    updatedAt: "u",
    messages: [],
  });
  window.__thintalkBridge?.resolve("call-1", null);
  await pending;
});

test("session bridge call ids increment across calls", async () => {
  const { bridge, posted } = harness();
  const first = bridge.list();
  const second = bridge.list();
  assert.equal(posted[0].callId, "call-1");
  assert.equal(posted[1].callId, "call-2");
  window.__thintalkBridge?.resolve("call-1", []);
  window.__thintalkBridge?.resolve("call-2", []);
  await Promise.all([first, second]);
});

test("session bridge rejects with the host's message", async () => {
  const { bridge, reject } = harness();
  const pending = bridge.load("missing");
  reject("call-1", "unreadable document");
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
