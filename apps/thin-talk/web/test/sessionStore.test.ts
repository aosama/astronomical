import assert from "node:assert/strict";
import { test } from "node:test";
import { ChatMessage } from "../src/ChatMessage";
import { ChatRole } from "../src/ChatRole";
import { SessionStore } from "../src/SessionStore";
import type { SessionBridgeRequest } from "../src/SessionBridgeRequest";
import { scriptedBridge } from "./support/scriptedBridge";

test("should_keep_the_current_conversation_when_another_cannot_be_opened", async () => {
  const sentRequests: SessionBridgeRequest[] = [];
  const store = new SessionStore(scriptedBridge({ load: null, save: null, list: [] }, sentRequests));
  const currentConversation = store.startNew();
  assert.equal(await store.open("missing-conversation"), null);
  assert.equal(store.activeSessionID(), currentConversation.id, "the open conversation stays active");
  store.stageMessages([new ChatMessage({ id: "m1", role: ChatRole.USER, content: "Still here?" })]);
  await store.saveNow();
  const savedConversation = sentRequests.find((request) => request.op === "save")?.payload as { id: string } | undefined;
  assert.equal(savedConversation?.id, currentConversation.id, "later turns are still saved");
});

test("should_switch_to_a_conversation_that_opens", async () => {
  const savedConversation = { id: "saved-1", title: "Verona", createdAt: "a", updatedAt: "b", messages: [] };
  const store = new SessionStore(scriptedBridge({ load: savedConversation }));
  store.startNew();
  assert.equal((await store.open("saved-1"))?.id, "saved-1");
  assert.equal(store.activeSessionID(), "saved-1");
});

test("should_keep_the_current_conversation_when_the_host_fails", async () => {
  const store = new SessionStore(scriptedBridge({}));
  const currentConversation = store.startNew();
  assert.equal(await store.open("any"), null);
  assert.equal(store.activeSessionID(), currentConversation.id);
});
